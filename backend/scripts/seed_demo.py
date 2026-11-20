import time, json
from sqlalchemy import select, func
from src.config.database import Session
from migrations.initial import migrate
from src.models import User, Dataset, Job, Report, Delivery, Audit, ExperimentPlan, ExperimentRound
from src.services import business, experiments
from src.services.reports import deliver
from src.utils.security import hash_password
from src.utils.serialization import checksum


def _seed_experiment(db, dataset, admin, name, params, approve, report_name):
    """幂等生成一次已跑齐十轮的复核试验（含冻结方案与逐轮结果）。"""
    existing = db.scalar(select(Job).where(Job.name == name))
    if existing:
        return existing
    job = Job(name=name, dataset_id=dataset.id, params=params, status='running', created_by=admin.id, created_at=time.time() - 3600)
    db.add(job)
    db.flush()
    plan_row = experiments.freeze_plan(db, dataset, params, job.id)
    plan = experiments._plan_dict(plan_row)
    rounds = []
    for index in range(1, plan["rounds"] + 1):
        result = business.run_round(dataset.records, plan, index)
        row = ExperimentRound(job_id=job.id, round_index=index, fingerprint=result["fingerprint"], result=result, created_at=time.time() - 3590, finished_at=time.time() - 3589)
        db.add(row)
        rounds.append(row)
    db.flush()
    job.result = experiments._conclude(db, job, plan_row, rounds)
    job.status = 'completed'
    job.approved = approve
    job.finished_at = time.time() - 3585
    db.flush()
    if report_name:
        r = Report(name=report_name, job_id=job.id, destination='业务复核组 · 本地收件箱', interval_seconds=604800, next_run=time.time() + 604800, enabled=True)
        db.add(r)
        db.flush()
        deliver(db, r)
    return job


def seed():
    migrate()
    with Session() as db:
        for role, name in [('admin', '林知 · 业务负责人'), ('analyst', '陈舟 · 分析员'), ('viewer', '周禾 · 观察员')]:
            email = role + '@demo.local'
            if not db.scalar(select(User).where(User.email == email)):
                db.add(User(email=email, name=name, role=role, password=hash_password('Demo123!')))
        db.flush()
        m = business.metadata()
        d = db.scalar(select(Dataset).where(Dataset.name == m['dataset_name']))
        if not d:
            rows = business.seed_records()
            d = Dataset(name=m['dataset_name'], source=m['short'] + '-202611.csv', records=rows, checksum=checksum(rows), created_at=time.time())
            db.add(d)
            db.flush()
        admin = db.scalar(select(User).where(User.role == 'admin'))
        params = {p['key']: p['default'] for p in m.get('parameters', [])}
        _seed_experiment(db, d, admin, '基线复核 · ' + m['name'], params, True, '每日业务复核简报')
        stricter = dict(params, threshold=0.7)
        _seed_experiment(db, d, admin, '收紧风险线 · ' + m['name'], stricter, False, None)
        if not db.scalar(select(Audit.id).where(Audit.action == 'demo.baseline')):
            db.add(Audit(actor='seed', action='demo.baseline', detail={'dataset': d.id}, at=time.time()))
        db.commit()
        counts = {model.__tablename__: db.scalar(select(func.count()).select_from(model)) for model in [User, Dataset, Job, Report, Delivery, Audit, ExperimentPlan, ExperimentRound]}
        print(json.dumps(counts, ensure_ascii=False))
        return counts


if __name__ == '__main__':
    seed()
