import time, logging
from pathlib import Path
from sqlalchemy import select, update
from redis import Redis
from src.config.database import Session
from src.config.settings import REDIS_URL
from src.models import Job, Report
from src.services import experiments
from src.services.reports import deliver
from src.services.audit import record
logging.basicConfig(level=logging.INFO)


def execute(job_id):
    with Session() as db:
        # 原子领取：并发重试时只有一个执行者进入，其余直接退出。
        claimed = db.execute(
            update(Job).where(Job.id == job_id, Job.status == 'queued').values(status='running')
        ).rowcount
        db.commit()
        if not claimed:
            return
        j = db.get(Job, job_id)
        try:
            experiments.run(db, j)
            j = db.get(Job, job_id)
            if j.status == 'completed':
                record(db, 'worker', 'job.completed', {'id': j.id})
                db.commit()
        except Exception as e:
            logging.exception('Analysis failed')
            db.rollback()
            j = db.get(Job, job_id)
            if j:
                # 已完成的轮次保留，重试时只补缺失轮次。
                j.status = 'failed'
                j.error = str(e)[:990]
                record(db, 'worker', 'job.failed', {'id': j.id, 'error': j.error})
                db.commit()


def retry(job_id):
    """把失败或未跑齐的试验重新排队；已落库轮次不会被覆盖。"""
    with Session() as db:
        j = db.get(Job, job_id)
        if not j or j.status == 'completed':
            return None
        j.status = 'queued'
        j.error = ''
        db.commit()
        return j.id


def with_progress(db):
    """轮询补跑：排队但缺轮的试验由 SQL 状态兜底，不依赖 Redis 消息是否送达。"""
    return list(db.scalars(select(Job.id).where(Job.status.in_(('queued', 'running'))).limit(3)))


def tick():
    with Session() as db:
        queued = with_progress(db)
    for job_id in queued:
        execute(job_id)
    with Session() as db:
        for r in db.scalars(select(Report).where(Report.enabled == True, Report.next_run <= time.time())):
            d = deliver(db, r)
            if d:
                record(db, 'worker', 'report.delivered', {'id': r.id, 'delivery': d.id})
        db.commit()


def main():
    r = Redis.from_url(REDIS_URL, socket_timeout=3)
    # This Compose stack has a single worker. A crash must not strand its jobs;
    # completed rounds are persisted, so re-queuing only fills the gap.
    with Session() as db:
        for j in db.scalars(select(Job).where(Job.status == 'running')):
            j.status = 'queued'
            record(db, 'worker', 'job.recovered', {'id': j.id})
        # A worker may die after claiming a job but before running it; queued
        # jobs are picked up again from their persisted rounds.
        for j in db.scalars(select(Job).where(Job.status == 'queued')):
            try:
                r.rpush('analysis:queue', j.id)
            except Exception:
                logging.exception('Requeue failed')
        db.commit()
    while True:
        Path('/tmp/heartbeat').touch()
        try:
            item = r.blpop('analysis:queue', timeout=1)
            if item:
                execute(int(item[1]))
            tick()
        except Exception:
            logging.exception('Worker retry')
            time.sleep(2)


if __name__ == '__main__':
    main()
