import time
from fastapi import HTTPException
from redis import Redis
from src.config.settings import REDIS_URL
from src.models import Dataset, Job
from src.services import business, experiments
from src.services.business import metadata


def create(db, data, user):
    dataset = db.get(Dataset, data.dataset_id)
    if not dataset:
        raise HTTPException(404, '数据批次不存在')
    allowed = {p['key']: p for p in metadata().get('parameters', [])}
    if set(data.params) - set(allowed):
        raise HTTPException(422, '包含未知分析参数')
    params = {k: p['default'] for k, p in allowed.items()}
    params.update(data.params)
    for key, p in allowed.items():
        v = params[key]
        if p['type'] == 'number':
            try:
                v = float(v)
                if not (p.get('min', -1e9) <= v <= p.get('max', 1e9)):
                    raise ValueError()
                params[key] = v
            except (ValueError, TypeError):
                raise HTTPException(422, f'{p["label"]} 超出范围或非数值')
        elif str(v) not in [str(o['value']) for o in p.get('options', [])]:
            raise HTTPException(422, f'{p["label"]} 不是有效选项')
    try:
        # 立项即冻结留出读者与风险线，后续轮次只在此方案内抽样。
        business.build_plan(dataset.records, params, dataset.checksum)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    j = Job(name=data.name, dataset_id=data.dataset_id, params=params, created_by=user.id, created_at=time.time(), status='queued')
    db.add(j)
    db.flush()
    experiments.freeze_plan(db, dataset, params, j.id)
    j.result = experiments.payload(db, j)['progress']
    db.flush()
    return j


def enqueue(job_id):
    try:
        r = Redis.from_url(REDIS_URL)
        r.rpush('analysis:queue', job_id)
        r.close()
    except Exception:
        pass  # SQL queued status is the durable fallback
