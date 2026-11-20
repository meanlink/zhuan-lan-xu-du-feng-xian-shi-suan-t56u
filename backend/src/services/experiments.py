"""续读风险试验：立项冻结方案，十轮有放回抽样复核，按轮持久化。"""
import time

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.models import Dataset, ExperimentPlan, ExperimentRound, Job
from src.services import business


def _plan_dict(row):
    return {
        "threshold": row.threshold,
        "c_value": row.c_value,
        "rounds": row.rounds,
        "min_passes": row.min_passes,
        "split_seed": row.split_seed,
        "stability_seed": row.stability_seed,
        "train_indices": row.train_indices,
        "holdout_indices": row.holdout_indices,
        "holdout_readers": row.holdout_readers,
        "train_size": row.train_size,
        "holdout_size": row.holdout_size,
        "dataset_checksum": row.dataset_checksum,
    }


def get_plan(db, job_id):
    return db.scalar(select(ExperimentPlan).where(ExperimentPlan.job_id == job_id))


def freeze_plan(db, dataset, params, job_id):
    """立项时固定留出读者与风险线；同一任务重复调用返回已冻结方案。"""
    existing = get_plan(db, job_id)
    if existing:
        return existing
    plan = business.build_plan(dataset.records, params, dataset.checksum)
    row = ExperimentPlan(
        job_id=job_id,
        dataset_checksum=plan["dataset_checksum"],
        threshold=plan["threshold"],
        c_value=plan["c_value"],
        rounds=plan["rounds"],
        min_passes=plan["min_passes"],
        split_seed=plan["split_seed"],
        stability_seed=plan["stability_seed"],
        train_indices=plan["train_indices"],
        holdout_indices=plan["holdout_indices"],
        holdout_readers=plan["holdout_readers"],
        train_size=plan["train_size"],
        holdout_size=plan["holdout_size"],
        created_at=time.time(),
    )
    db.add(row)
    db.flush()
    return row


def load_rounds(db, job_id):
    rows = list(db.scalars(select(ExperimentRound).where(ExperimentRound.job_id == job_id).order_by(ExperimentRound.round_index)))
    return rows


def save_round(db, job_id, result):
    """按 (job_id, round_index) 唯一落库；并发重试撞键时不重复计数。"""
    row = ExperimentRound(
        job_id=job_id,
        round_index=result["round"],
        fingerprint=result["fingerprint"],
        result=result,
        created_at=time.time(),
        finished_at=time.time(),
    )
    db.add(row)
    try:
        db.commit()
        return True
    except IntegrityError:
        db.rollback()
        return False


def progress(db, job):
    plan_row = get_plan(db, job.id)
    total = plan_row.rounds if plan_row else business.STABILITY_ROUNDS
    rounds = load_rounds(db, job.id) if plan_row else []
    done = sorted(r.round_index for r in rounds)
    missing = [i for i in range(1, total + 1) if i not in done]
    return {
        "status": job.status,
        "rounds_total": total,
        "rounds_completed": len(done),
        "missing_rounds": missing,
        "percent": round(len(done) / total * 100, 1) if total else 0,
        "conclusion_ready": len(done) == total and job.status == "completed",
        "rounds": [{"round": r.round_index, "fingerprint": r.fingerprint, "auc": r.result.get("auc"), "high_risk": r.result.get("high_risk"), "sample_size": r.result.get("sample_size"), "unique_sampled": r.result.get("unique_sampled")} for r in rounds],
    }


def _conclude(db, job, plan_row, rounds):
    plan = _plan_dict(plan_row)
    results = [r.result for r in rounds]
    briefing, stability, metrics, findings = business.conclude(plan, results)
    baseline = business.analyze(db.get(Dataset, job.dataset_id).records, job.params, plan)
    result = dict(baseline)
    result["summary"] = f"{baseline['summary']} {stability['conclusion']}"
    result["metrics"] = baseline["metrics"] + metrics
    result["findings"] = baseline["findings"] + findings
    result["stability"] = stability
    result["briefing_table"] = briefing
    result["progress"] = {"rounds_completed": plan["rounds"], "rounds_total": plan["rounds"], "percent": 100.0, "missing_rounds": []}
    return result


def run(db, job):
    """执行缺失轮次并落库；只从冻结训练集有放回抽样，十轮齐备才出结论。"""
    dataset = db.get(Dataset, job.dataset_id)
    plan_row = get_plan(db, job.id)
    if not plan_row:
        plan_row = freeze_plan(db, dataset, job.params, job.id)
        db.commit()
    plan = _plan_dict(plan_row)
    if job.status == "queued":
        job.status = "running"
        db.commit()

    for index in range(1, plan["rounds"] + 1):
        if db.scalar(select(ExperimentRound.id).where(ExperimentRound.job_id == job.id, ExperimentRound.round_index == index)):
            continue
        fresh = db.get(Job, job.id)
        if fresh.status not in ("queued", "running"):
            return
        result = business.run_round(dataset.records, plan, index)
        save_round(db, job.id, result)
        done_rounds = {r.round_index for r in load_rounds(db, job.id)}
        job = db.get(Job, job.id)
        job.result = {
            "progress": {"rounds_completed": len(done_rounds), "rounds_total": plan["rounds"], "percent": round(len(done_rounds) / plan["rounds"] * 100, 1), "missing_rounds": [i for i in range(1, plan["rounds"] + 1) if i not in done_rounds]},
            "stability": {"status": "running", "rounds_completed": len(done_rounds), "rounds_total": plan["rounds"], "min_passes": plan["min_passes"]},
        }
        db.commit()

    rounds = load_rounds(db, job.id)
    if len(rounds) != plan["rounds"]:
        # 正常情况下不会发生；把缺轮视为可重试失败，而不是带着不完整轮次出结论。
        raise ValueError(f"十轮复核未跑齐（已完成{len(rounds)}轮），请补齐缺失轮次后再出结论。")

    job.result = _conclude(db, job, plan_row, rounds)
    job.status = "completed"
    job.finished_at = time.time()
    db.commit()


def payload(db, job):
    plan_row = get_plan(db, job.id)
    data = {"job_id": job.id, "status": job.status, "plan": None, "progress": {"status": job.status, "rounds_total": business.STABILITY_ROUNDS, "rounds_completed": 0, "missing_rounds": list(range(1, business.STABILITY_ROUNDS + 1)), "percent": 0, "conclusion_ready": False, "rounds": []}, "readers": [], "conclusion": None, "briefing_table": None}
    if not plan_row:
        return data
    plan = _plan_dict(plan_row)
    rounds = load_rounds(db, job.id)
    result = [r.result for r in rounds]
    data["plan"] = {
        "dataset_checksum": plan_row.dataset_checksum,
        "threshold": plan_row.threshold,
        "c_value": plan_row.c_value,
        "rounds": plan_row.rounds,
        "min_passes": plan_row.min_passes,
        "split_seed": plan_row.split_seed,
        "stability_seed": plan_row.stability_seed,
        "train_size": plan_row.train_size,
        "holdout_size": plan_row.holdout_size,
        "holdout_readers": plan_row.holdout_readers,
        "created_at": plan_row.created_at,
        "sampling": f"固定分层75/25留出（种子{plan_row.split_seed}），训练{plan_row.train_size}位、留出{plan_row.holdout_size}位；仅从训练集有放回抽取{plan_row.rounds}轮（种子{plan_row.stability_seed}）；风险线{plan_row.threshold:.2f}、模型C={plan_row.c_value}在立项时冻结。",
    }
    data["progress"] = progress(db, job)
    data["readers"] = business.accumulate(plan, result) if rounds else []
    if job.status == "completed" and len(rounds) == plan_row.rounds:
        data["conclusion"] = job.result.get("stability")
        data["briefing_table"] = job.result.get("briefing_table")
    return data
