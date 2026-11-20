"""专栏续读风险：训练集拟合标准化，独立留出集评估，并有放回抽样复核。"""
import hashlib
import json
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["sessions_14d", "completion_pct", "days_inactive", "issues_read"]
LABELS = ["14日访问次数", "阅读完成率", "连续未读天数", "已读期数"]


STABILITY_ROUNDS = 10
STABILITY_MIN_PASSES = 8
SPLIT_SEED = 42
STABILITY_SEED = 1008


def metadata():
    return {
        "name": "专栏续读风险试算", "short": "reader-risk",
        "tagline": "用行为窗口解释续读风险，再试算编辑关怀的覆盖边界。",
        "client": "纸岸知识出版社 · 6人内容运营组 · 320名匿名年度专栏读者",
        "workflow": "编辑每周合并阅读日志与到期续读结果，在表格中凭访问次数选择关怀对象。",
        "pains": ["各行为量纲不同，人工排序偏向高频用户", "直接使用到期后的数据会泄漏结果", "风险阈值缺少精确率与召回率依据"],
        "upstream": "到期前14日阅读行为快照 reader_window_202611.csv",
        "downstream": "主编据验证集覆盖人数安排内容回访实验，不自动触达读者",
        "entities": ["读者行为样本", "续读风险试验", "编辑复核简报"],
        "dataset_name": "纸岸专栏到期前14日匿名样本",
        "columns": [{"key": "reader_id", "label": "匿名读者", "type": "string"}] + [{"key": k, "label": l, "type": "number"} for k, l in zip(FEATURES, LABELS)] + [{"key": "renewed", "label": "实际续读(1/0)", "type": "number"}],
        "parameters": [{"key": "threshold", "label": "高风险概率阈值", "type": "number", "default": 0.5, "min": 0.1, "max": 0.9}, {"key": "regularization", "label": "模型C（越小正则越强）", "type": "number", "default": 1.0, "min": 0.01, "max": 10}],
        "workbench_title": "续读风险阈值试验", "action_label": "训练并复核覆盖范围", "unit": "位读者",
        "accent": "oklch(53% 0.12 30)", "style": "editorial-press",
        "seed_story": ["08:40 编辑导入 reader_window_202611.csv 的320条匿名样本", "08:45 检查14日访问、完成率、未读天数和已读期数", "09:00 以0.5风险阈值创建一次续读试验，立项即冻结留出读者与风险线", "09:01 系统按随机种子42分层保留25%读者，训练集仅用于有放回抽样", "09:02 后台从训练集有放回抽取十轮，按轮保存进度，中断只补缺失轮", "09:05 十轮齐备后才出结论，至少八轮过线的读者进入新简报", "09:10 将阈值改为0.7时须新建试验，旧试验与旧简报保持原样", "09:20 下载稳定名单明细交主编", "10:00 管理员设置周一08:30内部复核简报"],
        "stability": {"rounds": STABILITY_ROUNDS, "min_passes": STABILITY_MIN_PASSES, "split_seed": SPLIT_SEED, "stability_seed": STABILITY_SEED, "description": "立项时冻结留出读者与风险线；只从训练集有放回抽样十轮；至少八轮过线的读者进入新简报。"},
        "method": "分层75/25固定留出；立项时冻结留出读者与风险线；StandardScaler与L2 logistic回归只在训练集拟合；训练集有放回抽样十轮复核；十轮齐备且≥8轮过线才进入新简报。",
        "limitations": ["模拟读者行为，不能当作真实留存效果", "相关系数与模型系数不代表因果", "稳定性复核降低换批样本的名单波动，但不消除模型误差", "未包含时间外验证或干预效果评估"],
    }


def seed_records():
    rng = np.random.default_rng(1008)
    rows = []
    for i in range(320):
        sessions = int(rng.poisson(6))
        completion = float(np.clip(rng.normal(62, 21), 4, 99))
        inactive = int(rng.integers(0, 15))
        issues = int(rng.integers(1, 25))
        logit = -0.5 - 0.20 * sessions - 0.025 * (completion - 50) + 0.27 * inactive - 0.035 * issues
        risk = 1 / (1 + math.exp(-logit))
        rows.append({"reader_id": f"RD-{i+1:04}", "sessions_14d": sessions, "completion_pct": round(completion, 1), "days_inactive": inactive, "issues_read": issues, "renewed": int(rng.random() >= risk)})
    return rows


def _params(params):
    try:
        threshold = float(params.get("threshold", 0.5))
        c_value = float(params.get("regularization", 1.0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("风险阈值与模型C须为有效数值。") from exc
    if not 0.1 <= threshold <= 0.9 or not 0.01 <= c_value <= 10:
        raise ValueError("风险阈值须为0.1~0.9，模型C须为0.01~10。")
    return threshold, c_value


def _arrays(records):
    if len(records) < 40:
        raise ValueError("至少需要40条完整样本，以保留独立验证集。")
    try:
        x = np.asarray([[float(row[k]) for k in FEATURES] for row in records])
        renewed = np.asarray([float(row["renewed"]) for row in records])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("行为字段与续读标签必须完整且为数值。") from exc
    if not np.isfinite(x).all() or not np.isin(renewed, [0, 1]).all() or (x < 0).any() or (x[:, 1] > 100).any():
        raise ValueError("行为值必须非负且有限，完成率0~100，续读标签只能为0/1。")
    y = 1 - renewed.astype(int)
    counts = np.bincount(y, minlength=2)
    if counts.min() < 8:
        raise ValueError("续读与未续读两类各至少8条样本。")
    ids = [str(row.get("reader_id", f"row-{i + 1}")) for i, row in enumerate(records)]
    if len(set(ids)) != len(ids):
        raise ValueError("读者标识 reader_id 必须唯一，才能按人统计过线次数。")
    return x, y, ids


def build_plan(records, params, dataset_checksum=""):
    """立项时冻结：留出读者、风险线、模型C与复核种子。"""
    threshold, c_value = _params(params)
    x, y, ids = _arrays(records)
    indices = np.arange(len(records))
    train, valid = train_test_split(indices, test_size=0.25, stratify=y, random_state=SPLIT_SEED)
    train = np.sort(train)
    valid = np.sort(valid)
    return {
        "threshold": threshold,
        "c_value": c_value,
        "rounds": STABILITY_ROUNDS,
        "min_passes": STABILITY_MIN_PASSES,
        "split_seed": SPLIT_SEED,
        "stability_seed": STABILITY_SEED,
        "train_indices": [int(v) for v in train],
        "holdout_indices": [int(v) for v in valid],
        "holdout_readers": [ids[int(v)] for v in valid],
        "train_size": int(train.size),
        "holdout_size": int(valid.size),
        "dataset_checksum": dataset_checksum,
    }


def analyze(records, params, plan=None):
    """基线：只用冻结训练集拟合，只在冻结留出读者上评估。"""
    if plan is None:
        plan = build_plan(records, params)
    threshold, c_value = plan["threshold"], plan["c_value"]
    x, y, ids = _arrays(records)
    train = np.asarray(plan["train_indices"], dtype=int)
    valid = np.asarray(plan["holdout_indices"], dtype=int)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, max_iter=1000, random_state=SPLIT_SEED))
    model.fit(x[train], y[train])
    probability = model.predict_proba(x[valid])[:, 1]
    prediction = (probability >= threshold).astype(int)
    auc = float(roc_auc_score(y[valid], probability))
    precision = float(precision_score(y[valid], prediction, zero_division=0))
    recall = float(recall_score(y[valid], prediction, zero_division=0))
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y[valid], prediction, labels=[0, 1]).ravel()]
    coefficients = model[-1].coef_[0]
    order = np.argsort(-probability)
    rows = [{"reader_id": ids[int(valid[j])], "risk_pct": round(float(probability[j]) * 100, 2), "decision": "建议编辑复核" if prediction[j] else "维持观察", "actual": "未续读" if y[valid[j]] else "已续读"} for j in order]
    strongest = int(np.argmax(np.abs(coefficients)))
    table = {"columns": [{"key": "reader_id", "label": "匿名读者"}, {"key": "risk_pct", "label": "风险概率(%)"}, {"key": "decision", "label": "阈值决策"}, {"key": "actual", "label": "留出集实际结果"}], "rows": rows}
    return {
        "summary": f"立项冻结：训练{len(train)}位、留出{len(valid)}位读者，风险线{threshold:.2f}；基线有{int(prediction.sum())}位建议编辑复核。",
        "metrics": [{"label": "验证AUC", "value": round(auc, 4)}, {"label": "风险召回率", "value": round(recall * 100, 2), "unit": "%"}, {"label": "风险精确率", "value": round(precision * 100, 2), "unit": "%"}, {"label": "验证集高风险", "value": int(prediction.sum()), "unit": "人"}],
        "table": table,
        "baseline_table": table,
        "chart": {"type": "bar", "labels": LABELS, "series": [{"name": "标准化log-odds系数", "values": [round(float(v), 4) for v in coefficients]}]},
        "findings": [f"验证混淆矩阵：命中未续读{tp}、误报{fp}、漏报{fn}、正确排除{tn}。", f"绝对系数最大的特征为{LABELS[strongest]}（{float(coefficients[strongest]):.3f}）；正值意味着更高未续读风险。", f"验证准确率{accuracy_score(y[valid], prediction)*100:.1f}%；阈值变更影响覆盖边界，不改变AUC。"],
        "method": metadata()["method"], "warnings": ["风险排序供编辑复核使用，不能据此断言某项行为导致未续读。"],
    }


def round_fingerprint(plan, round_index):
    payload = {
        "round": int(round_index),
        "threshold": plan["threshold"],
        "c_value": plan["c_value"],
        "split_seed": plan["split_seed"],
        "stability_seed": plan["stability_seed"],
        "train_indices": plan["train_indices"],
        "holdout_indices": plan["holdout_indices"],
        "dataset_checksum": plan.get("dataset_checksum", ""),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def run_round(records, plan, round_index):
    """只从冻结训练集有放回抽样一轮，评估冻结留出读者是否过风险线。"""
    x, y, ids = _arrays(records)
    train = np.asarray(plan["train_indices"], dtype=int)
    valid = np.asarray(plan["holdout_indices"], dtype=int)
    threshold, c_value = plan["threshold"], plan["c_value"]
    rng = np.random.default_rng([int(plan["stability_seed"]), int(round_index)])
    sampled = None
    for _ in range(100):
        draw = rng.integers(0, train.size, size=train.size)
        candidate = train[draw]
        if np.unique(y[candidate]).size == 2:
            sampled = candidate
            break
    if sampled is None:
        raise ValueError(f"第{round_index}轮有放回抽样未同时覆盖两类标签，无法拟合。")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, max_iter=1000, random_state=SPLIT_SEED))
    model.fit(x[sampled], y[sampled])
    probability = model.predict_proba(x[valid])[:, 1]
    passed = probability >= threshold
    auc = float(roc_auc_score(y[valid], probability))
    readers = [{"reader_id": ids[int(valid[j])], "risk_pct": round(float(probability[j]) * 100, 2), "passed": bool(passed[j])} for j in range(valid.size)]
    return {
        "round": int(round_index),
        "fingerprint": round_fingerprint(plan, round_index),
        "sample_size": int(sampled.size),
        "unique_sampled": int(np.unique(sampled).size),
        "threshold": threshold,
        "auc": round(auc, 4),
        "high_risk": int(passed.sum()),
        "readers": readers,
    }


def _round_summary(round_result):
    return {
        "round": round_result["round"],
        "fingerprint": round_result["fingerprint"],
        "auc": round_result["auc"],
        "high_risk": round_result["high_risk"],
        "sample_size": round_result["sample_size"],
        "unique_sampled": round_result["unique_sampled"],
    }


def accumulate(plan, round_results):
    """按冻结留出读者累计已完成轮次的过线次数与平均风险。"""
    passes = {rid: 0 for rid in plan["holdout_readers"]}
    risk_sum = {rid: 0.0 for rid in plan["holdout_readers"]}
    for result in round_results:
        for entry in result["readers"]:
            passes[entry["reader_id"]] += 1 if entry["passed"] else 0
            risk_sum[entry["reader_id"]] += float(entry["risk_pct"])
    evaluated = len(round_results)
    rows = []
    for rid in plan["holdout_readers"]:
        rows.append({
            "reader_id": rid,
            "passes": passes[rid],
            "evaluated_rounds": evaluated,
            "pass_label": f"{passes[rid]}/{plan['rounds']} 轮",
            "mean_risk_pct": round(risk_sum[rid] / evaluated, 2) if evaluated else None,
            "enters_brief": passes[rid] >= plan["min_passes"] if evaluated == plan["rounds"] else None,
            "decision": ("进入新简报" if passes[rid] >= plan["min_passes"] else "未达门槛") if evaluated == plan["rounds"] else "复核未跑齐",
        })
    rows.sort(key=lambda r: (-r["passes"], -(r["mean_risk_pct"] or 0), r["reader_id"]))
    return rows


def aggregate_rounds(plan, round_results):
    """按人汇总十轮过线次数；轮次不齐时不产生结论。"""
    if len(round_results) != plan["rounds"]:
        raise ValueError("复核轮次未跑齐，不能生成结论。")
    indexes = sorted(int(r["round"]) for r in round_results)
    if indexes != list(range(1, plan["rounds"] + 1)):
        raise ValueError("复核轮次不连续，不能生成结论。")
    rows = accumulate(plan, round_results)
    histogram = [0] * (plan["rounds"] + 1)
    for row in rows:
        histogram[row["passes"]] += 1
        row["decision"] = "进入新简报" if row["enters_brief"] else "未达门槛"
    return rows, histogram


def conclude(plan, round_results):
    readers, histogram = aggregate_rounds(plan, round_results)
    stable = [r for r in readers if r["enters_brief"]]
    threshold = plan["threshold"]
    briefing_rows = [{"reader_id": r["reader_id"], "passes": r["passes"], "mean_risk_pct": r["mean_risk_pct"], "decision": r["decision"]} for r in stable]
    briefing = {"columns": [{"key": "reader_id", "label": "匿名读者"}, {"key": "passes", "label": "过线轮次"}, {"key": "mean_risk_pct", "label": "平均风险概率(%)"}, {"key": "decision", "label": "新简报决定"}], "rows": briefing_rows}
    stability = {
        "status": "completed",
        "rounds_total": plan["rounds"],
        "rounds_completed": plan["rounds"],
        "min_passes": plan["min_passes"],
        "holdout_readers": len(readers),
        "stable_readers": len(stable),
        "missing_rounds": [],
        "round_summary": [_round_summary(r) for r in sorted(round_results, key=lambda r: r["round"])],
        "pass_histogram": {"labels": [f"{i}轮" for i in range(plan["rounds"] + 1)], "values": histogram},
        "readers": readers,
        "briefing_table": briefing,
        "conclusion": f"十轮有放回抽样复核已跑齐：{len(readers)}位冻结留出读者中，{len(stable)}位至少{plan['min_passes']}轮过线（风险线{threshold:.2f}），进入新简报；其余{len(readers) - len(stable)}位未达门槛，留在观察名单。",
    }
    metrics = [{"label": "稳定进入新简报", "value": len(stable), "unit": "人"}, {"label": "过线门槛", "value": f"≥{plan['min_passes']}/{plan['rounds']}轮"}]
    findings = [
        "十轮复核的过线次数分布：" + "、".join(f"{i}轮{histogram[i]}人" for i in range(plan["rounds"] + 1) if histogram[i]) + "。",
        f"只有至少{plan['min_passes']}轮过线的{len(stable)}位读者进入新简报；旧简报与既有归档保持原样，不会被本轮结果改写。",
        "抽样只从训练集有放回进行，冻结留出读者始终不参与拟合。",
    ]
    return briefing, stability, metrics, findings


