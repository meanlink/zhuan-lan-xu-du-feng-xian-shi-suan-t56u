from sqlalchemy import String, JSON, Double, ForeignKey, Integer, UniqueConstraint, Index
from sqlalchemy.orm import Mapped, mapped_column
from src.config.database import Base


class ExperimentPlan(Base):
    """立项时冻结的复核方案：留出读者、风险线、训练集抽样口径。"""

    __tablename__ = 'experiment_plans'
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey('jobs.id'), unique=True, index=True)
    dataset_checksum: Mapped[str] = mapped_column(String(64))
    threshold: Mapped[float] = mapped_column(Double)
    c_value: Mapped[float] = mapped_column(Double)
    rounds: Mapped[int] = mapped_column(Integer, default=10)
    min_passes: Mapped[int] = mapped_column(Integer, default=8)
    split_seed: Mapped[int] = mapped_column(Integer, default=42)
    stability_seed: Mapped[int] = mapped_column(Integer, default=1008)
    train_indices: Mapped[list] = mapped_column(JSON)
    holdout_indices: Mapped[list] = mapped_column(JSON)
    holdout_readers: Mapped[list] = mapped_column(JSON)
    train_size: Mapped[int] = mapped_column(Integer)
    holdout_size: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[float] = mapped_column(Double)


class ExperimentRound(Base):
    """每一轮有放回抽样复核。job_id + round_index 唯一，重试只计一次。"""

    __tablename__ = 'experiment_rounds'
    __table_args__ = (
        UniqueConstraint('job_id', 'round_index', name='uq_experiment_round'),
        Index('ix_experiment_rounds_job', 'job_id'),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey('jobs.id'))
    round_index: Mapped[int] = mapped_column(Integer)
    fingerprint: Mapped[str] = mapped_column(String(64))
    result: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[float] = mapped_column(Double)
    finished_at: Mapped[float] = mapped_column(Double)
