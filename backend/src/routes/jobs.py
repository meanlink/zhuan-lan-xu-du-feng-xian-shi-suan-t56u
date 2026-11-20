from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from src.config.database import get_db
from src.models import Job
from src.schemas.job import JobInput
from src.middlewares.auth import current_user, can_write, admin_only
from src.services import jobs, experiments
from src.services.audit import record
from src.utils.serialization import job_dict
router=APIRouter(prefix='/jobs')
@router.get('')
def listing(db=Depends(get_db),u=Depends(current_user)):
    return [job_dict(j) for j in db.scalars(select(Job).order_by(Job.id.desc()).limit(100))]
@router.post('',status_code=201)
def create(data:JobInput,db=Depends(get_db),u=Depends(can_write)):
    j=jobs.create(db,data,u); record(db,u.email,'job.create',{'id':j.id,'params':j.params})
    db.commit(); jobs.enqueue(j.id); return job_dict(j)
@router.get('/{id}')
def detail(id:int,db=Depends(get_db),u=Depends(current_user)):
    j=db.get(Job,id)
    if not j: raise HTTPException(404,'分析任务不存在')
    return job_dict(j)
@router.get('/{id}/experiment')
def experiment(id:int,db=Depends(get_db),u=Depends(current_user)):
    j=db.get(Job,id)
    if not j: raise HTTPException(404,'分析任务不存在')
    return experiments.payload(db,j)
@router.post('/{id}/retry',status_code=202)
def retry(id:int,db=Depends(get_db),u=Depends(can_write)):
    j=db.get(Job,id)
    if not j: raise HTTPException(404,'分析任务不存在')
    if j.status=='completed': raise HTTPException(409,'该试验已跑齐十轮，无需重跑')
    if not experiments.get_plan(db,j.id): raise HTTPException(409,'该任务不是十轮复核试验，请新建试验')
    from src.worker import retry as requeue
    requeue(j.id); record(db,u.email,'job.retry',{'id':id}); db.commit(); jobs.enqueue(j.id)
    db.expire_all()
    return experiments.payload(db,db.get(Job,id))
@router.post('/{id}/approve')
def approve(id:int,db=Depends(get_db),u=Depends(admin_only)):
    j=db.get(Job,id)
    if not j: raise HTTPException(404,'分析任务不存在')
    if j.status!='completed':raise HTTPException(409,'仅已完成结果可以批准')
    if not experiments.get_plan(db,j.id): raise HTTPException(409,'该结果没有十轮复核方案，不能批准进入新简报')
    if not j.result.get('stability',{}).get('status')=='completed': raise HTTPException(409,'十轮复核未跑齐，不能批准')
    j.approved=True; record(db,u.email,'job.approve',{'id':id}); db.commit()
    return job_dict(j)
