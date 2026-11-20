import {jobs} from '../api/entities.js';
import {workspace} from '../stores/workspace.js';
import {session} from '../stores/session.js';
import {heading} from '../components/common/shell.js';
import {table} from '../components/common/table.js';
import {status} from '../components/common/status.js';
import {esc,fmt,date,notify} from '../utils/dom.js';

let poller=null,rerender=null;
export function stopExperimentPoll(){if(poller){clearInterval(poller);poller=null;}}

function progressBar(p){
 const total=p.rounds_total||10,done=p.rounds_completed||0,missing=p.missing_rounds||[];
 return '<section class="stability-progress '+(p.conclusion_ready?'ready':'running')+'"><div class="progress-head"><div><p class="eyebrow">ROUND PROGRESS</p><h2>'+(p.conclusion_ready?'十轮复核已跑齐':'复核正在进行 · 暂不出结论')+'</h2></div><span class="progress-count">'+done+' / '+total+' 轮</span></div><div class="progress-track" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="'+(p.percent||0)+'"><span style="width:'+(p.percent||0)+'%"></span></div><p class="hint">'+(p.conclusion_ready?'全部轮次落库，结论与稳定名单已生成。':'未跑齐前只显示进度；中断后可从缺失轮次恢复，已完成轮次不会重复计数。')+(missing.length?' 待补轮次：'+missing.join('、')+'。':'')+'</p></section>';
}

function planPanel(plan,progress){
 if(!plan)return '<section class="panel"><div class="section-title"><h2>抽样依据</h2><span>方案未冻结</span></div><p class="empty">该任务尚未冻结复核方案。请从工作台重新创建试验。</p></section>';
 return '<section class="panel"><div class="section-title"><h2>抽样依据（立项时固定）</h2><span>新增批次不会改变本试验</span></div><div class="fact-grid"><div><span>风险线</span><strong>'+esc(fmt(plan.threshold))+'</strong></div><div><span>模型C</span><strong>'+esc(fmt(plan.c_value))+'</strong></div><div><span>固定留出读者</span><strong>'+plan.holdout_size+' 位</strong></div><div><span>训练读者</span><strong>'+plan.train_size+' 位</strong></div><div><span>分层种子</span><strong>'+plan.split_seed+'</strong></div><div><span>抽样种子</span><strong>'+plan.stability_seed+'</strong></div><div><span>复核轮次</span><strong>'+plan.rounds+' 轮</strong></div><div><span>过线门槛</span><strong>≥'+plan.min_passes+'/'+plan.rounds+' 轮</strong></div></div><p class="method-note">'+esc(plan.sampling)+'</p><p class="hint">输入快照 SHA256 '+esc(plan.dataset_checksum.slice(0,24))+'… · 立项 '+date(plan.created_at)+' · 当前进度 '+progress.rounds_completed+'/'+progress.rounds_total+' 轮。</p></section>';
}

function roundTable(progress){
 const rows=(progress.rounds||[]).map(r=>({round:r.round,fingerprint:r.fingerprint.slice(0,12),auc:r.auc,high_risk:r.high_risk,sample_size:r.sample_size,unique_sampled:r.unique_sampled}));
 const cols=[{key:'round',label:'轮次'},{key:'fingerprint',label:'抽样指纹'},{key:'auc',label:'留出AUC'},{key:'high_risk',label:'过线人数'},{key:'unique_sampled',label:'去重样本'}];
 return '<section class="panel"><div class="section-title"><h2>逐轮记录</h2><span>'+(progress.rounds||[]).length+' / '+progress.rounds_total+' 轮已落库</span></div>'+table(cols,rows,'本轮尚未完成，暂无逐轮记录。')+'</section>';
}

function readerTable(readers,plan){
 const cols=[{key:'reader_id',label:'匿名读者'},{key:'pass_label',label:'过线次数'},{key:'mean_risk_pct',label:'平均风险概率(%)'},{key:'decision',label:'复核决定'}];
 return '<section class="panel"><div class="section-title"><h2>每位留出读者的过线次数</h2><span>共 '+readers.length+' 位固定留出读者</span></div>'+table(cols,readers,'方案冻结后，逐轮累计会显示在这里。')+'<p class="hint">过线次数 = 十轮有放回抽样中，该读者被模型判为高风险的概率达到风险线（'+esc(fmt(plan?.threshold))+'）的轮数。只有至少 '+esc(plan?.min_passes||8)+' 轮的读者进入新简报。</p></section>';
}

function conclusionPanel(e){
 const c=e.conclusion;
 if(!c)return '<section class="panel stability-note"><div class="section-title"><h2>结论</h2><span>等待十轮跑齐</span></div><p class="empty">复核未跑齐，系统不生成结论，也不产生新简报名单。已完成的轮次会保留，中断后只补缺失轮次。</p></section>';
 return '<section class="stability-conclusion"><p class="eyebrow">CONCLUSION</p><h2>'+esc(c.conclusion)+'</h2><div class="metrics">'+[['稳定进入新简报',c.stable_readers+' 人'],['固定留出读者',c.holdout_readers+' 位'],['过线门槛','≥'+c.min_passes+'/'+c.rounds_total+' 轮']].map(([k,v])=>'<div><span>'+k+'</span><strong>'+esc(v)+'</strong></div>').join('')+'</div></section><section class="panel"><div class="section-title"><h2>新简报名单快照</h2><span>仅收录过线读者 · 旧简报保持原样</span></div>'+table(e.briefing_table?.columns||[],e.briefing_table?.rows||[],'没有读者达到过线门槛。')+'</section>';
}

export async function experimentPage(id){
 stopExperimentPoll();
 workspace.jobs=await jobs.list();
 const j=id?await jobs.get(id):(workspace.jobs.find(x=>x.id===workspace.selectedJob)||workspace.jobs[0]);
 if(!j)return heading('06 / STABILITY CHECK','十轮稳定性复核','尚无分析试验。')+'<div class="empty">先到分析工作台创建一个续读风险试验。<a href="#/workbench">前往工作台 ↗</a></div>';
 workspace.selectedJob=j.id;
 const e=await jobs.experiment(j.id);
 const p=e.progress;
 const canRetry=session.user.role!=='viewer'&&j.status!=='completed'&&e.plan;
 const selector='<div class="result-selector"><label>试验记录<select id="select-experiment">'+workspace.jobs.map(x=>'<option value="'+x.id+'" '+(x.id===j.id?'selected':'')+'>'+esc(x.name)+' · '+esc(x.status)+'</option>').join('')+'</select></label>'+status(j.status)+'<span>'+date(j.created_at)+'</span><span class="approval">'+(j.approved?'已批准进入简报':'未批准')+'</span></div>';
 const actions='<button class="button" id="refresh-experiment">刷新进度 ↻</button>'+(canRetry?'<button class="button primary" id="retry-experiment">补齐缺失轮次 ↗</button>':'');
 const html=heading('06 / STABILITY CHECK','十轮稳定性复核 · '+j.name,'立项时固定留出读者与风险线，只从训练集有放回抽十轮。未跑齐只显示进度，十轮齐备才出结论。',actions)+selector+progressBar(p)+'<div class="two-columns"><div>'+planPanel(e.plan,p)+'</div><div>'+conclusionPanel(e)+'</div></div>'+roundTable(p)+readerTable(e.readers,e.plan);
 if(j.status!=='completed'&&e.plan){const seen=(j.result?.progress?.rounds_completed||0);poller=setInterval(async()=>{try{const fresh=await jobs.get(j.id);if(fresh.status==='completed'||(fresh.result?.progress?.rounds_completed||0)!==seen){stopExperimentPoll();if(rerender)await rerender();}}catch(error){}},4000);}
 return html;
}

export function bindExperiment(refresh){
 rerender=refresh;
 document.querySelector('#select-experiment')?.addEventListener('change',e=>{workspace.selectedJob=Number(e.target.value);location.hash='#/experiment/'+e.target.value;});
 document.querySelector('#refresh-experiment')?.addEventListener('click',refresh);
 document.querySelector('#retry-experiment')?.addEventListener('click',async()=>{try{await jobs.retry(workspace.selectedJob);notify('已补排队，后台只补缺失轮次');await refresh();}catch(e){notify(e.message,true);}});
}
