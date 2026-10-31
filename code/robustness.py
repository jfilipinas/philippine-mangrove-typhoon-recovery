"""Fixed-model endpoint and baseline checks; no new test-driven tuning."""
import json
from pathlib import Path
import pandas as pd,numpy as np
from replicated_analysis import fit,OPT,RAD,metrics,macro
R=Path(__file__).resolve().parents[1];D=R/'data'
def weightedmedian(a):
 a=a.copy().sort_values('ndvi_deficit_2023');a['w']=1/a.site.map(a.site.value_counts());return float(a.loc[a.w.cumsum()>=a.w.sum()/2,'ndvi_deficit_2023'].iloc[0])
if __name__=='__main__':
 df=pd.read_csv(D/'analysis_table_primary.csv');tr=df[~df.site.str.startswith('palawan')];p=pd.read_csv(D/'primary_predictions.csv');bm=weightedmedian(tr);p['balanced_median']=bm;p.to_csv(D/'primary_predictions.csv',index=False);rows=[]
 for site,a in p.groupby('site'):rows.append({'landscape':site,'model':'balanced_median','n':len(a),**metrics(a.observed,a.balanced_median)})
 pd.DataFrame(rows).to_csv(D/'balanced_baseline_metrics.csv',index=False)
 obs=pd.read_csv(D/'cell_observations.csv');obs['date']=pd.to_datetime(obs.date);obs['year']=obs.date.dt.year;obs['month']=obs.date.dt.month;obs=obs[obs.month>=10];mm=[]
 for cell,a in obs.groupby('cell'):
  pre=a[a.date<'2021-12-16'].groupby(['year','month']).ndvi.median().groupby('month').mean();post=a[a.year==2023].groupby('month').ndvi.median();ix=pre.dropna().index.intersection(post.dropna().index)
  if len(ix)>=2:mm.append({'cell':cell,'matched_deficit':float((pre.loc[ix]-post.loc[ix]).mean())})
 d=df.merge(pd.DataFrame(mm),on='cell');tr=d[~d.site.str.startswith('palawan')];te=d[d.site.isin(['palawan_02','palawan_03','palawan_04'])];s=json.load(open(D/'primary_status.json'))['selected'];spec=(*s,OPT if s[0]=='optical' else RAD if s[0]=='radar' else OPT+RAD);m=fit(spec,tr,'matched_deficit');te=te.copy();te['pred']=m.predict(te[spec[3]]);te['median']=tr.matched_deficit.median();te.to_csv(D/'matched_month_predictions.csv',index=False)
 summ={'balanced_training_median':bm,'primary_balanced_median_macro_mae':float(pd.DataFrame(rows).query("landscape != 'palawan_01'").mae.mean()),'matched_month_train_n':len(tr),'matched_month_test_n':len(te),'matched_month_selected_macro_mae':macro(te,te.matched_deficit,te.pred),'matched_month_median_macro_mae':macro(te,te.matched_deficit,te['median']),'matched_month_model':s,'interpretation':'same development-selected family and parameter, refitted to alternate target; no test-driven model selection'}
 (D/'robustness_status.json').write_text(json.dumps(summ,indent=2));print(json.dumps(summ,indent=2))
