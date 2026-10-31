"""Landscape-grouped model selection, nested development assessment and transfer."""
import json,sys,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error
from analyse import observations,features,pipe,metrics,SEED
R=Path(__file__).resolve().parents[1];D=R/'data'
OPT=[v+s for v in ['ndvi','ndmi'] for s in ['_base','_base_sd','_early','_last','_slope']]
RAD=[v+s for v in ['vv','vh'] for s in ['_base','_base_sd','_early','_last','_slope']]
SPECS=[(f,n,p,c) for f,c in [('optical',OPT),('radar',RAD),('fused',OPT+RAD)] for n,pars in [('ridge',[1,10,100]),('forest',[5,15])] for p in pars]
def fit(spec,df,target):
 f,n,p,c=spec;m=pipe(n,p);w=1/df.site.map(df.site.value_counts()).to_numpy();w=w/w.mean()
 m.fit(df[c],df[target],**{m.steps[-1][0]+'__sample_weight':w});return m
def macro(df,y,p):
 t=pd.DataFrame({'site':df.site.to_numpy(),'error':np.abs(np.asarray(y)-np.asarray(p))});return float(t.groupby('site').error.mean().mean())
def select(df,target):
 out=[];folds=list(LeaveOneGroupOut().split(df,groups=df.site))
 for spec in SPECS:
  op=np.full(len(df),np.nan)
  for tr,va in folds:
   m=fit(spec,df.iloc[tr],target);op[va]=m.predict(df.iloc[va][spec[3]])
  out.append((macro(df,df[target],op),spec,op))
 out.sort(key=lambda r:(r[0],r[1][:3]));return out[0],out
def run(df,target='ndvi_deficit_2023',prefix='primary',nested=True):
 df=df.dropna(subset=[target]).reset_index(drop=True);train=df[~df.site.str.startswith('palawan')].reset_index(drop=True);test=df[df.site.isin(['palawan_02','palawan_03','palawan_04'])].reset_index(drop=True);legacy=df[df.site=='palawan_01'].reset_index(drop=True)
 if train.site.nunique()<4 or test.site.nunique()<2:
  s={'status':'insufficient_landscape_coverage','train_landscapes':train.site.nunique(),'test_landscapes':test.site.nunique(),'train_n':len(train),'test_n':len(test)};(D/f'{prefix}_status.json').write_text(json.dumps(s,indent=2));return s
 print('selecting',prefix,len(train),len(test),flush=True);best,allr=select(train,target);score,spec,op=best;m=fit(spec,train,target)
 pd.DataFrame([{'features':s[0],'model':s[1],'parameter':s[2],'selection_macro_mae':sc} for sc,s,_ in allr]).to_csv(D/f'{prefix}_tuning.csv',index=False)
 preds=[]
 for role,a in [('new_transfer',test),('legacy_diagnostic',legacy)]:
  if not len(a):continue
  a=a.copy();a['observed']=a[target];a['selected']=m.predict(a[spec[3]]);a['median']=np.median(train[target]);
  if prefix=='primary':a['persistence']=a.persistence
  a['role']=role;preds.append(a)
 pred=pd.concat(preds,ignore_index=True);pred.to_csv(D/f'{prefix}_predictions.csv',index=False)
 rows=[]
 for (role,site),a in pred.groupby(['role','site']):
  for name in ['selected','median']+(['persistence'] if prefix=='primary' else []):rows.append({'role':role,'landscape':site,'model':name,'n':len(a),**metrics(a.observed,a[name])})
 results=pd.DataFrame(rows);results.to_csv(D/f'{prefix}_landscape_metrics.csv',index=False)
 macrores={name:macro(test,test[target],pred[pred.role=='new_transfer'][name]) for name in ['selected','median']+(['persistence'] if prefix=='primary' else [])}
 # Nested selection: the entire omitted landscape is absent from preprocessing and tuning.
 nests=[]
 if nested:
  for held in sorted(train.site.unique()):
   tr=train[train.site!=held].reset_index(drop=True);va=train[train.site==held];b,_=select(tr,target);ss=b[1];mm=fit(ss,tr,target);pp=mm.predict(va[ss[3]])
   nests.append({'landscape':held,'n':len(va),'selected':str(ss[:3]),'mae':mean_absolute_error(va[target],pp),'median_mae':mean_absolute_error(va[target],np.repeat(np.median(tr[target]),len(va)))})
   print('outer',prefix,held,nests[-1]['mae'],flush=True)
 pd.DataFrame(nests).to_csv(D/f'{prefix}_nested_development.csv',index=False)
 summary={'status':'completed','target':target,'selected':list(spec[:3]),'selection_macro_mae':score,'nested_macro_mae':float(np.mean([x['mae'] for x in nests])) if nests else None,'nested_median_macro_mae':float(np.mean([x['median_mae'] for x in nests])) if nests else None,'test_macro_mae':macrores,'train_n':len(train),'test_n':len(test),'legacy_n':len(legacy),'train_landscapes':int(train.site.nunique()),'test_landscapes':int(test.site.nunique()),'equal_landscape_training_weights':True,'test_interpretation':'new locations reserved after pilot; one storm and one transfer region; not a new external prospective cohort','seed':SEED}
 (D/f'{prefix}_status.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True);return summary
if __name__=='__main__':
 mode=sys.argv[1] if len(sys.argv)>1 else 'all'
 if mode in ['build','all']:
  o=observations();df=features(o);df['region']=df.site.str.split('_').str[0];df.to_csv(D/'analysis_table_all.csv',index=False);df[df.primary_eligible].to_csv(D/'analysis_table_primary.csv',index=False)
  print(df.groupby('site').primary_eligible.agg(['sum','count']).to_string(),flush=True)
 if mode in ['model','all']:run(pd.read_csv(D/'analysis_table_primary.csv'))