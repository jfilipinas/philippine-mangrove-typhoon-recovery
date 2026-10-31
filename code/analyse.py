"""Primary frozen pilot analysis. Outputs always derived from acquired pixels."""
import json,warnings,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.ndimage import binary_dilation
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data';SEED=20261021
warnings.filterwarnings('ignore',message='All-NaN slice encountered')
def cell_mean(a,mask):
    a=np.where(mask,a,np.nan).reshape(24,5,24,5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');return np.nanmean(a,axis=(1,3)),np.isfinite(a).sum((1,3))
def observations():
    rows=[];sceneqa=[]
    allowed=set()
    if (DATA/'expanded_scene_manifest.csv').exists():
        man=pd.read_csv(DATA/'expanded_scene_manifest.csv');allowed=set(zip(man.landscape,man.scene_id))
    for site in sorted(p.name.replace('_grid.json','') for p in DATA.glob('*_grid.json')):
        g=np.load(DATA/f'{site}_grid.npz');mask=g['mask'];eligible=g['eligible'];den=mask.reshape(24,5,24,5).sum((1,3));rr,cc=np.where(eligible)
        for p in sorted((DATA/'observations'/site).glob('*.npz')):
            if allowed and (site,p.stem) not in allowed:continue
            d=np.load(p);col=str(d['collection']);date=str(d['date'])[:10]
            values={}
            if bool(d['all_cloud']):
                sceneqa.append({'site':site,'date':date,'collection':col,'valid_cells':0,'scene':p.stem});continue
            if col=='sentinel-2-l2a':
                valid=np.isin(d['SCL'],[4,5,6]);valid &= ~binary_dilation(~valid,iterations=1)
                props=json.loads(str(d['properties']));offset=1000 if float(props.get('s2:processing_baseline',0))>=4 else 0
                red=(d['B04']-offset)/10000;nir=(d['B08']-offset)/10000;swir=(d['B11']-offset)/10000
                for name,a,b in [('ndvi',nir,red),('ndmi',nir,swir)]:
                    with np.errstate(divide='ignore',invalid='ignore'):v=(a-b)/(a+b)
                    ok=mask&valid&np.isfinite(v)&(v>=-1)&(v<=1)&((a+b)>.01)
                    av,cnt=cell_mean(v,ok);av[cnt<np.maximum(5,.5*den)]=np.nan;values[name]=av
            else:
                for name in ['vv','vh']:
                    v=d[name];ok=mask&(v>0)&np.isfinite(v)
                    av,cnt=cell_mean(v,ok)
                    with np.errstate(divide='ignore',invalid='ignore'):av=10*np.log10(av)
                    av[cnt<np.maximum(5,.5*den)]=np.nan;values[name]=av
            count=sum(np.isfinite(next(iter(values.values()))[r,c]) for r,c in zip(rr,cc))
            sceneqa.append({'site':site,'date':date,'collection':col,'valid_cells':count,'scene':p.stem})
            for r,c in zip(rr,cc):
                row={'site':site,'cell':f'{site}_{r:02}_{c:02}','row':r,'col':c,'date':date,'scene':p.stem,'collection':col}
                for k,v in values.items():row[k]=v[r,c]
                rows.append(row)
    obs=pd.DataFrame(rows);obs.to_csv(DATA/'cell_observations.csv',index=False);pd.DataFrame(sceneqa).to_csv(DATA/'scene_quality.csv',index=False)
    return obs
def features(obs):
    obs['date']=pd.to_datetime(obs.date);obs['year']=obs.date.dt.year;obs['month']=obs.date.dt.month
    records=[]
    for cell,g in obs.groupby('cell'):
        site=g.site.iloc[0];r=int(g.row.iloc[0]);c=int(g.col.iloc[0])
        pre=g[g.date<'2021-12-16'];early=g[(g.date>='2021-12-18')&(g.date<='2022-03-16')];q4=pre[pre.month>=10]
        row={'cell':cell,'site':site,'row':r,'col':c,'group':f'{site}_{r//5}_{c//5}'}
        # Equal-year/season summaries, rather than weighting frequent satellite acquisitions more heavily.
        row['baseline_years']=int(q4.dropna(subset=['ndvi']).year.nunique())
        row['early_dates']=int(early.dropna(subset=['ndvi']).date.nunique())
        for v in ['ndvi','ndmi','vv','vh']:
            base=q4.groupby('year')[v].median();row[v+'_base']=base.mean();row[v+'_base_sd']=base.std()
            row[v+'_early_n']=int(early[v].notna().sum())
            mref=pre.groupby(['year','month'])[v].median().groupby('month').mean()
            e=early[['date','month',v]].dropna().copy();e['anomaly']=e[v]-e.month.map(mref);e=e.dropna(subset=['anomaly'])
            row[v+'_early']=e.anomaly.mean();row[v+'_last']=e.sort_values('date').anomaly.iloc[-1] if len(e) else np.nan
            row[v+'_slope']=np.polyfit((e.date-e.date.min()).dt.days,e.anomaly,1)[0]*30 if e.date.nunique()>=3 else np.nan
        for year in [2022,2023]:
            target=g[(g.year==year)&(g.month>=10)]
            row[f'target_{year}_n']=int(target.ndvi.notna().sum());row[f'target_{year}_months']=int(target.dropna(subset=['ndvi']).month.nunique())
            row[f'ndvi_deficit_{year}']=row['ndvi_base']-target.groupby('month').ndvi.median().mean()
            row[f'ndmi_deficit_{year}']=row['ndmi_base']-target.groupby('month').ndmi.median().mean()
        # Persistence: early anomaly (positive if recovery exceeds seasonal expectation) -> deficit sign reversal.
        row['persistence']=-row['ndvi_early']
        row['primary_eligible']=bool(row['baseline_years']>=2 and row['early_dates']>=2 and row['target_2023_n']>=2 and row['target_2023_months']>=2 and np.isfinite(row['ndvi_deficit_2023']) and row['vv_early_n']>=1 and row['vh_early_n']>=1)
        records.append(row)
    df=pd.DataFrame(records);df.to_csv(DATA/'analysis_table_all.csv',index=False);df[df.primary_eligible].to_csv(DATA/'analysis_table_primary.csv',index=False)
    return df
def pipe(name,param):
    if name=='ridge':m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=param))
    elif name=='forest':m=make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=150,min_samples_leaf=param,max_features=.8,random_state=SEED,n_jobs=2))
    else:m=make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,l2_regularization=1,min_samples_leaf=param,random_state=SEED))
    return m
def metrics(y,p):return {'mae':mean_absolute_error(y,p),'rmse':np.sqrt(mean_squared_error(y,p)),'bias':float(np.mean(p-y)),'r2':r2_score(y,p)}
def model(df):
    # Canonical persisted table ensures reruns and sensitivity use identical float precision.
    df=pd.read_csv(DATA/'analysis_table_primary.csv');train=df[df.site!='palawan'];test=df[df.site=='palawan']
    counts=df.groupby('site').size().to_dict()
    if any(counts.get(s,0)<20 for s in ['siargao','leyte','palawan']):
        (DATA/'model_status.json').write_text(json.dumps({'status':'blocked_insufficient_coverage','counts':counts},indent=2));return
    optical=[v+s for v in ['ndvi','ndmi'] for s in ['_base','_base_sd','_early','_last','_slope']]
    radar=[v+s for v in ['vv','vh'] for s in ['_base','_base_sd','_early','_last','_slope']]
    y=train.ndvi_deficit_2023.to_numpy();yt=test.ndvi_deficit_2023.to_numpy();groups=train.group.to_numpy();folds=list(GroupKFold(n_splits=5).split(train,y,groups));out=[];pred=test[['cell','site','row','col','group','ndvi_deficit_2023']].copy();oofs=train[['cell','site','row','col','group','ndvi_deficit_2023']].copy();tuning=[]
    # Split identities precede final test evaluation and are persisted.
    foldids=[]
    for i,(_,va) in enumerate(folds):foldids += [{'cell':train.iloc[j].cell,'fold':i} for j in va]
    pd.DataFrame(foldids).to_csv(DATA/'spatial_folds.csv',index=False)
    for baseline in ['median','persistence']:
        p=np.repeat(np.median(y),len(test)) if baseline=='median' else test.persistence.to_numpy()
        op=np.empty(len(train))
        for tr,va in folds:op[va]=np.median(y[tr]) if baseline=='median' else train.iloc[va].persistence.to_numpy()
        pred[baseline]=p;oofs[baseline]=op;out.append({'features':'baseline','model':baseline,'parameter':'','test_n':len(test),'train_n':len(train),'cv_mae':mean_absolute_error(y,op),**metrics(yt,p)})
    for feat,cols in [('optical',optical),('radar',radar),('fused',optical+radar)]:
        for name,params in [('ridge',[1,10,100]),('forest',[5,15]),('histgb',[10,30])]:
            best=None
            for param in params:
                op=np.empty(len(train))
                for tr,va in folds:
                    m=pipe(name,param);m.fit(train.iloc[tr][cols],y[tr]);op[va]=m.predict(train.iloc[va][cols])
                score=mean_absolute_error(y,op);tuning.append({'features':feat,'model':name,'parameter':param,'cv_mae':score})
                if best is None or score<best[0]:best=(score,param,op)
            score,param,op=best;m=pipe(name,param);m.fit(train[cols],y);p=m.predict(test[cols]);key=feat+'_'+name
            pred[key]=p;oofs[key]=op;out.append({'features':feat,'model':name,'parameter':param,'test_n':len(test),'train_n':len(train),'cv_mae':score,**metrics(yt,p)})
    res=pd.DataFrame(out);res.to_csv(DATA/'model_metrics.csv',index=False);pd.DataFrame(tuning).to_csv(DATA/'hyperparameter_search.csv',index=False);pred.to_csv(DATA/'heldout_predictions.csv',index=False);oofs.to_csv(DATA/'oof_predictions.csv',index=False)
    # Model family is selected on development CV, not test performance.
    selected=res[res.features!='baseline'].sort_values('cv_mae').iloc[0];key=selected.features+'_'+selected.model
    residual=np.abs(y-oofs[key].to_numpy());q=float(np.quantile(residual,.90));coverage=float((np.abs(yt-pred[key])<=q).mean())
    # OOF residual interval is explicitly approximate; neither conformal nor a formal shift-robust guarantee.
    pred['selected_lower']=pred[key]-q;pred['selected_upper']=pred[key]+q;pred.to_csv(DATA/'heldout_predictions.csv',index=False)
    rng=np.random.default_rng(SEED);bs=[];gs=test.group.unique()
    for _ in range(1000):
        sample=rng.choice(gs,len(gs),replace=True);ix=np.concatenate([np.flatnonzero(test.group.to_numpy()==g) for g in sample]);p=pred[key].to_numpy()[ix];truth=yt[ix];b=pred['persistence'].to_numpy()[ix];bs.append(np.mean(np.abs(p-truth))-np.mean(np.abs(b-truth)))
    summ={'status':'completed_pilot','counts':counts,'selected_by_training_cv':key,'selected_test':selected.to_dict(),'approx_interval_halfwidth':q,'approx_interval_coverage':coverage,'interval_method':'pooled development out-of-fold absolute residual 90th percentile; descriptive only, not formal conformal coverage','test_groups':len(gs),'mae_difference_vs_persistence_ci95':np.quantile(bs,[.025,.975]).tolist(),'bootstrap_unit':'500 m blocks inside one withheld landscape; conditional descriptive uncertainty','seeds':SEED}
    (DATA/'model_status.json').write_text(json.dumps(summ,indent=2))
    print(json.dumps(summ,indent=2),flush=True)
if __name__=='__main__':
    obs=observations();df=features(obs);print(df.groupby('site').primary_eligible.agg(['sum','count']),flush=True);model(df)