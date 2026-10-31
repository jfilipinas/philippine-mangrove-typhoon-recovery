"""Independent-sensor canopy comparison with explicit spatial/temporal matching."""
import json,sys,warnings
from pathlib import Path
import numpy as np,pandas as pd
from pyproj import Transformer
from shapely.geometry import Point,box
from scipy.ndimage import binary_dilation
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from analyse import metrics
from replicated_analysis import run
R=Path(__file__).resolve().parents[1];D=R/'data'
def weights(x,y,g,mask):
 t=g['transform'];px=(x-t[2])/20;py=(t[5]-y)/20;r=int(np.floor(py));c=int(np.floor(px));circle=Point(x,y).buffer(12.5,resolution=32);out=[]
 for rr in range(max(0,r-2),min(120,r+3)):
  for cc in range(max(0,c-2),min(120,c+3)):
   if not mask[rr,cc]:continue
   area=circle.intersection(box(t[2]+20*cc,t[5]-20*(rr+1),t[2]+20*(cc+1),t[5]-20*rr)).area
   if area>0:out.append((rr,cc,area/circle.area))
 return out
def match():
 rows=[];aud=[];manifest=pd.read_csv(D/'expanded_scene_manifest.csv');allow=set(zip(manifest.landscape,manifest.scene_id))
 if (D/'validation_scene_manifest.csv').exists():
  vm=pd.read_csv(D/'validation_scene_manifest.csv');allow |= set(zip(vm.landscape,vm.scene_id))
 for gp in sorted(D.glob('*_grid.json')):
  g=json.load(open(gp));site=g['site'];region=site.split('_')[0];a=np.load(D/f'{site}_grid.npz');mask=a['mask'];eligible=a['eligible'];raw=pd.concat([pd.read_csv(D/f'{region}_gedi_raw.csv',dtype={'shot_number':str}),pd.read_csv(D/f'{region}_gedi_baseline_raw.csv',dtype={'shot_number':str})],ignore_index=True).drop_duplicates('shot_number');raw['period']=np.where(pd.to_datetime(raw.time)<'2021-12-16','prestorm','poststorm_2022_23')
  tf=Transformer.from_crs('EPSG:4326',g['crs'],always_xy=True);x,y=tf.transform(raw.longitude.to_numpy(),raw.latitude.to_numpy());t=g['transform'];rr=np.floor((t[5]-y)/20).astype(int);cc=np.floor((x-t[2])/20).astype(int);inside=(rr>=0)&(rr<120)&(cc>=0)&(cc<120)
  raw=raw.loc[inside].copy();raw['x']=x[inside];raw['y']=y[inside];raw['r20']=rr[inside];raw['c20']=cc[inside];ninside=len(raw)
  quality=(raw.quality_flag==1)&(raw.l2a_quality_flag==1)&(raw.l2b_quality_flag==1)&(raw.degrade_flag==0)&raw.sensitivity.between(.95,1)&raw.cover.between(0,1)
  raw=raw.loc[quality].copy();nquality=len(raw)
  scenes=[]
  for p in sorted((D/'observations'/site).glob('*.npz')):
   if (site,p.stem) not in allow:continue
   d=np.load(p);dt=pd.Timestamp(str(d['date'])[:10])
   if str(d['collection'])!='sentinel-2-l2a' or dt>pd.Timestamp('2023-03-31') or bool(d['all_cloud']):continue
   props=json.loads(str(d['properties']));off=1000 if float(props.get('s2:processing_baseline',0))>=4 else 0
   red=(d['B04']-off)/10000;nir=(d['B08']-off)/10000
   with np.errstate(divide='ignore',invalid='ignore'):v=(nir-red)/(nir+red)
   clear=np.isin(d['SCL'],[4,5,6]);clear &= ~binary_dilation(~clear,iterations=1);v[~(clear&np.isfinite(v)&(v>=-1)&(v<=1)&((nir+red)>.01))]=np.nan
   scenes.append((dt,p.stem,v))
  purityn=0
  for _,s in raw.iterrows():
   r=int(s.r20)//5;c=int(s.c20)//5
   if not eligible[r,c]:continue
   w=weights(s.x,s.y,g,mask);purity=sum(z[2] for z in w)
   if purity<.7:continue
   purityn+=1;record={'site':site,'region':region,'cell':f'{site}_{r:02}_{c:02}','row':r,'col':c,**s.to_dict(),'footprint_mangrove_fraction':purity};dt=pd.Timestamp(s.time)
   shift=[sum(z[2] for z in weights(s.x+dx,s.y+dy,g,mask)) for dx,dy in [(10,0),(-10,0),(0,10),(0,-10)]];record['all_shifted_pure']=min(shift)>=.7;record['shifted_min_mangrove_fraction']=min(shift)
   valid=[]
   for sd,scene,v in scenes:
    if s.period=='prestorm' and sd>=pd.Timestamp('2021-12-16'):continue
    lag=abs((sd-dt).days)
    if lag>30:continue
    vw=[(v[rr,cc],wt) for rr,cc,wt in w if np.isfinite(v[rr,cc])];cw=sum(z[1] for z in vw)
    if cw<.7:continue
    nd=sum(val*wt for val,wt in vw)/cw;valid.append((lag,sd,scene,nd,cw))
   record.update(matched=False,ndvi=np.nan,lag_days=np.nan)
   if valid:
    lag,sd,scene,nd,cw=sorted(valid,key=lambda z:(z[0],z[1],z[2]))[0];record.update(matched=True,ndvi=nd,lag_days=lag,optical_date=str(sd.date()),optical_scene=scene,clear_mapped_footprint_fraction=cw)
   rows.append(record)
  aud.append({'landscape':site,'region':region,'returned_shots_inside_window':ninside,'quality_pass':nquality,'mapped_footprint_and_cell_pass':purityn,'optical_matched_30d':sum(z['site']==site and z['matched'] for z in rows),'matched_prestorm':sum(z['site']==site and z['matched'] and z['period']=='prestorm' for z in rows),'matched_poststorm':sum(z['site']==site and z['matched'] and z['period']=='poststorm_2022_23' for z in rows),'optical_validation_scenes':len(scenes)})
 ref=pd.DataFrame(rows);ref.to_csv(D/'gedi_mangrove_reference.csv',index=False);pd.DataFrame(aud).to_csv(D/'gedi_attrition.csv',index=False);return ref
def calibration(ref):
 d=ref[ref.matched].copy();dev=d[~d.site.str.startswith('palawan')];test=d[d.site.isin(['palawan_02','palawan_03','palawan_04'])]
 if dev.site.nunique()<3 or test.site.nunique()<2:
  (D/'canopy_calibration_status.json').write_text(json.dumps({'status':'insufficient_landscape_coverage','development_shots':len(dev),'new_test_shots':len(test),'development_landscapes':dev.site.nunique(),'new_test_landscapes':test.site.nunique()},indent=2));return
 # Balance landscapes in calibration; shots are correlated and are not independent replicates.
 m=LinearRegression();w=1/dev.site.map(dev.site.value_counts()).to_numpy();m.fit(dev[['ndvi']],dev.cover,sample_weight=w);d['predicted_cover']=np.clip(m.predict(d[['ndvi']]),0,1);d['median_cover']=dev.cover.median();d['orbit']=d.shot_number.str.zfill(18).str[:5];d['orbit_date']=d.orbit+'_'+d.time.astype(str)
 dv=dev.sort_values('cover').copy();dv['weight']=1/dv.site.map(dv.site.value_counts());balanced=float(dv.loc[dv.weight.cumsum()>=dv.weight.sum()/2,'cover'].iloc[0]);d['balanced_median_cover']=balanced
 d.to_csv(D/'canopy_calibration_predictions.csv',index=False);res=[]
 for site,a in d.groupby('site'):
  for name in ['predicted_cover','median_cover','balanced_median_cover']:
   ma=metrics(a.cover,a[name]);rho=spearmanr(a.ndvi,a.cover).statistic if len(a)>2 else np.nan
   res.append({'landscape':site,'n_shots':len(a),'n_cells':a.cell.nunique(),'n_dates':a.time.nunique(),'n_orbit_dates':a.orbit_date.nunique(),'model':name,'spearman_ndvi_cover':rho,**ma})
 pd.DataFrame(res).to_csv(D/'canopy_landscape_metrics.csv',index=False)
 orbit=d.groupby(['site','orbit_date']).agg(n=('cover','size'),observed_cover=('cover','median'),predicted_cover=('predicted_cover','median'),median_cover=('median_cover','median')).reset_index();orbit.to_csv(D/'canopy_orbit_date_metrics.csv',index=False)
 summ={'status':'completed','development_shots':len(dev),'new_test_shots':len(test),'development_landscapes':dev.site.nunique(),'new_test_landscapes':test.site.nunique(),'intercept':float(m.intercept_),'slope':float(m.coef_[0]),'median_development_cover':float(dev.cover.median()),'notes':'Concurrent NDVI-cover calibration. Independent lidar measurement, not field truth or a validation of Q4 2023 recovery.'}
 for name,cond in [('primary_30d',np.ones(len(d),bool)),('prestorm',d.period=='prestorm'),('poststorm',d.period=='poststorm_2022_23'),('lag_15d',d.lag_days<=15),('sensitivity_098',d.sensitivity>=.98),('shift_10m_pure',d.all_shifted_pure),('night',d.solar_elevation<0)]:
  a=d.loc[cond&d.site.isin(['palawan_02','palawan_03','palawan_04'])];land=a.groupby('site').apply(lambda x:pd.Series({'mae':np.abs(x.cover-x.predicted_cover).mean(),'median_mae':np.abs(x.cover-x.median_cover).mean(),'balanced_median_mae':np.abs(x.cover-x.balanced_median_cover).mean()}),include_groups=False)
  summ[name]={'shots':len(a),'landscapes':a.site.nunique(),'mae_macro':float(land.mae.mean()) if len(land) else None,'median_mae_macro':float(land.median_mae.mean()) if len(land) else None,'balanced_median_mae_macro':float(land.balanced_median_mae.mean()) if len(land) else None}
 summ['balanced_development_cover_median']=balanced
 (D/'canopy_calibration_status.json').write_text(json.dumps(summ,indent=2));print(json.dumps(summ,indent=2),flush=True)
def forecast(ref):
 ref=ref[ref.period=='poststorm_2022_23']
 # Use all quality/purity-qualified lidar measurements, independent of optical match availability.
 lid=ref.groupby(['cell','site']).agg(gedi_cover=('cover','median'),gedi_shots=('shot_number','size'),gedi_dates=('time','nunique')).reset_index();allf=pd.read_csv(D/'analysis_table_all.csv');a=allf[(allf.baseline_years>=2)&(allf.early_dates>=2)&(allf.vv_early_n>=1)&(allf.vh_early_n>=1)].merge(lid,on=['cell','site']);a.to_csv(D/'canopy_forecast_table.csv',index=False);run(a,target='gedi_cover',prefix='canopy_forecast',nested=True)
if __name__=='__main__':
 mode=sys.argv[1] if len(sys.argv)>1 else 'all'
 if mode in ['match','all']:ref=match()
 else:ref=pd.read_csv(D/'gedi_mangrove_reference.csv',dtype={'shot_number':str})
 if mode in ['calibrate','all']:calibration(ref)
 if mode in ['forecast','all']:forecast(ref)