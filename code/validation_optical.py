"""Add optical acquisitions in months of quality-qualified lidar shots.
Selection uses shot dates/locations and scene cloud metadata, never cover values.
These scenes are for concurrent reference checks only, never forecasting inputs.
"""
import json,concurrent.futures,collections
from pathlib import Path
import pandas as pd
from shapely.geometry import Point,shape
from extract import extract
R=Path(__file__).resolve().parents[1];D=R/'data'
if __name__=='__main__':
 ref=pd.read_csv(D/'gedi_mangrove_reference.csv');jobs=[]
 for site,a in ref.groupby('site'):
  g=json.load(open(D/f'{site}_grid.json'));b=g['bbox'];pt=Point((b[0]+b[2])/2,(b[1]+b[3])/2);months=set(a.time.str[:7]);fs=json.load(open(D/f'{site}_sentinel-2-l2a_catalogue.json'))['features'];groups=collections.defaultdict(list)
  for f in fs:
   if f['properties']['datetime'][:7] in months and all(k in f['assets'] for k in ['SCL','B04','B08','B11']) and shape(f['geometry']).covers(pt):groups[f['properties']['datetime'][:7]].append(f)
  for mo,ls in groups.items():
   used=set()
   for f in sorted(ls,key=lambda f:(f['properties'].get('eo:cloud_cover',100),f['id'])):
    day=f['properties']['datetime'][:10]
    if day in used:continue
    jobs.append((site,'sentinel-2-l2a',f));used.add(day)
    if len(used)==6:break
 pd.DataFrame([{'landscape':s,'collection':c,'scene_id':f['id'],'datetime':f['properties']['datetime'],'purpose':'concurrent GEDI comparison only','assets':';'.join(f['assets'][k]['href'] for k in ['SCL','B04','B08','B11'])} for s,c,f in jobs]).to_csv(D/'validation_scene_manifest.csv',index=False)
 todo=[j for j in jobs if not (D/'observations'/j[0]/(j[2]['id']+'.npz')).exists()];print('validation windows',len(jobs),'new',len(todo),flush=True);errors=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
  fs={ex.submit(extract,j):j for j in todo}
  for i,f in enumerate(concurrent.futures.as_completed(fs),1):
   try:f.result()
   except Exception as e:
    s,c,obj=fs[f];errors.append({'landscape':s,'scene':obj['id'],'error':type(e).__name__+': '+str(e).split('?')[0][:100]})
   if i%10==0 or i==len(todo):print('done',i,'errors',len(errors),flush=True)
 pd.DataFrame(errors).to_csv(D/'validation_acquisition_errors.csv',index=False)