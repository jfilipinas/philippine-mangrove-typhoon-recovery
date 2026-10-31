"""Outcome-blind spatial replication: retain legacy windows; add separated windows.
Uses only 2020 WorldCover to choose locations. No recovery outcomes enter selection.
"""
import os,sys,json,shutil,concurrent.futures,collections
from pathlib import Path
import numpy as np,pandas as pd,rasterio
from rasterio.warp import transform_bounds,Resampling
from rasterio.transform import from_origin
from scipy.ndimage import uniform_filter
from catalogue import query
from extract import read_grid,get_jobs,extract
R=Path(__file__).resolve().parents[1];D=R/'data';OLD=R.parent/'mangrove_research'
REGIONS={'siargao':[125.88,9.64,126.18,10.08],'leyte':[124.70,10.05,125.12,10.55],'palawan':[119.08,10.08,119.76,10.82]}
def catalogues():
 jobs=[(r,b,c,t) for r,b in REGIONS.items() for c,t in [('esa-worldcover','2020-01-01/2020-12-31'),('sentinel-2-l2a','2019-01-01/2023-12-31'),('sentinel-1-rtc','2019-01-01/2023-12-31')]]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
  for f in concurrent.futures.as_completed([ex.submit(query,*j) for j in jobs]):print('catalogue',f.result(),flush=True)
def select(region):
 # Existing original landscape is preserved as number 01, never used as a fresh blind test.
 legacy=json.load(open(OLD/'data'/f'{region}_grid.json'));g=legacy.copy();g.update(site=region+'_01',region=region,landscape=region+'_01',selection='legacy pilot window; original outcomes already seen')
 lid=g['landscape'];(D/f'{lid}_grid.json').write_text(json.dumps(g,indent=2));shutil.copy2(OLD/'data'/f'{region}_grid.npz',D/f'{lid}_grid.npz')
 od=D/'observations'/lid;od.mkdir(parents=True,exist_ok=True)
 for p in (OLD/'data/observations'/region).glob('*.npz'):
  dst=od/p.name
  if not dst.exists():os.link(p,dst)
 b=REGIONS[region];epsg=32600+int(((b[0]+b[2])/2+180)//6)+1;crs=f'EPSG:{epsg}'
 x0,y0,x1,y1=transform_bounds('EPSG:4326',crs,*b);n=int(max(x1-x0,y1-y0)//100)+1;tr=from_origin(x0,y1,100,100)
 maskfile=D/f'{region}_screen2020.npz'
 if maskfile.exists():a=np.load(maskfile)['classes']
 else:
  a=np.zeros((n,n),dtype='float32')
  for item in json.load(open(D/f'{region}_esa-worldcover_catalogue.json'))['features']:
   a=np.maximum(a,read_grid(item['assets']['map']['href'],crs,tr,n,Resampling.nearest))
  np.savez_compressed(maskfile,classes=a,transform=tuple(tr)[:6],crs=crs)
 density=uniform_filter((a==95).astype(float),size=24,mode='constant');density[:12,:]=-1;density[-12:,:]=-1;density[:,:12]=-1;density[:,-12:]=-1
 from pyproj import Transformer
 tf=Transformer.from_crs('EPSG:4326',crs,always_xy=True);oldb=g['bbox'];centres=[tf.transform((oldb[0]+oldb[2])/2,(oldb[1]+oldb[3])/2)]
 yy,xx=np.indices(a.shape);px=tr.c+100*(xx+.5);py=tr.f-100*(yy+.5)
 rows=[g]
 for k in [2,3,4]:
  lid=f'{region}_{k:02}';existing=D/f'{lid}_grid.json'
  if existing.exists():
   meta=json.load(open(existing));rows.append(meta);bb=meta['bbox'];centres.append(tf.transform((bb[0]+bb[2])/2,(bb[1]+bb[3])/2));continue
  for cx,cy in centres:density[((px-cx)**2+(py-cy)**2)<8000**2]=-1
  accepted=False
  for attempt in range(12):
   rr,cc=np.unravel_index(np.argmax(density),density.shape)
   if density[rr,cc]<=0:break
   cx,cy=px[rr,cc],py[rr,cc];t=from_origin(cx-1200,cy+1200,20,20);local=np.zeros((120,120),dtype='float32')
   for item in json.load(open(D/f'{region}_esa-worldcover_catalogue.json'))['features']:
    local=np.maximum(local,read_grid(item['assets']['map']['href'],crs,t,120,Resampling.nearest))
   mask=local==95;cover=mask.reshape(24,5,24,5).mean((1,3));eligible=cover>=.70
   if eligible.sum()>=50:
    bb=transform_bounds(crs,'EPSG:4326',*rasterio.transform.array_bounds(120,120,t));meta={'site':lid,'region':region,'landscape':lid,'crs':crs,'transform':tuple(t)[:6],'shape':[120,120],'bbox':bb,'eligible_cells_70pct':int(eligible.sum()),'mask':'ESA WorldCover 2020 v100 class 95','selection':'pre-storm mapped mangrove density; centres >=8 km apart; no post-storm outcome inspection'}
    np.savez_compressed(D/f'{lid}_grid.npz',mask=mask,cover=cover,eligible=eligible,transform=tuple(t)[:6],crs=crs);existing.write_text(json.dumps(meta,indent=2));centres.append((cx,cy));rows.append(meta);accepted=True;break
   density[((px-cx)**2+(py-cy)**2)<3000**2]=-1
  if not accepted:print('insufficient qualifying spatially separated window',region,k,flush=True)
 for meta in rows:
  lid=meta['landscape']
  for col in ['sentinel-1-rtc','sentinel-2-l2a']:
   # Each landscape still selects its own footprint-covering, outcome-blind overpasses.
   shutil.copy2(D/f'{region}_{col}_catalogue.json',D/f'{lid}_{col}_catalogue.json')
  print('selected',lid,meta['bbox'],meta['eligible_cells_70pct'],flush=True)
 return rows
def acquire():
 landscapes=[p.name.replace('_grid.json','') for p in D.glob('*_grid.json')];jobs=[j for s in sorted(landscapes) for c in ['sentinel-2-l2a','sentinel-1-rtc'] for j in get_jobs(s,c)]
 rows=[{'landscape':s,'region':s.split('_')[0],'collection':c,'scene_id':f['id'],'datetime':f['properties']['datetime'],'relative_orbit':f['properties'].get('sat:relative_orbit'),'cloud_cover':f['properties'].get('eo:cloud_cover'),'assets':';'.join(f['assets'][k]['href'] for k in (['SCL','B04','B08','B11'] if c=='sentinel-2-l2a' else ['vv','vh']))} for s,c,f in jobs]
 pd.DataFrame(rows).to_csv(D/'expanded_scene_manifest.csv',index=False);todo=[j for j in jobs if not (D/'observations'/j[0]/(j[2]['id']+'.npz')).exists()];print('total requests',len(jobs),'new',len(todo),flush=True);counts=collections.Counter();errors=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
  fs={ex.submit(extract,j):j for j in todo}
  for i,f in enumerate(concurrent.futures.as_completed(fs),1):
   try:z=f.result();counts[z[0]]+=1
   except Exception as e:
    s,c,it=fs[f];errors.append({'landscape':s,'collection':c,'scene':it['id'],'error':type(e).__name__+': '+str(e).split('?')[0][:150]});counts['error']+=1
   if i%25==0 or i==len(todo):print(i,dict(counts),flush=True)
 pd.DataFrame(errors).to_csv(D/'expanded_acquisition_errors.csv',index=False)
if __name__=='__main__':
 mode=sys.argv[1] if len(sys.argv)>1 else 'all'
 if mode in ['select','all']:
  catalogues()
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
   result=[v for row in ex.map(select,REGIONS) for v in row]
  pd.DataFrame(result).to_csv(D/'landscape_inventory.csv',index=False)
 if mode in ['acquire','all']:acquire()