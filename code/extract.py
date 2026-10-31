"""Extract actual 20 m pilot windows from public cloud-optimised GeoTIFFs.
Fixed pre-storm mask, outcome-blind site selection, resumable NPZ observations.
"""
import os
os.environ.setdefault('GDAL_CURL_CA_BUNDLE',os.environ.get('SSL_CERT_FILE','/etc/ssl/certs/ca-certificates.crt'))
os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN','EMPTY_DIR')
import json,sys,concurrent.futures,collections,datetime,hashlib,warnings
from pathlib import Path
import numpy as np,pandas as pd,rasterio
from rasterio.warp import reproject,Resampling,transform_bounds
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from scipy.ndimage import uniform_filter,binary_dilation
from pyproj import Transformer
from shapely.geometry import shape,Point
from http_raster import open_remote
from catalogue import SITES
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data'
def read_grid(href,crs,tr,n,resampling=Resampling.nearest):
    out=np.zeros((n,n),dtype='float32')
    with open_remote(href) as s:
        b=rasterio.transform.array_bounds(n,n,tr);bb=transform_bounds(crs,s.crs,*b)
        w=from_bounds(*bb,s.transform).round_offsets().round_lengths()
        if w.width<1 or w.height<1:raise ValueError('empty source window')
        a=s.read(1,window=w,boundless=True,fill_value=0)
        reproject(a,out,src_transform=s.window_transform(w),src_crs=s.crs,dst_transform=tr,dst_crs=crs,resampling=resampling,src_nodata=s.nodata or 0,dst_nodata=0)
    return out
def mask_site(site):
    path=DATA/f'{site}_grid.npz'
    if path.exists():return site,'cached grid'
    b=SITES[site];epsg=32600+int((sum(b[::2])/2+180)//6)+1;crs=f'EPSG:{epsg}'
    x0,y0,x1,y1=transform_bounds('EPSG:4326',crs,*b)
    # Work on the full candidate bounding box at 20 m and choose by pre-event mask only.
    n=int(max(x1-x0,y1-y0)//20);tr=from_origin(x0,y1,20,20)
    obj=json.load(open(DATA/f'{site}_esa-worldcover_catalogue.json'))['features'][0]
    a=read_grid(obj['assets']['map']['href'],crs,tr,n)
    # Fixed 2.4 km landscape; maximum density of 2020 mapped mangrove, no outcome access.
    width=120;density=uniform_filter((a==95).astype(float),size=width,mode='constant')
    density[:width//2,:]=-1;density[-width//2:,:]=-1;density[:,:width//2]=-1;density[:,-width//2:]=-1
    rr,cc=np.unravel_index(np.argmax(density),density.shape);r0=rr-width//2;c0=cc-width//2
    mask=a[r0:r0+width,c0:c0+width]==95;t=tr*rasterio.Affine.translation(c0,r0)
    cover=mask.reshape(24,5,24,5).mean((1,3));eligible=cover>=.70
    np.savez_compressed(path,mask=mask,cover=cover,eligible=eligible,transform=np.array(tuple(t)[:6]),crs=crs)
    geo=transform_bounds(crs,'EPSG:4326',*rasterio.transform.array_bounds(width,width,t))
    meta={'site':site,'crs':crs,'transform':tuple(t)[:6],'shape':[width,width],'bbox':geo,'eligible_cells_70pct':int(eligible.sum()),'mask':'ESA WorldCover 2020 v100 class 95','selection':'maximum pre-storm mapped mangrove density in candidate bbox; 2.4 km square'}
    (DATA/f'{site}_grid.json').write_text(json.dumps(meta,indent=2));return meta
def get_jobs(site,collection,max_per_month=2):
    grid=json.load(open(DATA/f'{site}_grid.json'));b=grid['bbox'];pt=Point((b[0]+b[2])/2,(b[1]+b[3])/2)
    fs=json.load(open(DATA/f'{site}_{collection}_catalogue.json'))['features'];groups=collections.defaultdict(list)
    # Only specified baseline seasons, predictor windows and outcome seasons; no future predictor information.
    allowed=set([f'{y}-{m:02}' for y in [2019,2020,2021] for m in [1,2,3,10,11,12]]+[f'2022-{m:02}' for m in [1,2,3,10,11,12]]+[f'2023-{m:02}' for m in [1,2,3,10,11,12]])
    for f in fs:
        d=f['properties']['datetime']
        needed=['SCL','B04','B08','B11'] if collection=='sentinel-2-l2a' else ['vv','vh']
        if not all(k in f['assets'] for k in needed):continue
        if d[:7] not in allowed or not shape(f['geometry']).covers(pt):continue
        groups[d[:7]].append(f)
    if collection=='sentinel-1-rtc':
        cnt=collections.Counter(f['properties']['sat:relative_orbit'] for g in groups.values() for f in g if f['properties']['datetime']<'2021-12-16')
        orbit=cnt.most_common(1)[0][0]
        groups={m:[f for f in g if f['properties']['sat:relative_orbit']==orbit] for m,g in groups.items()}
    jobs=[]
    for m,g in groups.items():
        if collection=='sentinel-2-l2a':
            g=sorted(g,key=lambda f:(f['properties'].get('eo:cloud_cover',100),f['id']))
        else:g=sorted(g,key=lambda f:f['properties']['datetime'])
        # Deduplicate overpasses across overlapping tiles / slices.
        seen=set();selected=[]
        for f in g:
            day=f['properties']['datetime'][:10]
            if day not in seen:selected.append(f);seen.add(day)
            if len(selected)>=max_per_month:break
        jobs += [(site,collection,f) for f in selected]
    return jobs
def extract(job):
    site,col,f=job;p=DATA/'observations'/site/f"{f['id']}.npz";p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():return 'cached',site,f['id']
    g=json.load(open(DATA/f'{site}_grid.json'));tr=rasterio.Affine(*g['transform']);crs=g['crs'];n=g['shape'][0]
    bands=['SCL','B04','B08','B11'] if col=='sentinel-2-l2a' else ['vv','vh']
    a={}
    for band in bands:
        a[band]=read_grid(f['assets'][band]['href'],crs,tr,n,Resampling.nearest if band=='SCL' else Resampling.average)
        if band=='SCL':
            # Keep water over the fixed pre-event cohort to avoid selecting only survivors.
            if np.isin(a[band],[4,5,6]).sum()==0:
                np.savez_compressed(p,SCL=a[band],date=f['properties']['datetime'],collection=col,scene=f['id'],all_cloud=True);return 'cloud',site,f['id']
    a.update(date=f['properties']['datetime'],collection=col,scene=f['id'],properties=json.dumps(f['properties']),all_cloud=False)
    np.savez_compressed(p,**a)
    return 'ok',site,f['id']
if __name__=='__main__':
    mode=sys.argv[1] if len(sys.argv)>1 else 'all'
    if mode in ['mask','all']:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            for f in concurrent.futures.as_completed([ex.submit(mask_site,s) for s in SITES]):
                try:print(f.result(),flush=True)
                except Exception as e:print('MASK ERROR',type(e).__name__,str(e)[:180],flush=True)
    if mode in ['extract','all','pilot']:
        jobs=[j for s in SITES if (DATA/f'{s}_grid.json').exists() for c in ['sentinel-2-l2a','sentinel-1-rtc'] for j in get_jobs(s,c)]
        if mode=='pilot':jobs=[j for j in jobs if j[2]['properties']['datetime'].startswith('2020-02')]
        rows=[{'site':s,'collection':c,'scene_id':f['id'],'datetime':f['properties']['datetime'],'cloud_cover':f['properties'].get('eo:cloud_cover'),'relative_orbit':f['properties'].get('sat:relative_orbit'),'assets':';'.join(f['assets'][k]['href'] for k in (['SCL','B04','B08','B11'] if c=='sentinel-2-l2a' else ['vv','vh']))} for s,c,f in jobs]
        pd.DataFrame(rows).to_csv(DATA/f'{mode}_scene_manifest.csv',index=False)
        print('jobs',len(jobs),flush=True)
        done=collections.Counter();errors=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            fs={ex.submit(extract,j):j for j in jobs}
            for i,fu in enumerate(concurrent.futures.as_completed(fs),1):
                try:r=fu.result();done[r[0]]+=1
                except Exception as e:
                    s,c,f=fs[fu];errors.append({'site':s,'collection':c,'scene':f['id'],'error':type(e).__name__+': '+str(e).split('?')[0][:240]});done['error']+=1
                if i%10==0 or i==len(jobs):print(i,dict(done),flush=True)
        pd.DataFrame(errors).to_csv(DATA/f'{mode}_acquisition_errors.csv',index=False)