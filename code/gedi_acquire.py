"""Acquire an independent lidar reference; preserve raw public query output."""
import os,json,urllib.parse,sys,time
from pathlib import Path
import gedidb as gdb,tiledb,geopandas as gpd
from shapely.geometry import box
R=Path(__file__).resolve().parents[1];D=R/'data'
def provider():
 p=gdb.GEDIProvider(storage_type='s3',s3_bucket='dog-ext.gedidb.gedi-l2-l4-v002.0',url='https://s3.gfz.de')
 c=p.ctx.config();pr=urllib.parse.urlsplit(os.environ.get('HTTPS_PROXY',os.environ.get('https_proxy','')))
 if pr.hostname:
  c['vfs.s3.proxy_host']=pr.hostname;c['vfs.s3.proxy_port']=str(pr.port or 80);c['vfs.s3.proxy_scheme']=pr.scheme
  if pr.username:c['vfs.s3.proxy_username']=urllib.parse.unquote(pr.username)
  if pr.password:c['vfs.s3.proxy_password']=urllib.parse.unquote(pr.password)
 ca=os.environ.get('SSL_CERT_FILE')
 if ca:c['vfs.s3.ca_file']=ca;c['ssl.ca_file']=ca
 c['vfs.s3.connect_timeout_ms']='15000';c['vfs.s3.request_timeout_ms']='60000';c['vfs.s3.max_tries']='2'
 p.ctx=tiledb.Ctx(c);return p
if __name__=='__main__':
 period=sys.argv[1] if len(sys.argv)>1 else 'baseline'
 suffix='baseline_' if period=='baseline' else ''
 start,end=('2019-04-18','2021-12-15') if period=='baseline' else ('2022-10-01','2023-03-16')
 p=provider();v=p.get_available_variables();v.to_csv(D/'gedidb_variable_metadata.csv');print(v.to_string()[:15000],flush=True)
 for region in ['siargao','leyte','palawan']:
  dest=D/f'{region}_gedi_{suffix}raw.csv'
  if dest.exists():continue
  grids=[json.load(open(f)) for f in D.glob(f'{region}_*_grid.json')]
  b=[min(g['bbox'][0] for g in grids)-.002,min(g['bbox'][1] for g in grids)-.002,max(g['bbox'][2] for g in grids)+.002,max(g['bbox'][3] for g in grids)+.002]
  variables=['cover','quality_flag','l2b_quality_flag','sensitivity','rh:98']
  available=set(v.index.astype(str))|set(v['name'].astype(str) if 'name' in v else [])
  for extra in ['degrade_flag','l2a_quality_flag','solar_elevation']:
   if extra in available:variables.append(extra)
  print('query',region,b,variables,flush=True)
  d=p.get_data(variables=variables,query_type='bounding_box',geometry=gpd.GeoDataFrame(geometry=[box(*b)],crs='EPSG:4326'),start_time=start,end_time=end,return_type='dataframe')
  if d is None:raise RuntimeError('No returned GEDI data for '+region)
  d.to_csv(dest,index=False)
  (D/f'{region}_gedi_{suffix}query.json').write_text(json.dumps({'region':region,'bbox':b,'variables':variables,'start_time':start,'end_time':end,'rows':len(d),'provider':'GFZ gediDB public GEDI v2 mirror','bucket':'dog-ext.gedidb.gedi-l2-l4-v002.0','endpoint':'https://s3.gfz.de','source_DOI':'10.5067/GEDI/GEDI02_B.002','access_date':'2026-10-21'},indent=2))
  print('saved',region,d.shape,flush=True)