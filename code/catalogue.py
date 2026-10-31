"""Download actual STAC metadata; no credentials required for catalogue queries."""
import json, urllib.request, urllib.parse, concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
API='https://planetarycomputer.microsoft.com/api/stac/v1'
SITES={'siargao':[125.94,9.80,126.00,9.88], 'leyte':[124.77,10.12,124.87,10.23], 'palawan':[119.30,10.28,119.40,10.38]}
def get(url):
    with urllib.request.urlopen(url,timeout=50) as r:return json.load(r)
def query(site,bbox,collection,dates):
    p=ROOT/'data'/f'{site}_{collection}_catalogue.json'
    if p.exists():return site,collection,len(json.loads(p.read_text())['features'])
    url=API+'/search?'+urllib.parse.urlencode({'collections':collection,'bbox':','.join(map(str,bbox)),'datetime':dates,'limit':1000})
    data=get(url); features=data['features']
    while any(l['rel']=='next' for l in data.get('links',[])):
        link=next(l for l in data['links'] if l['rel']=='next');data=get(link['href']);features+=data['features']
    p.write_text(json.dumps({'type':'FeatureCollection','features':features,'query':url},indent=1))
    return site,collection,len(features)
if __name__=='__main__':
    jobs=[(s,b,c,d) for s,b in SITES.items() for c,d in [('esa-worldcover','2020-01-01/2020-12-31'),('sentinel-2-l2a','2019-01-01/2023-12-31'),('sentinel-1-rtc','2019-01-01/2023-12-31')]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futs=[ex.submit(query,*j) for j in jobs]
        for f in concurrent.futures.as_completed(futs):
            try:print(f.result(),flush=True)
            except Exception as e:print(type(e).__name__,str(e),flush=True)
