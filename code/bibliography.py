"""Resolve DOI metadata from Crossref and save its exact response for audit."""
import urllib.request,urllib.parse,json,concurrent.futures,csv,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DOIS={
'segales2024':'10.5194/isprs-archives-XLVIII-5-2024-105-2024',
'long2016':'10.1016/j.marpolbul.2016.06.080',
'primavera2016':'10.1016/j.marpolbul.2016.05.050',
'lomeo2022':'10.3390/rs14102291',
'kuenzer2011':'10.3390/rs3050878',
'bunting2022':'10.3390/rs14153657',
'roberts2017':'10.1111/ecog.02881',
'olofsson2014':'10.1016/j.rse.2014.02.015',
'belgiu2016':'10.1016/j.isprsjprs.2016.01.011',
'duncan2016':'10.1016/j.marpolbul.2016.05.049',
'knapp2010':'10.1175/2009BAMS2755.1',
'amaral2023':'10.1016/j.scitotenv.2023.165413',
'yuan2026':'10.3389/fmars.2026.1632093',
'chrisnanto2026':'10.1016/j.eswa.2025.130766',
'field2025':'10.1016/j.gecco.2025.e03667',
'pedregosa2011':'10.5555/1953048.2078195',
'breiman2001':'10.1023/A:1010933404324',
'angelopoulos2023':'10.1561/2200000101',
'drusch2012':'10.1016/j.rse.2011.11.026',
'torres2012':'10.1016/j.rse.2011.05.028',
'worldcover2020':'10.5281/zenodo.5571936',
'leytedata2023':'10.17632/w35ktjjzvn.1',
'ibtracs2024':'10.25921/82ty-9e16'}
# Dataset DOIs are DataCite/provider records, not failed Crossref journal papers.
for _key in ['pedregosa2011','worldcover2020','leytedata2023','ibtracs2024']:
    DOIS.pop(_key,None)
DOIS.update({'easternsamar2021':'10.5194/isprs-annals-V-3-2021-243-2021','crossdomain2026':'10.1016/j.jenvman.2025.128283'})
def task(k,d):
 p=ROOT/'audit'/'doi_metadata'/f'{k}.json';p.parent.mkdir(exist_ok=True)
 if p.exists():return k,json.loads(p.read_text())
 try:
  req=urllib.request.Request('https://api.crossref.org/works/'+urllib.parse.quote(d,safe=''),headers={'User-Agent':'MangroveResearch/1.0'})
  with urllib.request.urlopen(req,timeout=25) as r:obj=json.load(r)['message']
  p.write_text(json.dumps(obj,indent=2));return k,obj
 except Exception as e:return k,{'DOI':d,'error':str(e)}
def tex(s):
 return str(s).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')
if __name__=='__main__':
 rows=[];bib=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
  for k,o in ex.map(lambda kv:task(*kv),DOIS.items()):
   a=o.get('author',[]);authors=' and '.join(x.get('family',x.get('name',''))+', '+x.get('given','') for x in a)
   y=o.get('published',o.get('issued',{})).get('date-parts',[[None]])[0][0]
   title=o.get('title',[''])[0];journal=o.get('container-title',[''])[0]
   rows.append({'key':k,'doi':DOIS[k],'title':title,'authors':authors,'year':y,'journal':journal,'type':o.get('type',''),'metadata_status':o.get('error','verified Crossref registry'),'content_access':'metadata only; claim verification recorded separately'})
   if title:
    bib.append('@article{'+k+',\n author={'+tex(authors)+'},\n title={'+tex(title)+'},\n year={'+str(y)+'},\n journal={'+tex(journal)+'},\n volume={'+tex(o.get('volume',''))+'},\n pages={'+tex(o.get('page',o.get('article-number','')))+'},\n doi={'+DOIS[k]+'}\n}')
   print(k,y,title or o.get('error'),flush=True)
 with (ROOT/'audit/literature_matrix.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 extra='''\n\n@misc{worldcover2020,author={Zanaga, Daniele and others},title={ESA WorldCover 10 m 2020 v100},year={2021},doi={10.5281/zenodo.5571936},note={Version 1.0.0; accessed 21 October 2026}}\n@misc{ibtracs2024,author={Gahtan, J. and Knapp, K. R. and Schreck, C. J. and Diamond, H. J. and Kossin, J. P. and Kruk, M. C.},title={International Best Track Archive for Climate Stewardship (IBTrACS) Project, Version 4r01},year={2024},doi={10.25921/82ty-9e16},note={Western Pacific subset; accessed 21 October 2026}}\n'''
 (ROOT/'manuscript/references.bib').write_text('\n\n'.join(bib)+extra)