import pandas as pd,numpy as np,statsmodels.formula.api as smf
from pathlib import Path
p=Path('/mnt/data/Mangrove_Submission_Revision'); data=p/'source/Philippine_Mangrove_Revision/data'; out=p/'results';out.mkdir(exist_ok=True)
g=pd.read_csv(data/'gedi_mangrove_reference.csv');g=g[g.matched.eq(True)&g.ndvi.notna()&g.cover.notna()].copy();g['ndvi_mean']=g.groupby('site').ndvi.transform('mean');g['ndvi_within']=g.ndvi-g.ndvi_mean
reports=[]
for name,formula,re in [('random_intercept','cover ~ ndvi_within + ndvi_mean','1'),('random_slope','cover ~ ndvi_within + ndvi_mean','1+ndvi_within')]:
 try:
  m=smf.mixedlm(formula,g,groups=g.site,re_formula=re).fit(reml=True,method='lbfgs',maxiter=500)
  reports.append((name,str(m.summary()),m.converged));
  pd.DataFrame({'term':m.params.index,'estimate':m.params.values,'se':[m.bse.get(i,np.nan) for i in m.params.index]}).to_csv(out/f'{name}_coefficients.csv',index=False)
 except Exception as e:reports.append((name,str(e),False))
# landscape bootstrap within/between OLS (resample landscape clusters)
rs=np.random.default_rng(20261027);sites=g.site.unique(); estimates=[]
for b in range(1200):
 sampled=rs.choice(sites,size=len(sites),replace=True);chunks=[]
 for j,s in enumerate(sampled):
  a=g[g.site.eq(s)].copy();a['boot_site']=str(j);chunks.append(a)
 z=pd.concat(chunks,ignore_index=True);z['ndvi_mean']=z.groupby('boot_site').ndvi.transform('mean');z['ndvi_within']=z.ndvi-z.ndvi_mean
 try:
  fit=smf.ols('cover ~ ndvi_within + ndvi_mean',z).fit();estimates.append([fit.params['ndvi_within'],fit.params['ndvi_mean']])
 except Exception:pass
arr=np.array(estimates)
with open(out/'hierarchical_and_bootstrap.txt','w') as f:
 f.write(f'Matched shots: {len(g)}; landscapes: {len(sites)}\nLandscape bootstrap: {len(arr)} valid replicates, seed 20261027\n')
 for n,s,conv in reports:f.write(f'\n{n} converged={conv}\n{s}\n')
 for i,n in enumerate(['within','between']):f.write(f'Bootstrap {n} percentile CI 2.5%,50%,97.5%: {np.quantile(arr[:,i],[.025,.5,.975])}\n')
# selection diagnostics, only covariates available in both groups
c=pd.read_csv(data/'analysis_table_all.csv');c['eligible']=c.primary_eligible.astype(str).str.lower().eq('true'); rows=[]
for v in ['ndvi_base','ndmi_base','ndvi_early','ndmi_early','vv_base','vh_base','early_dates','target_2023_n']:
 for site,z in c.groupby('site'):
  a=z.loc[z.eligible,v].dropna();b=z.loc[~z.eligible,v].dropna();den=np.sqrt((a.var()+b.var())/2) if len(a)>1 and len(b)>1 else np.nan
  rows.append(dict(site=site,covariate=v,eligible_n=len(a),excluded_n=len(b),eligible_mean=a.mean(),excluded_mean=b.mean(),standardized_difference=(a.mean()-b.mean())/den if den and den>0 else np.nan))
pd.DataFrame(rows).to_csv(out/'selection_standardized_differences.csv',index=False)
print('Finished',len(g),len(sites),len(c));print((out/'hierarchical_and_bootstrap.txt').read_text()[-1100:])