import pandas as pd,numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
P=Path('/mnt/data/Philippine_Mangrove_Recovery_Extracted/Philippine_Mangrove_Revision/data'); O=Path('/mnt/data/Mangrove_Minerva_Analysis')
a=pd.read_csv(P/'analysis_table_all.csv'); g=pd.read_csv(P/'canopy_calibration_predictions.csv'); p=pd.read_csv(P/'primary_predictions.csv')
# Within/between decomposition at landscape level; cluster-robust SE by site.
g=g.dropna(subset=['ndvi','cover','site']).copy();g['site_ndvi_mean']=g.groupby('site').ndvi.transform('mean');g['ndvi_within']=g.ndvi-g.site_ndvi_mean
fits={}
for name,formula in [('pooled','cover ~ ndvi'),('within_between','cover ~ ndvi_within + site_ndvi_mean'),('site_fixed_effects','cover ~ ndvi_within + C(site)')]:
 fit=smf.ols(formula,g).fit(cov_type='cluster',cov_kwds={'groups':g.site});fits[name]=fit
pd.DataFrame([{'model':k,'term':term,'estimate':v,'se_cluster_site':fit.bse[term],'ci_low':fit.conf_int().loc[term,0],'ci_high':fit.conf_int().loc[term,1],'p_value':fit.pvalues[term],'n_shots':len(g),'n_landscapes':g.site.nunique()} for k,fit in fits.items() for term,v in fit.params.items() if term in ['ndvi','ndvi_within','site_ndvi_mean']]).to_csv(O/'gedi_within_between_results.csv',index=False)
# Selection diagnostics, explicitly observational rather than causal.
a['eligible']=a.primary_eligible.astype(str).str.lower().isin(['true','1','yes']); metrics=['ndvi_base','ndvi_early','ndmi_base','ndmi_early','target_2022_n','target_2023_n']
rows=[]
for site,d in a.groupby('site'):
 row={'site':site,'region':d.region.iloc[0],'candidate_cells':len(d),'eligible_cells':int(d.eligible.sum()),'retention_rate':d.eligible.mean()}
 for v in metrics:
  row[v+'_eligible_mean']=d.loc[d.eligible,v].mean();row[v+'_excluded_mean']=d.loc[~d.eligible,v].mean();row[v+'_difference']=row[v+'_eligible_mean']-row[v+'_excluded_mean']
 rows.append(row)
pd.DataFrame(rows).to_csv(O/'selection_diagnostics_by_landscape.csv',index=False)
# Forecast residuals, comparing actual vs available model prediction columns.
p['residual_observed_minus_predicted']=p.observed-p.selected
p['abs_error']=p.residual_observed_minus_predicted.abs()
rows=[]
for site,d in p.dropna(subset=['observed','selected']).groupby('site'):
 rows.append({'site':site,'region':d.region.iloc[0],'role':d.role.iloc[0],'n':len(d),'bias':d.residual_observed_minus_predicted.mean(),'mae':d.abs_error.mean(),'rmse':np.sqrt(np.mean(d.residual_observed_minus_predicted**2)),'sd_observed':d.observed.std(),'sd_predicted':d.selected.std(),'variance_ratio':d.selected.var()/d.observed.var() if d.observed.var()>0 else np.nan})
pd.DataFrame(rows).to_csv(O/'forecast_residuals_by_landscape.csv',index=False)
# Association diagnostics: descriptive only; no external environmental layers available.
rows=[]
for col in ['ndvi_base','ndvi_early','ndmi_base','ndmi_early','vv_early','vh_early','persistence']:
 d=p[['abs_error',col,'site']].dropna()
 if len(d)>10:
  f=smf.ols('abs_error ~ Q("'+col+'") + C(site)',d).fit(cov_type='cluster',cov_kwds={'groups':d.site});term='Q("'+col+'")';rows.append({'predictor':col,'n':len(d),'coefficient':f.params[term],'se_cluster_site':f.bse[term],'p_value':f.pvalues[term]})
pd.DataFrame(rows).to_csv(O/'residual_predictor_associations.csv',index=False)
print('GEDI',len(g),g.site.nunique());print('Selection',len(a));print('Forecast',len(p));print(pd.read_csv(O/'gedi_within_between_results.csv').to_string(index=False))