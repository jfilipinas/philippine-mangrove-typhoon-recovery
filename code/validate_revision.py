"""Scientific consistency checks of the delivered empirical revision."""
import json,re,hashlib,sys,platform
from pathlib import Path
import pandas as pd,numpy as np
from sklearn.metrics import mean_absolute_error,r2_score
R=Path(__file__).resolve().parents[1];D=R/'data';A=R/'audit'
def main():
 checks={};man=pd.read_csv(D/'all_requested_scene_manifest.csv');paths=[D/'observations'/z.landscape/(z.scene_id+'.npz') for _,z in man.iterrows()];checks['every_requested_cutout_present']=all(p.is_file() for p in paths);checks['unique_requested_pairs']=not man.duplicated(['landscape','scene_id']).any()
 a=pd.read_csv(D/'analysis_table_all.csv');p=pd.read_csv(D/'analysis_table_primary.csv');spl=pd.read_csv(D/'landscape_split_assignments.csv');pred=pd.read_csv(D/'primary_predictions.csv');status=json.load(open(D/'primary_status.json'));sep=pd.read_csv(D/'landscape_separation.csv')
 checks['twelve_landscapes_four_per_region']=a.groupby('region').site.nunique().eq(4).all();checks['minimum_geodesic_separation_8km']=bool((sep.minimum_centre_distance_km>=8).all());checks['candidate_and_complete_case_counts']=len(a)==4196 and len(p)==1562 and p.primary_eligible.all();checks['development_landscapes_8']=spl[spl.role=='development'].site.nunique()==8;checks['fresh_transfer_landscapes_3']=set(spl[spl.role=='new_transfer'].site)=={'palawan_02','palawan_03','palawan_04'};checks['legacy_test_not_development_or_fresh']=set(spl[spl.site=='palawan_01'].role)=={'legacy_diagnostic'}
 table=pd.read_csv(D/'primary_landscape_metrics.csv');valid=True
 for _,row in table.iterrows():
  dd=pred[pred.site==row.landscape];v=mean_absolute_error(dd.observed,dd[row.model]);valid &= bool(np.isclose(v,row.mae,atol=1e-12))
 checks['per_landscape_MAE_recomputed']=valid
 for name in ['selected','median','persistence']:
  z=table[(table.role=='new_transfer')&(table.model==name)].mae.mean();checks['macro_MAE_'+name]=bool(np.isclose(z,status['test_macro_mae'][name],atol=1e-12))
 checks['all_three_new_R2_negative']=bool((table[(table.role=='new_transfer')&(table.model=='selected')].r2<0).all())
 ref=pd.read_csv(D/'gedi_mangrove_reference.csv',dtype={'shot_number':str});cal=pd.read_csv(D/'canopy_calibration_predictions.csv',dtype={'shot_number':str});cs=json.load(open(D/'canopy_calibration_status.json'));rawids=set()
 for path in D.glob('*gedi*raw.csv'):rawids.update(pd.read_csv(path,usecols=['shot_number'],dtype=str).shot_number)
 checks['GEDI_shot_identifiers_preserved_unique']=not ref.shot_number.duplicated().any() and ref.shot_number.str.fullmatch(r'\d{17,18}').all() and set(ref.shot_number).issubset(rawids)
 checks['GEDI_flags_and_sensitivity']=bool(((ref.quality_flag==1)&(ref.l2a_quality_flag==1)&(ref.l2b_quality_flag==1)&(ref.degrade_flag==0)&ref.sensitivity.between(.95,1)).all());checks['GEDI_fraction_units_and_purity']=bool(ref.cover.between(0,1).all() and (ref.footprint_mangrove_fraction>=.7-1e-9).all());checks['temporal_match_and_clear_support']=bool(cal.lag_days.between(0,30).all() and (cal.clear_mapped_footprint_fraction>=.7-1e-9).all());checks['prestorm_matching_does_not_cross_event']=bool((pd.to_datetime(cal[cal.period=='prestorm'].optical_date)<'2021-12-16').all());checks['no_GEDI_Q42023_reference_claim']=bool((pd.to_datetime(ref.time)<'2023-03-17').all());checks['calibration_sample_counts']=len(cal[~cal.site.str.startswith('palawan')])==cs['development_shots'] and len(cal[cal.site.isin(['palawan_02','palawan_03','palawan_04'])])==cs['new_test_shots']
 new=cal[cal.site.isin(['palawan_02','palawan_03','palawan_04'])];cm=new.groupby('site').apply(lambda z:np.abs(z.cover-z.predicted_cover).mean(),include_groups=False).mean();checks['canopy_macro_MAE_recomputed']=bool(np.isclose(cm,cs['primary_30d']['mae_macro'],atol=1e-12));blocked=json.load(open(D/'canopy_forecast_status.json'));checks['insufficient_canopy_forecast_not_estimated']=blocked['status']=='insufficient_landscape_coverage'
 text=(R/'manuscript/manuscript.tex').read_text();bib=(R/'manuscript/references.bib').read_text();abstract=text.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0];checks['abstract_max_300_words']=len(abstract.split())<=300;checks['maximum_25_papers']=bib.count('@article')<=25;checks['no_unresolved_template_tokens']='@@' not in text
 # Forecast-derived tables only use the frozen forecasting manifest. Validation-only scenes are excluded by observations().
 obs=pd.read_csv(D/'cell_observations.csv',usecols=['site','scene','date']);fm=pd.read_csv(D/'expanded_scene_manifest.csv');allowed=set(zip(fm.landscape,fm.scene_id));checks['validation_only_scenes_excluded_from_forecast']=set(zip(obs.site,obs.scene)).issubset(allowed)
 hashes=[]
 for pth in paths:hashes.append({'file':str(pth.relative_to(R)),'bytes':pth.stat().st_size,'sha256':hashlib.sha256(pth.read_bytes()).hexdigest()})
 for pth in D.glob('*gedi*raw.csv'):hashes.append({'file':str(pth.relative_to(R)),'bytes':pth.stat().st_size,'sha256':hashlib.sha256(pth.read_bytes()).hexdigest()})
 pd.DataFrame(hashes).to_csv(A/'data_checksums.csv',index=False)
 result={'status':'passed' if all(checks.values()) else 'failed','checks':{k:bool(v) for k,v in checks.items()},'cutouts':len(paths),'matched_GEDI_shots':len(cal),'research_papers':bib.count('@article'),'abstract_words':len(abstract.split()),'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},'limits':'Checks establish numerical/provenance consistency, not ecological accuracy or full statistical independence.'};(A/'validation_report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));assert all(checks.values())
if __name__=='__main__':main()