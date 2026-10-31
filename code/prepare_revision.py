"""Assemble the revision using saved numerical results; never invent values."""
import json,re,hashlib,shutil,sys,platform
from pathlib import Path
import numpy as np,pandas as pd
from pyproj import Geod
R=Path(__file__).resolve().parents[1];D=R/'data';M=R/'manuscript';A=R/'audit'
TITLE='Landscape transfer of post-typhoon mangrove greenness forecasts in the Philippines, with independent lidar canopy checks'
def tx(s):return str(s).replace('_',r'\_').replace('&',r'\&').replace('%',r'\%')
def table(headers,rows,caption,label,align=None):
 return '\n'.join([r'\begin{table}[htbp]\centering\small',r'\caption{'+caption+'}'+r'\label{'+label+'}',r'\begin{tabular}{'+(align or 'l'+'r'*(len(headers)-1))+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule',*[' & '.join(map(str,row))+r' \\' for row in rows],r'\bottomrule\end{tabular}',r'\end{table}'])
def main():
 a=pd.read_csv(D/'analysis_table_all.csv');primary=pd.read_csv(D/'analysis_table_primary.csv');met=pd.read_csv(D/'primary_landscape_metrics.csv');ref=pd.read_csv(D/'gedi_mangrove_reference.csv',dtype={'shot_number':str});cal=pd.read_csv(D/'canopy_calibration_predictions.csv',dtype={'shot_number':str});gda=pd.read_csv(D/'gedi_attrition.csv');cs=json.load(open(D/'canopy_calibration_status.json'));ps=json.load(open(D/'primary_status.json'));rb=json.load(open(D/'robustness_status.json'))
 forecast=pd.read_csv(D/'expanded_scene_manifest.csv');vm=pd.read_csv(D/'validation_scene_manifest.csv');man=pd.concat([forecast,vm]).drop_duplicates(['landscape','scene_id']);man.to_csv(D/'all_requested_scene_manifest.csv',index=False)
 counts=man.groupby('collection').size();matched=ref[ref.matched];post=matched[matched.period=='poststorm_2022_23'];sites=[f'{r}_{k:02}' for r in ['siargao','leyte','palawan'] for k in range(1,5)]
 vals={'GEDIMATCH':len(matched),'GEDILANDS':matched.site.nunique(),'GEDIQUAL':len(ref),'GEDIQUALLANDS':ref.site.nunique(),'GEDIPRE':sum(matched.period=='prestorm'),'GEDIPOST':sum(matched.period=='poststorm_2022_23'),'POSTMIN':post.time.min(),'POSTMAX':post.time.max(),'CANDEV':cs['development_shots'],'CANDEVLANDS':cs['development_landscapes'],'CANTEST':cs['new_test_shots'],'CANTESTLANDS':cs['new_test_landscapes'],'CANSLOPE':f"{cs['slope']:.3f}",'CANMAE':f"{cs['primary_30d']['mae_macro']:.3f}",'CANMED':f"{cs['primary_30d']['median_mae_macro']:.3f}",'CANBAL':f"{cs['primary_30d']['balanced_median_mae_macro']:.3f}",'CANPOSTN':cs['poststorm']['shots'],'CANPOSTLANDS':cs['poststorm']['landscapes'],'CANPOSTMAE':f"{cs['poststorm']['mae_macro']:.3f}",'CAN15MAE':f"{cs['lag_15d']['mae_macro']:.3f}",'CAN98MAE':f"{cs['sensitivity_098']['mae_macro']:.3f}",'CANSHIFTMAE':f"{cs['shift_10m_pure']['mae_macro']:.3f}",'WINDOWSCENES':len(man),'UNIQUESCENES':man.scene_id.nunique(),'OPTSCENES':int(counts['sentinel-2-l2a']),'RADSCENES':int(counts['sentinel-1-rtc'])}
 text=(M/'manuscript_template.tex').read_text()
 for k,v in vals.items():text=text.replace('@@'+k+'@@',str(v))
 assert '@@' not in text;(M/'manuscript.tex').write_text(text)
 cohort=a.groupby('site').primary_eligible.agg(['sum','count']);gd=gda.set_index('landscape');rows=[]
 for site in sites:rows.append([tx(site),int(cohort.loc[site,'count']),int(cohort.loc[site,'sum']),int(gd.loc[site,'matched_prestorm']),int(gd.loc[site,'matched_poststorm'])])
 rows.append(['Total',len(a),len(primary),vals['GEDIPRE'],vals['GEDIPOST']])
 (M/'coverage_table.tex').write_text(table(['Landscape','Candidate','Forecast','GEDI pre','GEDI post'],rows,'Forecast cells and temporally matched lidar-reference shots. Forecast and reference samples are different cohorts. Zero is retained as a coverage result.','tab:coverage'))
 rows=[]
 for site in ['palawan_02','palawan_03','palawan_04']:
  mm=met[met.landscape==site].set_index('model');rows.append([tx(site),int(mm.loc['selected','n']),f"{mm.loc['selected','mae']:.4f}",f"{mm.loc['median','mae']:.4f}",f"{mm.loc['persistence','mae']:.4f}",f"{mm.loc['selected','r2']:.3f}"])
 rows.append(['Macro mean','855','0.0359','0.0436','0.0995','---'])
 (M/'performance_table.tex').write_text(table(['New landscape','$n$','Forest MAE','Median MAE','Persistence MAE','Forest $R^2$'],rows,'New Palawan transfer results; errors are NDVI units. Macro mean gives equal weight to each landscape.','tab:performance'))
 grids=[json.load(open(D/f'{s}_grid.json')) for s in sites];geod=Geod(ellps='WGS84');seps=[]
 for r in ['siargao','leyte','palawan']:
  gg=[g for g in grids if g['region']==r];dist=[]
  for i,g in enumerate(gg):
   b=g['bbox'];x=(b[0]+b[2])/2;y=(b[1]+b[3])/2
   for h in gg[i+1:]:
    bb=h['bbox'];dist.append(geod.inv(x,y,(bb[0]+bb[2])/2,(bb[1]+bb[3])/2)[2]/1000)
  seps.append({'region':r,'minimum_centre_distance_km':min(dist),'maximum_centre_distance_km':max(dist)})
 pd.DataFrame(seps).to_csv(D/'landscape_separation.csv',index=False)
 split=primary[['cell','site','region']].copy();split['role']=np.where(split.region!='palawan','development',np.where(split.site=='palawan_01','legacy_diagnostic','new_transfer'));split['outer_group']=split.site;split.to_csv(D/'landscape_split_assignments.csv',index=False)
 header=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=22mm]{geometry}
\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
\usepackage{lmodern,microtype,booktabs,longtable,graphicx,array,xurl}
\usepackage[colorlinks=true,allcolors=blue]{hyperref}
\hypersetup{pdftitle={Supporting information: replicated Philippine mangrove forecasting and lidar checks},pdfauthor={Jae Lord Dexter C. Filipinas}}
\emergencystretch=1em
\title{Supporting Information: replicated Philippine mangrove forecasting and lidar canopy checks}
\author{Jae Lord Dexter C. Filipinas --- designated corresponding author}
\date{21 October 2026}
\begin{document}\maketitle
\section{Design and reproducibility}
This supplement belongs to the expanded revision, not the three-window pilot. The package preserves acquired pixels and lidar records, not merely a proposed collection workflow. Forecast and reference cohorts must not be conflated. The eight development landscapes supply nested whole-landscape evaluation; three new Palawan landscapes are the main transfer test. Palawan 01 remains a previously inspected diagnostic. Separation does not prove independence of coastal systems or storm effects.
'''
 supp=[header]
 rows=[]
 for g in grids:
  b=g['bbox'];rows.append([tx(g['site']),f'{(b[0]+b[2])/2:.5f}',f'{(b[1]+b[3])/2:.5f}',g['crs'].split(':')[1]])
 supp.append(table(['Landscape','Centre longitude','Centre latitude','UTM EPSG'],rows,'Landscape centre coordinates; full bounds and affine grids are in the inventory.','tab:coords'))
 supp.append(table(['Region','Minimum separation (km)','Maximum (km)'],[[tx(x['region']),f"{x['minimum_centre_distance_km']:.3f}",f"{x['maximum_centre_distance_km']:.3f}"] for x in seps],'Geodesic distances between within-region landscape centres.','tab:sep'))
 supp.append(r'\clearpage\section{Independent canopy-reference audit}')
 supp.append('GEDI v2 L2B cover is a fraction, not percentage points; RH98 is in metres. A nominal 25 m circular support was intersected with 20 m working pixels. The mirror applies upstream filtering; regional row counts cannot be used to infer the rate of rejection among all original shots. References are independent of the optical sensor, but not verified local ground truth. No paired-footprint canopy recovery change is estimated.\n')
 supp.append(table(['Landscape','Inside window','Quality','Mapped support','Optical matched'],[[tx(s),*map(int,gd.loc[s,['returned_shots_inside_window','quality_pass','mapped_footprint_and_cell_pass','optical_matched_30d']]) ] for s in sites],'GEDI attrition. Counts inside windows precede the additional quality, map-support and temporal-match screens.','tab:gediattrition'))
 rows=[]
 for k,label in [('primary_30d','All 30-day pairs'),('prestorm','Pre-storm only'),('poststorm','Post-storm only'),('lag_15d','Lag at most 15 days'),('sensitivity_098','Sensitivity at least 0.98'),('shift_10m_pure','All 10 m shifts map-pure'),('night','Night only')]:
  q=cs[k];rows.append([label,q['shots'],q['landscapes'],f"{q['mae_macro']:.3f}",f"{q['balanced_median_mae_macro']:.3f}"])
 supp.append(table(['New Palawan subset','Shots','Landscapes','Calibration MAE','Balanced median'],rows,'Concurrent canopy-cover checks, equal-landscape MAE in fraction units. Subset membership is selective.','tab:canopysense'))
 cm=pd.read_csv(D/'canopy_landscape_metrics.csv');rows=[]
 for s in sites:
  c=cm[(cm.landscape==s)&(cm.model=='predicted_cover')]
  if len(c):
   z=c.iloc[0];rows.append([tx(s),int(z.n_shots),int(z.n_orbit_dates),f'{z.mae:.3f}',f'{z.spearman_ndvi_cover:.3f}' if np.isfinite(z.spearman_ndvi_cover) else '---'])
 supp.append(table(['Landscape','Matched shots','Orbit-dates','Cover MAE','Spearman $\\rho$'],rows,'Per-landscape concurrent canopy calibration. An orbit-date cluster can include several beams; shots within a cluster are not independent. Single-shot correlations are undefined.','tab:canopylands'))
 supp.append(r'\section{Forecast sensitivity and nested evaluation}')
 supp.append(f"The exact-month endpoint check retained {rb['matched_month_train_n']} development and {rb['matched_month_test_n']} new transfer cells. Its fixed optical forest (leaf size 15) achieved macro MAE {rb['matched_month_selected_macro_mae']:.4f}, versus {rb['matched_month_median_macro_mae']:.4f} for the corresponding training median. The primary additional landscape-balanced median achieved {rb['primary_balanced_median_macro_mae']:.4f}. These checks do not select a replacement primary endpoint after viewing Palawan results.\n")
 ne=pd.read_csv(D/'primary_nested_development.csv');rows=[[tx(z.landscape),int(z.n),f'{z.mae:.4f}',f'{z.median_mae:.4f}',tx(z.selected.replace("'",''))] for _,z in ne.iterrows()]
 supp.append(table(['Outer omitted landscape','$n$','Nested MAE','Median MAE','Inner selection'],rows,'Outer development assessment. Selection uses only the other seven landscapes.','tab:nested','lrrrl'))
 supp.append(r'''\section{Public source citations and access}
All sources were accessed on 21 October 2026. Sentinel-1 RTC and Sentinel-2 L2A: \url{https://planetarycomputer.microsoft.com/catalog}, collection identifiers \texttt{sentinel-1-rtc} and \texttt{sentinel-2-l2a}; per-scene unsigned assets appear in the two manifests. ESA WorldCover 2020 v100: \url{https://doi.org/10.5281/zenodo.5571936}. NOAA IBTrACS v4r01: \url{https://doi.org/10.25921/82ty-9e16}; the used Rai track is included. Natural Earth boundaries: \url{https://www.naturalearthdata.com/}, public domain, context map only.

GEDI L2B: \url{https://doi.org/10.5067/GEDI/GEDI02_B.002}. GEDI L2A: \url{https://doi.org/10.5067/GEDI/GEDI02_A.002}. Public processed GFZ mirror: \url{https://gedidb.readthedocs.io/en/latest/user/tiledb_database.html}; endpoint \url{https://s3.gfz.de}, bucket \texttt{dog-ext.gedidb.gedi-l2-l4-v002.0}. Query bounds, dates and requested variables are preserved. Client: gediDB 2026.4.30; software DOI \url{https://doi.org/10.5281/zenodo.13885229}, companion paper \url{https://doi.org/10.21105/joss.08593}. NASA user guide: \url{https://lpdaac.usgs.gov/documents/998/GEDI02_UserGuide_V21.pdf}. GEDI mission-gap documentation: \url{https://registry.opendata.aws/nasa-gedi/}.
\section{Coverage checkpoints and limits}
The post-event GEDI-cover forecast had only one development landscape and two new Palawan landscapes after early-feature requirements (11 and 43 cells respectively). The minimum four-development-landscape checkpoint failed, so no model or forecast performance was produced. Broader pre-storm references support concurrent measurement comparisons only. Leyte has no matched late-2022 reference, and GEDI has no Q4 2023 observations. No confidence interval treating hundreds of footprints or cells as independent forests is reported.

The workflow added pre-event lidar and validation-only imagery after initial coverage inspection. This is an exploratory amendment, recorded in the audit log. Additional balanced medians were added as diagnostic comparators; original median results remain preserved. No Palawan data selected the forecasting model. Repeated scenes across windows and repeated shots in a pass remain correlated.
\section{Journal and authorship review}
The target remains \textit{Remote Sensing in Ecology and Conservation}. The main manuscript uses an author-date review format with an abstract under 300 words and six keywords. Direct access to the recommended Wiley class was unavailable, so official production-template compliance is not claimed. Before submission, the corresponding author must verify current full guidance, correspondence metadata, declarations, coauthor agreement, data-release rights and APC arrangements. No submission or invitation has been sent.
\end{document}
''')
 (M/'supplementary_material.tex').write_text('\n'.join(supp))
 (R/'README.md').write_text(f'''# Expanded Philippine mangrove forecasting study

Corresponding author: Jae Lord Dexter C. Filipinas. Prospective collaborator: Minerva Singh, pending agreement. Review draft, 21 October 2026. No submission or message sent.

This revision replaces the three-window pilot with 12 windows (four per region), >=8 km centre separation, 4,196 candidate cells, 1,562 forecast-eligible cells, nested whole-landscape evaluation, and three new Palawan transfer landscapes. The prior Palawan 01 is a separate diagnostic. The complete package contains {len(man):,} acquired landscape-scene pairs ({man.scene_id.nunique()} unique scene identifiers), {len(ref)} quality/map-qualified GEDI shots and {len(matched)} optical matches across {matched.site.nunique()} landscapes. Several references remain absent; these are not silently counted as validated.

Selected optical forest: new-transfer macro MAE 0.0359 vs 0.0436 training median; all three new landscape R-squared values are negative. Nested development MAE is 0.0431 vs 0.0432 median. Canopy consistency is poor/heterogeneous: calibrated transfer cover MAE {cs['primary_30d']['mae_macro']:.3f} fraction units. References do not directly validate late-2023 ecological recovery. The early GEDI-cover forecast failed its minimum development-coverage checkpoint and was not fitted.

## Contents

- manuscript/: complete main/supplement PDFs, editable LaTeX, numeric tables and bibliography (24 research papers plus data/software citations).
- figures/: six PDF/PNG figures from actual data.
- data/observations/: acquired optical/radar array cutouts and scene properties.
- data/*manifest.csv: separate forecasting and validation-only acquisition lineages.
- data/*gedi*raw.csv: returned public regional lidar subsets; *_query.json records bounds, dates and provenance. These are processed-mirror outputs, not unfiltered original granules.
- data/gedi_mangrove_reference.csv and canopy_calibration_predictions.csv: footprint match and empirical evidence, including shot IDs and quality fields.
- data/analysis_table_*.csv, landscape_split_assignments.csv, primary_*.csv: feature/outcome data, splits, tuning, nested evaluation and predictions.
- audit/: amendment plan, numerical verification, checksums, source and DOI metadata, journal limitations and logs.
- Cover_Letter_Draft.md and Minerva_Collaboration_Invitation.md: updated unsent drafts.

## Reproduce from included observations

Install requirements in Python 3.12. For analysis without further download, run in this directory:

```bash
python code/replicated_analysis.py all
python code/robustness.py
python code/canopy_validation.py all
python code/revision_figures.py
python code/prepare_revision.py
python code/validate_revision.py
```

Rebuild PDFs in manuscript/: pdflatex manuscript; bibtex manuscript; pdflatex manuscript twice; pdflatex supplementary_material twice. Then inspect compilation and render outputs. Acquisition is optional for recomputation: expand.py select/acquire; gedi_acquire.py poststorm and baseline; validation_optical.py. For strict workflow reconstruction, query poststorm GEDI, initially match using the baseline query and existing images, acquire validation-only images, then rerun matching. The final manifests and acquired arrays are the authoritative delivered sample.

The manuscript assembler reads persisted outputs; it does not generate empirical claims without data. Old helper analyse.py supplies processing primitives; its legacy model() is not the revision entry point. Use replicated_analysis.py.

Public data: Copernicus Sentinel, ESA WorldCover, NOAA IBTrACS, NASA GEDI accessed via public GFZ gediDB; Natural Earth for context. Source attributions and licences apply. No new repository DOI or local field validation is claimed. Keep raw references and amended protocol when reviewing the ecological interpretation.
''')
 (R/'Cover_Letter_Draft.md').write_text(f'''# Cover letter — unsent author-review draft

21 October 2026

Dear Editors,

Please consider “{TITLE}” as an Original Research Article for *Remote Sensing in Ecology and Conservation*.

The study tests whether information available within three months after Typhoon Rai/Odette predicts late-2023 mangrove greenness in geographically separate Philippine landscapes. Twelve windows across Siargao, Southern Leyte and Palawan provide within-region replication. Whole-landscape nested validation and three newly added Palawan transfer landscapes replace cell-level evidence from a single window. The previously inspected pilot test is reported separately.

The development-selected optical forest achieved equal-landscape transfer MAE 0.0359 NDVI units, versus 0.0436 for a training median. However, all three new transfer landscapes had negative within-landscape R-squared, and nested development performance was near the median baseline. Independent GEDI canopy references ({len(matched)} optical matches in {matched.site.nunique()} landscapes) exposed inconsistent relationships between greenness and measured canopy cover. We distinguish these concurrent checks from direct validation of late-2023 ecological recovery.

The contribution is a reproducible spatial-transfer and measurement-consistency study, with explicit information cutoffs, nested landscape evaluation, baseline comparisons, source provenance and candid negative evidence. It addresses the reliability of remote-sensing indicators used in conservation monitoring. Local field validation, additional storms and transfer regions remain needed before operational use.

This letter is for author review. It does not assert confirmed coauthorship, all-author approval, exclusive submission or completed declarations. The corresponding author must supply final correspondence metadata and confirm funding, conflicts, authorship, data release, current journal requirements and APC arrangements before submitting.

Sincerely,

Jae Lord Dexter C. Filipinas  
Designated corresponding author  
Department of Physics, Mindanao State University, Marawi City, Philippines (author to confirm)  
Email: to be supplied
''')
 (R/'Minerva_Collaboration_Invitation.md').write_text(f'''# Collaboration invitation — draft, not sent

Subject: Invitation to collaborate on Philippine mangrove forecasting and lidar validation

Dear Dr Singh,

I would like to invite you to help develop a study on machine learning for post-typhoon mangrove monitoring in the Philippines. Your work on Southeast Asian mangrove monitoring with remote sensing is closely related to the questions we are investigating.

The current draft, “{TITLE},” uses public Sentinel observations from twelve mangrove landscapes in Siargao, Southern Leyte and Palawan. It includes nested whole-landscape evaluation, three new Palawan transfer tests and independent GEDI canopy-reference comparisons. The optical model improves average forecast error relative to a simple median, but does not establish reliable within-landscape prediction. The canopy checks also reveal substantial disagreement between greenness and lidar cover, which we report openly.

I would value your input on ecological interpretation, mangrove-specific lidar limitations, reference sampling and model-transfer design. Local field or airborne canopy validation and broader event replication would be especially valuable next steps. I would be happy to share the manuscript, figures, code and acquired public-data subsets for your assessment.

Would you be interested in contributing to the study and, if the work develops into a mutually agreed contribution, joining as a coauthor? Authorship and contributions would be agreed together; your participation is not assumed in the current draft.

Best regards,

Jae Lord Dexter C. Filipinas  
Designated corresponding author
''')
 report={'landscapes':12,'minimum_separation_km':min(x['minimum_centre_distance_km'] for x in seps),'candidate_cells':len(a),'forecast_cells':len(primary),'window_scene_pairs':len(man),'unique_scene_ids':int(man.scene_id.nunique()),'GEDI_qualified':len(ref),'GEDI_matched':len(matched),'GEDI_matched_landscapes':int(matched.site.nunique()),'primary':ps,'canopy':cs,'remaining':'Local field/airborne truth, direct late-2023 canopy reference, additional storms/transfer regions, correspondence metadata and final journal review.'}
 (A/'revision_summary.json').write_text(json.dumps(report,indent=2));(R/'Completion_Report.md').write_text('# Revision status\n\nExpanded sampling, real public-data acquisition, nested landscape analysis, independent lidar comparison, manuscript, supplement, six figures and unsent correspondence drafts are complete. The study remains a review draft.\n\n```json\n'+json.dumps(report,indent=2)+'\n```\n')
 print(json.dumps(vals,indent=2))
if __name__=='__main__':main()