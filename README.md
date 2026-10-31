# Philippine mangrove post-typhoon forecasting

Research associated with *Limits of landscape transfer in post-typhoon mangrove greenness forecasts: independent lidar and observation-selection diagnostics*.

**Authors:** Jae Lord Dexter Filipinas and Minerva Singh.

## Repository contents

- `code/`: original acquisition, processing, model evaluation, validation, figure-generation, and post-hoc diagnostic scripts.
- `data/analysis_table_primary.csv`: complete primary cell-level analysis table.
- `data/primary_predictions.csv`: original held-out model predictions.
- `data/landscape_split_assignments.csv`: development, legacy and new-transfer split assignments.
- `data/gedi_mangrove_reference.csv`: quality-filtered GEDI reference observations and optical matching.
- `data/canopy_calibration_predictions.csv`: canopy calibration outputs.
- `data/rai_2021_track.csv`: cyclone best-track information.
- `data/*status.json` and `results/`: model selection, diagnostics, and statistical summaries.
- `documentation/audit/`: source provenance, validation and software-version records.

## Scientific interpretation

The optical forest improves average transfer MAE over the training median but does not establish useful within-landscape discrimination: all three new Palawan landscapes have negative within-landscape R². The within-landscape NDVI–GEDI slope is statistically uncertain, and the matched GEDI observations do not validate late-2023 structural recovery.

## Reproduction status

The code and derived tables were copied from the existing research package. **They have not been rerun end-to-end or independently verified after upload.** Some scripts require the original `data/cell_observations.csv`, scene acquisition cutouts, source manifests and external data services. Those large inputs have not all been uploaded. Therefore, this repository is **not yet a complete reproducibility deposit**.

Before journal submission, verify exact dependencies, dataset redistribution terms, code execution, all reported numerical results, and the public archive's completeness. The repository currently has no assigned archival DOI.

## Data sources

Sentinel-1/2, GEDI, ESA WorldCover, and IBTrACS data retain their respective provider attribution and license terms. This repository does not grant additional rights to redistribute third-party observations.

## Additional archived inputs

The complete original research package and a GitHub-ready ZIP have been prepared separately. Large acquired observation cutouts and the approximately 56 MB `cell_observations.csv` require a suitable archive such as Zenodo or Git LFS rather than the browser uploader.

