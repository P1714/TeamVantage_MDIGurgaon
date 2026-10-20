# Real-world validation data (open, de-identified)

Used only by `madhutwin/realworld.py` to validate MadhuTwin on real people.

| File | Content |
|---|---|
| `cgm.parquet` | patient_id, time, glucose (mg/dL), 5-min CGM |
| `clinical.csv` | static clinical / lab record per participant |

## Sources and licences
- **Hall H, Perelman D, Breschi A, et al.** "Glucotypes reveal new patterns of glucose dysregulation." *PLOS Biology* 16(7): e2005143 (2018). Licence: CC-BY 4.0.
- **Colas A, Vigil L, Vargas B, Cuesta-Frau D, Varela M.** "Detrended Fluctuation Analysis in the prediction of type 2 diabetes mellitus in patients at risk: Model optimization and comparison with other metrics." *PLOS ONE* 14(12): e0225817 (2019). Licence: CC-BY.
- Pre-processed copies from **GlucoBench** (Sergazinov R, et al., ICLR 2024), github.com/IrinaStatsLab/GlucoBench (`raw_data.zip`).

Changes made: participant ids prefixed by cohort, clinical variables renamed, `-1` missing codes set to empty, Colas diagnosis derived from baseline HbA1c / fasting glucose (ADA cut-offs). Colas timestamps keep true intervals but not true clock time, so clock features are left empty for that cohort.
To rebuild: download `raw_data.zip` from GlucoBench, unzip, then `python -c "from madhutwin import realworld as R; R.prepare('raw_data')"`.
