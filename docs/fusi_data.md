# Public fUS data

## Caltech fUS-BMI (Griggs et al., Nat. Neurosci. 2024)

| Field | Value |
|---|---|
| Record | https://data.caltech.edu/records/pa710-cdn95 (DOI 10.22002/pa710-cdn95) |
| File | `fUS_BMI_Data.zip`, 34,253,258,547 bytes (34.3 GB) |
| md5 | `6d07baffa0158c9fa9bd57c9c3d5aded` |
| License | CC BY-NC 4.0: non-commercial use only; attribution required |
| Code | https://github.com/wsgriggs2/rt_fUS_BMI (MATLAB analysis code) |

Contents: 2 Hz real-time power Doppler `.mat` files (two rhesus macaques, posterior
parietal cortex, eye and hand movements, 2022-03-04 to 2022-05-22),
`project_record.json` (per-session metadata), `DescriptionOfVariables.pdf`
(the repo README calls it `DescriptionOfFiles.pdf`).

Layout per the repo README: `doppler data/rt_fUS_data_S{session}_R{run}.mat`.

### Variables

Confirmed from rt_fUS_BMI `data loading/load_doppler_data.m` and
`behavior/extract_training_data_and_labels.m`:

- `dop`: (depth, lateral, time) array of 2-D power Doppler frames
- `timestamps`: (n_frames, 3); column 1 = frame time in seconds
- `behavior`: struct array, one element per trial; state fields `trialstart`,
  `initialfixation`, `fixationhold`, `cue`, `memory`, `target_acquire`, `target_hold`,
  `reward`, `iti` with `'HH:MM:SS,mmm'` strings
- also present: `UF`, `neurovascular_map`, `actual_labels`, `predicted_labels`,
  `coreParams`, `frame_num`, `memory_length`, `state_num`, `time_acq`

Assumed (not confirmed): the per-trial direction field name (loader tries `direction`,
`target`, `target_num`, `label`, `angle`); numeric vs string `timestamps`; MAT version
(v7.3 needs `h5py`, not a dependency; behavior is not read for v7.3).
The loader is untested on the real files.

### Download (opt-in)

```bash
uv run python scripts/download_fusi_caltech.py            # info only, no network
uv run python scripts/download_fusi_caltech.py --yes --extract
uv run python scripts/download_fusi_caltech.py --verify-only
```

- target: `$ML_FUSI_DATA_DIR`, default `~/data/ml-fusi/caltech-pa710-cdn95`
- refuses a target inside the repository; data is never committed
- resumable (`.part` + HTTP Range), full-file md5 check, free-space check
- needs ~35 GB free, ~70 GB with `--extract`

### Loading

```python
from ml_platform.workloads.fusi_realtime.realdata import (
    iter_caltech_sessions,
    load_caltech_session,
    to_xarray,
)
```

`doppler` is float32 (t, x, y, z) with y = 1 (imaging plane), `fs_hz` ~ 2.0.

## ConfUSIus open collection

- bucket: `s3://confusius-datasets` (us-west-2), public, no credentials
- list: `aws s3 ls --no-sign-request s3://confusius-datasets/`
- or: `curl -s "https://confusius-datasets.s3.us-west-2.amazonaws.com/?list-type=2"`
- top level (seen 2026-10-25): `last_versions.conf`, `datasets/`, `templates/`
- `datasets/`: `cybis-pereira-2026-bids`, `khallaf-2026-bids`, `landemard-2026-bids`,
  `nunez-elizalde-2022-bids`, `pepe-mariani-2026-bids`, `pereira-2025-bids`
- `templates/` (from `last_versions.conf`): `huang-2025`, `pepe-mariani-2026`,
  `brunner-2021`, `brunner-grillet-2021`
- licenses vary per dataset; check each dataset's license before any use
