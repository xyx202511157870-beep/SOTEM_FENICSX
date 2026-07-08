# COMSOL Uniform Halfspace Validation Design

## Goal

Add a first COMSOL-backed validation path for the existing ATEM3D/SimPEG time-domain electric-source workflow. The initial reference model is:

`COMSOL/均匀半空间.mph`

The validation target is a three-component time response:

- `Ex`
- `Ey`
- `dBzdt`

COMSOL is treated as an external reference source for this first version. The existing solver internals remain unchanged.

## Coordinate Convention

All COMSOL reference data and ATEM3D predictions in this validation path use the same survey coordinate convention:

- `z = 0` is the ground surface.
- `z > 0` points downward into the earth.
- `z < 0` points upward into the air.

Receiver depths, source depths, and exported field components must be interpreted in this `z_down` convention. The adapter will record this convention in output metadata and reject conflicting configuration metadata when it is present.

## Recommended Approach

Use a CSV adapter first:

1. COMSOL solves `均匀半空间.mph`.
2. COMSOL exports a table with columns `time_obs, Ex, Ey, dBzdt`.
3. ATEM3D reads the algorithm prediction from an existing HDF5 result or validation CSV.
4. ATEM3D reads the COMSOL CSV as the reference response.
5. The existing three-component validation writer produces comparison CSV, error CSV, plots, diagnostics, and summary JSON.

This keeps the first version robust and does not require LiveLink, Java API calls, or editing the COMSOL model. A later batch-runner can call `comsolbatch.exe` once the model's study and export tags are known.

## CLI Shape

Add a command under the existing `atem3d.cli` entry point:

```powershell
python -m atem3d.cli validate-comsol-3comp `
  --prediction outputs/your_algorithm_result.h5 `
  --reference COMSOL/均匀半空间_reference.csv `
  --output-dir outputs/comsol_uniform_halfspace_validation
```

The command accepts:

- `--prediction`: ATEM3D result HDF5 containing `times` and `data`, or a CSV with `time_obs, Ex, Ey, dBzdt`.
- `--reference`: COMSOL CSV with `time_obs, Ex, Ey, dBzdt`.
- `--output-dir`: validation artifact directory.
- `--threshold`: optional relative-error threshold, default `0.05`.
- `--coordinate-system`: default `z_down`; only `z_down` is supported for the first version.

## Data Contract

The COMSOL CSV must contain:

```text
time_obs,Ex,Ey,dBzdt
```

Rules:

- `time_obs` is in seconds and must be positive.
- `Ex` and `Ey` are electric-field components in the shared coordinate frame.
- `dBzdt` is the vertical magnetic-flux-density time derivative in the shared coordinate frame.
- No implicit unit conversion is applied in the first version.
- Prediction and reference times must match exactly, unless an explicit future interpolation option is added.
- Component order is normalized internally to `["Ex", "Ey", "dBzdt"]`.

## Output Artifacts

The command writes the same artifact family used by current three-component validation:

- `predictions.csv`
- `reference_comsol.csv`
- `errors.csv`
- `error_summary.json`
- `diagnostics.json`
- `run_config_resolved.yaml`
- `comparison_3comp.png`
- `error_curves_3comp.png`

The validation metadata uses `reference_type: comsol_uniform_halfspace` and `validation_scope: diagnostic_comsol_reference`. This prevents the COMSOL check from being mistaken for the existing final-acceptance `empymod` or `1d` gates.

Implementation must register `comsol_uniform_halfspace` as a diagnostic reference type in the three-component validation layer, not as a final-acceptance reference type.

## Error Handling

The command fails fast when:

- Required columns are missing.
- Times are nonpositive, unsorted, duplicated, or not aligned between prediction and reference.
- The prediction result does not expose three columns matching `Ex`, `Ey`, and `dBzdt`.
- Metadata declares a coordinate system other than `z_down`.
- Any values are NaN or infinite.

Error messages should name the file and the failed contract.

## Testing

Add focused tests for:

- Reading a valid COMSOL CSV.
- Rejecting missing columns.
- Rejecting nonpositive or mismatched times.
- Comparing synthetic prediction/reference arrays through the three-component artifact writer.
- Preserving the `z_down` coordinate convention in resolved metadata.

No COMSOL installation is required for unit tests. Batch execution of the `.mph` model remains a manual or later integration step.

## Future Extension

After the CSV path is stable, add optional automation:

```powershell
comsolbatch.exe -inputfile COMSOL/均匀半空间.mph -outputfile outputs/comsol_uniform_halfspace/solved.mph
```

That extension should only be added after confirming the COMSOL model's study tag, dataset tag, and export table tag, so the automation can run without modifying the original `.mph` file.
