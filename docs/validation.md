# Validation

`validate/` is a uv project (Python 3.12) that downloads Radiance rad5R4 and EPW files, builds fixtures with
ladybug-geometry, computes references (gendaymtx, ladybug Sunpath, RadiationStudy, DirectSunStudy, pvlib) and compares
them with the CLI output layer by layer; `validate/ingest.py` times the mesh transport per format. See
[validate/README.md](../validate/README.md).

## Results (`validate/compare.py`, Boston + Chicago EPWs)

| layer | reference | result |
|---|---|---|
| sun position | ladybug Sunpath (8760 h, 5 sites) / gendaymtx -D / pvlib SPA | 5e-7 deg, 0 day/night flips / 5e-5 deg / 0.015 deg |
| sky matrix | gendaymtx -A -O1 ASCII (rad5R4) | byte-identical rows |
| cumulative radiation, `--mode ladybug` | ladybug-radiance RadiationStudy (faces and vertices, MF1/MF2, north 0/30) | ~1e-6 relative on every sensor after edge-ray decomposition |
| sun hours, `--mode ladybug` | ladybug-radiance DirectSunStudy | exact |
| visibility bits | rcontrib intersection matrix | all mismatches are edge-grazing or zero-cosine rays |
| `--mode physical` vs ladybug | RadiationStudy | 1-4 % mean difference (exact sun vs 4-patch binning, patch vectors), informational |
| USD (python binding) vs OBJ | same scene through both | identical except edge-grazing rays |
| `.dmesh` vs `.obj` | same meshes through both | byte-identical values |
| `--clear-sky` DNI/DHI | pvlib `Location.get_clearsky('ineichen')`, SPA positions, 8760 h, Boston TL 3 | 0.21 % max (sun > 5 deg), 4e-5 on the annual sums; model goldens 1e-6 |
| analytic | sum(DHI + DNI sin alt), sum(DNI cos) on a south wall | < 1e-6 |
