# validate: Ladybug / Radiance / pvlib parity harness for `daylight`

Python 3.12 project managed by [uv](https://docs.astral.sh/uv/). Pins: ladybug-radiance 0.2.12, ladybug-core 0.44.62,
ladybug-geometry 1.35.6, pvlib 0.16.1, usd-core 26.8, numpy. Radiance is **rad5R4** (`Radiance_4ee32974_Windows.zip`),
never a system install: rad6R0P2+ adds a `NAVERAGED=` header line that makes ladybug-radiance 0.2.12 misparse the
sky matrix by one patch.

```
cd validate
uv sync                                  # creates .venv from pyproject.toml / .python-version
uv run setup.py                          # downloads Radiance rad5R4 + EPWs into _dl/, verifies them, writes sha256.txt
uv run fixtures.py                       # scenes -> _out/fixtures/<scene>/{sensors.obj, context.obj, fixture.json} + usd_xform/usd_xform.usda
uv run reference.py                      # gendaymtx / Sunpath / pvlib / RadiationStudy / DirectSunStudy -> _out/ref/  (~10 min)
uv run reference.py --all                # every EPW in _dl/epw (default: Boston + Chicago)
uv run compare.py selftest               # parser + residual-decomposition unit checks, no CLI needed
uv run compare.py all --exe ../build/RelWithDebInfo/daylight.exe      # layers 1a, 2, 3, 4a, 4b, 4c, 6, 7; exit 1 on hard failures
uv run compare.py bits --exe ... --scene box --epw USA_MA_Boston-City.WSO_TMY
uv run python -I ingest.py --exe ../build/RelWithDebInfo/daylight.exe       # 1M-quad ingest benchmark -> _out/ingest/, markdown table
```

Run the scripts with `uv run python -I script.py` if you want the interpreter isolated from the current directory
(everything under `_dl/` is downloaded, untrusted data). The venv also installs `bindings/python` (editable) as
`daylight`: layer 6 converts the USD fixture with `python -I -m daylight usd_xform.usda --sensors /World/Group -o DIR`,
the same command a user runs on their own stage (see `bindings/python/README.md`).

## Layout

| path | content |
|---|---|
| `_dl/` (gitignored) | `Radiance_4ee32974_Windows.zip`, `radiance/` (unzipped; `radiance_path.txt` points at the folder holding `bin/`), `epw/*.epw`, `sha256.txt` |
| `_dl/epw/` | Boston, Ottawa (local copies; Ottawa's short `DESIGN CONDITIONS` line is replaced by `DESIGN CONDITIONS,0` so ladybug can parse it), Chicago TMY3 (NREL/EnergyPlus), Singapore-Changi, Tromso AP-Langnes, New Delhi-Safdarjung (climate.onebuilding.org TMYx), synthesised `chicago_leap.epw` (8784 rows, Feb 29 = copy of Feb 28, header leap flag Yes) and `missing.epw` (Boston with rows 4000-4005 radiation = 9999) |
| `_out/fixtures/<scene>/` | `sensors.obj`, `context.obj` (if any), `fixture.json` (vertices, faces, ladybug face_centroids / face_normals / vertex_normals per mesh) |
| `_out/fixtures/usd_xform/` | `usd_xform.usda`: box scene re-authored Y-up, cm, nested rotations, `leftHanded` sensors, mirrored (negative scale) context box, down-facing lids with `purpose = guide`, `active = false` and under an invisible Xform (all three must be pruned); `<prim stem>.dmesh` per mesh + `meshes.json` written here by the Python binding during layer 6 |
| `_out/ref/epw/<stem>/` | `wea.wea` (written by ladybug, x.5 hours), `sky_m{1,2}_{d,s}_O1_A.txt` (`gendaymtx -A` ASCII), `suns_D.npz` (`-D` suns: hoy, vec, rad), `sunpath.npz` (alt, azi, up, vec), `pvlib.npz`, `skymtx_m*_n*.npz` (ladybug per-patch kWh/m2) |
| `_out/ref/scene/<scene>/` | `imtx_m{mf}_n{north}_{face,vertex}.npy` (RadiationStudy intersection matrix, float cos x visible, columns `[sky | ground]`), `rad_<stem>_<tag>.npy`, `sun_imtx_<stem>_*.npy` (bool `[N x n_up]`), `sunhours_<stem>_*.npy` |
| `_out/ref/index.json` | file index with timings |
| `_out/run/<case>/` | CLI outputs per compare case (`out.txt`, `out.txt.json`, `bits.npy`, `sky.txt`, `sun.txt`) |

Scenes (tri/quad only): `flat` 10x10 m, 1 m quads; `flat_tri`; `walls` N/E/S/W 10x3 m, 0.5 m cells, outward normals,
mesh order north, east, south, west; `box` = flat + two open 4x4x6 m boxes (no bottom face) at (0,6) and (-8,0);
`tilt30` 6x6 m plane tilted 30 deg to the south, 0.5 m cells; `overhang` 6x3 m south wall with a 1.5 m slab on top in
the same mesh (self-shading: the reference passes the mesh itself as context, the CLI runs without `--no-self-occlude`).

## Tolerances (compare.py)

| layer | reference | metric | limit | hard |
|---|---|---|---|---|
| 1a analytic | numpy | flat physical annual vs `sum(DHI + DNI sin alt)/1000` | 1 % | yes |
| 1a analytic | numpy | south wall `--quantity direct` vs `sum(DNI max(0,cos))/1000` | 0.1 % | yes |
| 1a analytic | ourselves | `--north 90` == walls results rotated by one wall | 1e-6 | yes |
| 2 sky | rad5R4 `gendaymtx -m 1/2 -d/-s -O1 -A` | `--dump-sky` RGB text rows differing (byte compare apart from header) | 0 | yes |
| 3 sun | ladybug Sunpath (x.5 hoys) | NOAA alt / azimuth deg; `is_during_day` flips | 1e-6 deg; 0 | yes |
| 3 sun | `gendaymtx -D` | sun.c vector angle; sun-up count (`alt > -0.2665 deg`, night rows included) | 1e-4 deg; exact | yes |
| 3 sun | pvlib `spa_python` | NOAA apparent altitude | 0.02 deg | info |
| 4a bits | RadiationStudy intersection matrix > 0; DirectSunStudy matrix | mismatching bits / all; mismatching rays' distance to a triangle edge (grazing cos < 1e-6 excluded) | 0.1 %; 1e-4 m | yes |
| 4b cumulative | RadiationStudy.radiation_values (MF 1/2, north 0/30, faces/vertices) | non-edge sensors rel err; all sensors residual after bit-difference correction `ours - lb - sum_{d differs}(b_ours - b_lb) cos W` | 0.1 % | yes |
| 4b sun hours | DirectSunStudy | non-edge exact; all sensors | 0 h; 1 h | yes |
| 4c physical | ladybug | `--mode physical` K=1 / K=4 mean rel diff | 5 % | info |
| 6 usd | ourselves | `usd_xform.usda` converted by `python -m daylight` (bindings/python) to `.dmesh` + `meshes.json` vs the box OBJ fixtures, both in `--mode ladybug --cull-backfaces`: sensor count; sensors differing > 1e-5 rel; sky/ground bit differences must be edge-grazing rays; sun bits | equal; 5 %; 1e-4 m; 0 | yes |
| 7 clearsky | pvlib `Location.get_clearsky(model='ineichen', linke_turbidity=3)` on the same 8760 mid-hours (SPA, year 2017) | `--clear-sky --lat 42.35 --lon -71.07 --tz -5` dumped DNI / DHI (`--dump-sun` columns 11-12): max rel diff where pvlib apparent elevation > 5 deg; annual sum rel diff; rows zero on one side and > 1 W/m2 on the other; flat `--quantity irradiance` vs `mean(DHI + DNI sin alt)` | 0.5 %; 0.3 %; 10 (info); 2 % (info) | yes / info |

Layer 7 needs no EPW reference: the clear-sky weather is synthetic (Ineichen-Perez, Linke turbidity 3, sea level).

Relative errors use `max(|lb|, 1e-3 max|lb|)` as the scale. W per direction is reconstructed from the ladybug
SkyMatrix data (`dir + dif` per sky patch, `rho * mean` for every ground direction) over the ladybug direction order
`[tregenza/reinhart dome vectors | the same reversed]`, rotated by `north`.

The CLI dump formats compare.py reads (`--out` + `.json` manifest, `--dump-bits`, `--dump-sky`, `--dump-sun`) are
listed at the top of `compare.py`; they follow what `src/run.cpp` / `src/main.cpp` write.
