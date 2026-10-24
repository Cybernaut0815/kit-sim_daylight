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
uv run reference.py                      # gendaymtx / Sunpath / pvlib / RadiationStudy / DirectSunStudy + layer 8's rcontrib -> _out/ref/  (~10 min)
uv run reference.py --all                # every EPW in _dl/epw (default: Boston + Chicago)
uv run reference.py --epw --scene canopy # only layer 8's reference: nodni.epw skies + rcontrib at -ad 8192 / 32768 (~1 min)
uv run compare.py selftest               # parser + residual-decomposition unit checks, no CLI needed
uv run compare.py all --exe ../build/RelWithDebInfo/daylight.exe      # layers 1a, 2, 3, 4a, 4b, 4c, 6, 7, 8; exit 1 on hard failures
uv run compare.py bits --exe ... --scene box --epw USA_MA_Boston-City.WSO_TMY
uv run compare.py transmit --exe ...     # layer 8 alone (~3 s)
uv run python -I ingest.py --exe ../build/RelWithDebInfo/daylight.exe       # 1M-quad ingest benchmark -> _out/ingest/, markdown table
uv run python -I golden.py record        # once, on main: byte goldens of ~30 CLI cases -> _out/golden/<case>/ (~20 s)
uv run python -I golden.py check         # rerun them into a temp dir, list every differing file, exit 1 on any
```

Run the scripts with `uv run python -I script.py` if you want the interpreter isolated from the current directory
(everything under `_dl/` is downloaded, untrusted data). The venv also installs `bindings/python` (editable) as
`daylight`: layer 6 converts the USD fixture with `python -I -m daylight usd_xform.usda --sensors /World/Group -o DIR`,
the same command a user runs on their own stage (see `bindings/python/README.md`).

## Layout

| path | content |
|---|---|
| `_dl/` (gitignored) | `Radiance_4ee32974_Windows.zip`, `radiance/` (unzipped; `radiance_path.txt` points at the folder holding `bin/`), `epw/*.epw`, `sha256.txt` |
| `_dl/epw/` | Boston, Ottawa (local copies; Ottawa's short `DESIGN CONDITIONS` line is replaced by `DESIGN CONDITIONS,0` so ladybug can parse it), Chicago TMY3 (NREL/EnergyPlus), Singapore-Changi, Tromso AP-Langnes, New Delhi-Safdarjung (climate.onebuilding.org TMYx), synthesised `chicago_leap.epw` (8784 rows, Feb 29 = copy of Feb 28, header leap flag Yes), `missing.epw` (Boston with rows 4000-4005 radiation = 9999) and `nodni.epw` (Boston with DNI 0 and GHI = DHI: sky only, for the rcontrib layers) |
| `_out/fixtures/<scene>/` | `sensors.obj`, `context.obj` (if any), `fixture.json` (vertices, faces, ladybug face_centroids / face_normals / vertex_normals per mesh; `optics` {mesh: [rho, tau, kind]} where not the core default) |
| `_out/fixtures/usd_xform/` | `usd_xform.usda`: box scene re-authored Y-up, cm, nested rotations, `leftHanded` sensors, mirrored (negative scale) context box, down-facing lids with `purpose = guide`, `active = false` and under an invisible Xform (all three must be pruned); `<prim stem>.dmesh` per mesh + `meshes.json` written here by the Python binding during layer 6 |
| `_out/ref/epw/<stem>/` | `wea.wea` (written by ladybug, x.5 hours), `sky_m{1,2}_{d,s}_O1_A.txt` (`gendaymtx -A` ASCII), `suns_D.npz` (`-D` suns: hoy, vec, rad), `sunpath.npz` (alt, azi, up, vec), `pvlib.npz`, `skymtx_m*_n*.npz` (ladybug per-patch kWh/m2) |
| `_out/ref/scene/<scene>/` | `imtx_m{mf}_n{north}_{face,vertex}.npy` (RadiationStudy intersection matrix, float cos x visible, columns `[sky | ground]`), `rad_<stem>_<tag>.npy`, `sun_imtx_<stem>_*.npy` (bool `[N x n_up]`), `sunhours_<stem>_*.npy` |
| `_out/ref/epw/nodni/` | `wea.wea`, `skysum_m1.npz`: `gendaymtx -m 1 -O1` / `-O0 -of` summed over the 8760 rows (`O1`, `O0` float64 `[146 bins x RGB]`, bin 0 = ground; `rows`) |
| `_out/ref/scene/canopy/` | `scene.rad` / `scene.oct` (sky glow + context polygons with their materials), `dc_ab1_ad{8192,32768}.npy` (rcontrib coefficients float32 `[N x 146 bins x RGB]`) |
| `_out/ref/index.json` | file index with timings |
| `_out/run/<case>/` | CLI outputs per compare case (`out.txt`, `out.txt.json`, `bits.npy`, `sky.txt`, `sun.txt`) |
| `_out/golden/<case>/` | golden.py byte goldens: `out.npy` + `out.npy.json`, `series.npy` + `series.npy.hoys.txt`, `bits.npy` where the case asks; `recorded.txt` = commit + exe they came from. Variants (slot-block `.dmesh`, opaque `--optics`, `--transmission off`, `--bounces 0`) are compared to their base case's files and run once `daylight --help` lists their flags |

Scenes (tri/quad only): `flat` 10x10 m, 1 m quads; `flat_tri`; `walls` N/E/S/W 10x3 m, 0.5 m cells, outward normals,
mesh order north, east, south, west; `box` = flat + two open 4x4x6 m boxes (no bottom face) at (0,6) and (-8,0);
`tilt30` 6x6 m plane tilted 30 deg to the south, 0.5 m cells; `overhang` 6x3 m south wall with a 1.5 m slab on top in
the same mesh (self-shading: the reference passes the mesh itself as context, the CLI runs without `--no-self-occlude`);
`canopy` = flat under a 6x6 m single sheet at 3 m (x -2..4, y -3..3, 1 m quads), porous rho 0.2 tau 0.5;
`glass_canopy` the same sheet as glass tau 0.64; `canyon` two facing 20x10 m walls 10 m apart and the ground between
them (context, opaque), a sensor grid on each wall (1 m cells). The last two are for the reflected-light layer.

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
| 8 transmit | rcontrib `-ab 1` with materials, gendaymtx `-O1` / `-O0` skies of `nodni.epw` | canopy, `--transmission on --patch-samples 64 --sky-density 1 --no-self-occlude` + its `--optics`, radiation and illuminance against the `-ad 32768` run: `\|sum ours / sum ref - 1\|`; per-sensor p95 of `\|ours / ref - 1\|`, limit 3 sigma (rcontrib's per-sensor sigma from `-ad 8192` vs `32768`, noise ~ 1/sqrt(ad)) | 1 %; 3 sigma (info) | yes / info |

Layer 7 needs no EPW reference: the clear-sky weather is synthetic (Ineichen-Perez, Linke turbidity 3, sea level).

Layer 8 builds the fixture for Radiance from `fixture.json`: every context mesh as polygons with the material of its
`optics` (opaque `plastic rho rho rho 0 0`; glass `glass tn tn tn`, tn from Tn at n 1.52 as `src/glass.h`; porous
`trans c c c 0 0 tau/c 1` with c = rho + tau: diffuse reflectance rho, straight transmittance tau), the sky as
`void glow skyglow 0 0 4 1 1 1 0`, `skyglow source sky 0 0 4 0 0 1 180`, `skyglow source ground 0 0 4 0 0 -1 180`, and
the sensors (face centroids, normals) through

```
RAYPATH=_dl/radiance/lib rcontrib -I+ -ab 1 -ad AD -lw 1e-5*8192/AD -e MF:1 -f reinhart.cal -bn Nrbins -b rbin -m skyglow -faf scene.oct
```

(`-e MF:1` before `-f`: reinhart.cal needs MF while loading). A distant glow is never a direct source, so `-ab 1` sees
the sky through the canopy and no reflection, as the CLI without bounces. compare.py multiplies the coefficients by the
summed skies, channels weighted as the core's `bright()`: /1000 for kWh/m2, x 179 / rows for mean lux. Measured on 100
sensors (15 % of the light passes the sheet; opaque or open it would miss by -15 / +15 %): sums +0.036 % radiation,
+0.17 % illuminance; per-sensor p95 0.085 % / 0.22 % against a reference sigma of 0.017 % (rcontrib reseeds every
run: a new reference moves these by about that sigma). The p95 rows are informative because the CLI's resolution is
above that noise: physical mode takes the Perez sky at the NOAA apparent sun, gendaymtx at sun.c (+0.13 % open-sky
illuminance through the efficacy model, -0.08 to +0.03 % radiation per sensor from the sky's shape), tau 0.5 is stored
as 128/255 (+0.39 % of the light through the sheet), and 64 samples per patch differ from 256 by 0.06 % (p95).
Evaluated at the NOAA sun with 1024 samples per patch (a one-off diagnostic port of the overcast Perez sky), the
residual is 0.02 % with a spread of 0.015 %: rcontrib's noise.

Relative errors use `max(|lb|, 1e-3 max|lb|)` as the scale. W per direction is reconstructed from the ladybug
SkyMatrix data (`dir + dif` per sky patch, `rho * mean` for every ground direction) over the ladybug direction order
`[tregenza/reinhart dome vectors | the same reversed]`, rotated by `north`.

The CLI dump formats compare.py reads (`--out` + `.json` manifest, `--dump-bits`, `--dump-sky`, `--dump-sun`) are
listed at the top of `compare.py`; they follow what `src/run.cpp` / `src/main.cpp` write.
