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
uv run reference.py                      # gendaymtx / Sunpath / pvlib / RadiationStudy / DirectSunStudy + layers 8 / 9's rcontrib -> _out/ref/  (~12 min)
uv run reference.py --all                # every EPW in _dl/epw (default: Boston + Chicago)
uv run reference.py --epw --scene canopy # only layer 8's reference: nodni.epw skies + rcontrib -ab 1..3 at -ad 8192 / 32768 (~30 s)
uv run reference.py --scene canopy glass_canopy canopy_ground glass_canopy_ground canyon plane   # layers 8 and 9: + the --epw skies (~5 min, 6 rcontrib at a time)
uv run compare.py selftest               # parser + residual-decomposition unit checks, no CLI needed
uv run compare.py all --exe ../build/RelWithDebInfo/daylight.exe      # layers 1a, 2, 3, 4a, 4b, 4c, 6, 7, 8, 9; exit 1 on hard failures
uv run compare.py bits --exe ... --scene box --epw USA_MA_Boston-City.WSO_TMY
uv run compare.py transmit --exe ...     # layer 8 alone (~3 s)
uv run compare.py reflect --exe ...      # layer 9 alone (~30 s)
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
| `_out/ref/epw/<stem>/skysum_m1.npz` | nodni and the `--epw` EPWs (with a DC scene): `gendaymtx -m 1 -O1` / `-O0 -of` (sun binned) summed over the 8760 rows (`O1`, `O0` float64 `[146 bins x RGB]`, bin 0 = ground; `rows`) |
| `_out/ref/scene/{canopy,glass_canopy,canopy_ground,glass_canopy_ground,canyon,plane}/` | `scene.rad` / `scene.oct` (sky glow + context polygons with their materials), `dc_ab{1,2,3}_ad{8192,32768}.npy` (rcontrib coefficients float32 `[N x 146 bins x RGB]`) |
| `_out/ref/index.json` | file index with timings |
| `_out/run/<case>/` | CLI outputs per compare case (`out.txt`, `out.txt.json`, `bits.npy`, `sky.txt`, `sun.txt`) |
| `_out/golden/<case>/` | golden.py byte goldens: `out.npy` + `out.npy.json`, `series.npy` + `series.npy.hoys.txt`, `bits.npy` where the case asks; `recorded.txt` = commit + exe they came from. Variants (slot-block `.dmesh`, opaque `--optics` with `--transmission` off and on, translucent rows with `--transmission off`, `--bounces 0`) are compared to their base case's files and run once `daylight --help` lists their flags |

Scenes (tri/quad only): `flat` 10x10 m, 1 m quads; `flat_tri`; `walls` N/E/S/W 10x3 m, 0.5 m cells, outward normals,
mesh order north, east, south, west; `box` = flat + two open 4x4x6 m boxes (no bottom face) at (0,6) and (-8,0);
`tilt30` 6x6 m plane tilted 30 deg to the south, 0.5 m cells; `overhang` 6x3 m south wall with a 1.5 m slab on top in
the same mesh (self-shading: the reference passes the mesh itself as context, the CLI runs without `--no-self-occlude`);
`canopy` = flat under a 6x6 m single sheet at 3 m (x -2..4, y -3..3, 1 m quads), porous rho 0.2 tau 0.5;
`glass_canopy` the same sheet as glass tau 0.64; `canopy_ground` / `glass_canopy_ground` the two sheets over a 40x40 m
opaque ground mesh, flat's grid 0.5 m up facing up and again facing down (the sheet passes what the ground reflects,
two reflections go back and forth); `canyon` two facing 20x10 m walls 10 m apart and the ground between them (context,
opaque), a sensor grid on each wall (1 m cells); `plane` flat's grid 1 m above a 2 km plane (context, opaque), facing
down. All but canopy are for the reflected-light layer.

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
| 8 transmit | rcontrib `-ab 1` with materials, gendaymtx `-O1` / `-O0` skies of `nodni.epw` | canopy, `--transmission on --patch-samples 64 --sky-density 1 --no-self-occlude` + its `--optics`, radiation and illuminance against the `-ad 32768` run: `\|sum ours / sum ref - 1\|`; per-sensor p95 of `\|ours / ref - 1\|` (budget below); 3 sigma of rcontrib's per-sensor noise (`-ad 8192` vs `32768`, noise ~ 1/sqrt(ad)) | 1 %; 0.5 %; 0.5 % | yes / yes / info |
| 9 reflect | closed form | `plane`: rcontrib `-ab 2`'s share of the plane's reflected light per MF 1 row vs the cosine law `rho pi (sin^2 a1 - sin^2 a0)`; ours `--bounces 1` vs rho x ours' unobstructed `flat` | 5 sigma; 5e-4 | yes |
| 9 reflect | rcontrib `-ab N+1` with materials, `nodni.epw` skies | totals of layer 8's flags + `--bounces N --bounce-samples 65536`, N = 1, 2 (canyon also N = 0: the physical direct pass) on canyon, both canopies, both over a ground, radiation and illuminance: `\|sum ours / sum ref - 1\|`; per-sensor p95 of `\|ours - ref\| / ref` | 2 % (N = 0: 1 %); budget (canopies 0.5 %, canyon 0.9 %) + 3 sigma | yes |
| 9 reflect | rcontrib `-ab 1`'s ground bin, `nodni.epw` | the glass canopy's mirror, ours N = 1 - ours N = 0: sum; per-sensor p95 of the difference / the sensor's reference total | 3 %; 0.1 % + 3 sigma | yes |
| 9 reflect | rcontrib `-ab N+1` - `-ab 1`, the `--epw` years, sun binned on both sides | reflected light ours N - ours 0, canyon, canopy and canopy_ground, N = 1, 2: sum; per-sensor p95 of the difference / the sensor's reference total | 3 %; 0.1 % + 3 sigma | yes |
| 9 reflect | both | 3 sigma of the per-sensor noise relative to the reference total: ours from a copy of its sensors in the same run, rcontrib from `-ad 8192` vs `32768` | - | info |

Layer 7 needs no EPW reference: the clear-sky weather is synthetic (Ineichen-Perez, Linke turbidity 3, sea level).

Layer 8 builds the fixture for Radiance from `fixture.json`: every context mesh as polygons with the material of its
`optics` (opaque `plastic rho rho rho 0 0`; glass `glass tn tn tn`, tn from Tn at n 1.52 as `src/glass.h`; porous
`trans c c c 0 0 tau/c 1` with c = rho + tau: diffuse reflectance rho, straight transmittance tau), the sky as
`void glow skyglow 0 0 4 1 1 1 0`, `skyglow source sky 0 0 4 0 0 1 180`, `skyglow source ground 0 0 4 0 0 -1 180`, and
the sensors (face centroids, normals) through

```
RAYPATH=_dl/radiance/lib rcontrib -I+ -ab AB -ad AD -lw 0.5/AD -e MF:1 -f reinhart.cal -bn Nrbins -b rbin -m skyglow -faf scene.oct
```

at AB 1, 2, 3 and AD 8192, 32768 (`-e MF:1` before `-f`: reinhart.cal needs MF while loading). A distant glow is never
a direct source, so `-ab 1` sees the sky through the canopy and no reflection, as the CLI without bounces. compare.py
multiplies the coefficients by the summed skies, channels weighted as the core's `bright()`: /1000 for kWh/m2, x 179 /
rows for mean lux. Measured on 100 sensors (15 % of the light passes the sheet; opaque or open it would miss by -15 /
+15 %): sums +0.04 % radiation, +0.17 % illuminance; per-sensor p95 0.10 % / 0.23 % against a reference sigma of
0.019 % (rcontrib seeds its sampling with the time in seconds and reference.py starts one run per second: a new
reference moves these by about that sigma). The CLI's resolution is above that noise, so the p95 limit is
a budget: physical mode takes the Perez sky at the NOAA apparent sun, gendaymtx at sun.c (+0.13 % open-sky
illuminance through the efficacy model, -0.08 to +0.03 % radiation per sensor from the sky's shape), tau 0.5 is stored
as 128/255 (+0.39 % of the light through the sheet, at most 0.16 % of a sensor's total), 64 samples per patch differ
from 256 by 0.06 % (p95), plus 3 sigma: 0.5 %. The p95 is the row that sees tau: tau 0.47 (6 % off) moves the sums by
only -0.92 % / -0.79 %, but the p95 to 2.2 %. Evaluated at the NOAA sun with 1024 samples per patch (a one-off
diagnostic port of the overcast Perez sky), the residual is 0.02 % with a spread of 0.015 %: rcontrib's noise.

Layer 9 runs layer 8's flags with `--bounces N --bounce-samples 65536` against `-ab N + 1` (the sensor's own hemisphere
takes the first ambient bounce, so `-ab N + 1` holds N diffuse reflections). With rcontrib's defaults (`-aa 0 -as 0
-lr -10`) `-lw` decides how a reflection is sampled: at layer 8's first `-lw 1e-5*8192/AD` rad5R4 samples every
reflection with 2 x 2 cells, skewed away from the surface normal. A plane's reflected light (the `plane` fixture) came
out -15.6 % in the zenith cap and +4.8 % in the horizon row against the cosine law (a sensor hemisphere of 2 x 2 or
4 x 4 cells, `-ad 4` / `16`, does the same; 1 x 1 and 181 x 181 do not), and the canyon walls' reflected light 1 % off
(bin by bin, ours +20 % in the zenith cap, -2 % in rows 1-3). `-lw 0.5/AD` caps each reflection at one sample (rcontrib
cuts a hemisphere's samples to keep their weight above `-lw`), leaves the sensor's `-ad` alone and runs 2.6x faster;
the plane's rows then sit within 2.0 sigma of the cosine law, the layer's first row. Russian roulette thins the second
reflection (survival ~0.25 at rho 0.2): unbiased, noisier.

Sigma per sensor, relative to its reference total: rcontrib's from `-ad 8192` vs `32768` (`var(lo - hi) = 5 var(hi)`),
ours' from a copy of the sensors listed again in the same run (`var(a - b) = 2 var(a)`): paths are seeded by the global
sensor index, so the copy is an independent estimate at the same M. A run at M / 4 shares its per-sample random
numbers with the run at M and understates sigma by 8-15 %; 4 M exceeds the CLI's 65536. The per-sensor p95 limits are
a budget + 3 sigma, hard:

- totals on nodni: the direct pass carries most of each total. The canopies take layer 8's 0.5 %. The canyon's 0.9 %:
  its walls see the sky at the NOAA apparent sun up to 0.41 % off gendaymtx's (bare walls on nodni against rcontrib on
  the bare sky: north +0.10, east -0.16, south -0.41, west -0.23 % radiation; weighted with the core's own sky, a
  one-off through sky.cpp, the canyon's wall means move from +0.10 / -0.40 % to -0.02 / -0.03 %), and 64 samples per
  patch resolve the patches the walls cut to p95 0.54 % (0.15 % at 4096, rcontrib's noise).
- reflected light on the `--epw` years: 0.1 %. The bounce pass bins the sun as gendaymtx -d does but at the NOAA
  apparent position: up to 0.9 % of a canyon wall's reflected light, 0.06 % of its total (with the core's own sky the
  walls agree within +0.03 to +0.23 %). The glass canopy's reflected light is its mirror (it has no diffuse surface),
  which rcontrib `-ab 1` puts in the ground bin (the sensors face up: nothing else lands there), checked on nodni.

Measured at ebaaa86: plane rows 2.0 sigma, ours on the plane 1.6e-5 of rho x flat. Totals on nodni (sums; p95 against
the limit): canyon N = 0 -0.18 % / -0.03 % (radiation / illuminance), 0.74 % / 0.70 % against 1.13 %; N = 1, 2 -0.15
to -0.14 % / -0.01 to 0.00 %, 0.69-0.70 % / 0.63-0.64 % against 1.31-1.35 %; canopy +0.03-0.04 % / +0.16-0.17 %,
0.09-0.10 % / 0.22-0.23 % against 0.57 %; glass canopy -0.01 % / +0.12 %, 0.08-0.09 % / 0.17-0.18 % against 0.55 %;
over a ground (measured at the review fix after ebaaa86) porous +0.03 % / +0.16 %, 0.37-0.40 % against 0.99-1.02 %,
glass -0.02 to -0.01 % / +0.11 to +0.12 %, 0.32-0.36 % against 0.89-0.91 %. The
glass mirror (0.42 % of the total) +0.01 % / +0.14 %, p95 0.020 % of the total against 0.13 %. Reflected light, Boston
and Chicago: canyon (11 % of the total, 25 % on the north-facing wall) sums +0.40 to +0.61 %, p95 0.51-0.65 % of the
total against 0.84-0.97 % (3 sigma 0.74-0.86 %: rcontrib 0.21-0.24 %, ours 0.12-0.17 % per sensor); canopy (1.2 %: the
sheet's underside) -0.43 to +0.02 %, p95 0.11-0.14 % against 0.29-0.34 %; canopy over the ground (17 % of the total,
all of a face looking down) +0.33 to +0.39 % / +0.42 to +0.47 %, p95 0.98-1.20 % against 1.20-1.36 % (mostly 3 sigma:
the down-facing faces see nothing else). The default EPWs' totals are not compared:
ours' direct pass takes the exact sun, rcontrib the binned one.

Relative errors use `max(|lb|, 1e-3 max|lb|)` as the scale. W per direction is reconstructed from the ladybug
SkyMatrix data (`dir + dif` per sky patch, `rho * mean` for every ground direction) over the ladybug direction order
`[tregenza/reinhart dome vectors | the same reversed]`, rotated by `north`.

The CLI dump formats compare.py reads (`--out` + `.json` manifest, `--dump-bits`, `--dump-sky`, `--dump-sun`) are
listed at the top of `compare.py`; they follow what `src/run.cpp` / `src/main.cpp` write.
