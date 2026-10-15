# CLI reference

Complete CLI reference (`daylight --help` prints the short form). The C++ `daylight::Options` field for each flag
is given for in-process callers; the Python and TypeScript `run(..., args=[...])` pass flags through verbatim.

| flag | value | default | meaning | `Options` |
|---|---|---|---|---|
| `FILE...` | `.dmesh` `.obj` `.usd*` | required (or `--bench`) | sensor meshes: analysed, and blocking unless `--no-self-occlude` | `Mesh::sensor = true` |
| `--context FILE` | mesh file, repeatable | none | blocks rays only, never analysed | `Mesh::sensor = false` |
| `--on` | `faces` \| `vertices` | `faces` | element = polygon (sensor at the centroid, face normal) or point (area-weighted vertex normal) | `onVertices` |
| `--mode` | `physical` \| `ladybug` | `physical` | sky/sun model, see [modes.md](modes.md) | `mode` |
| `--quantity` | `Q[,Q...]` | `radiation` | one output column per quantity: `radiation` kWh/m2 over the run, `direct` (sun only), `diffuse` (sky + ground), `irradiance` mean W/m2 (instantaneous for a single `--time`), `illuminance` mean lux (physical mode), `sunhours` hours of unoccluded sun | `quantities` |
| `--sides` | `front` \| `back` \| `both` | `front` | receiving hemisphere of each element; `both` = two columns per quantity (`_front`, `_back`) | `sides` |
| `--cull-backfaces` | flag | off | occluder triangles block only from their front (winding) side; default both sides | `cullBackfaces` |
| `--no-self-occlude` | flag | off | sensor meshes do not block rays (only `--context` files do) | `selfOcclude = false` |
| `--epw FILE` | EPW path | none | hourly weather (DNI/DHI) and location from the header; 8760 or 8784 rows, hour-ending in the file's own zone, sun evaluated at mid-hour. PVGIS files (recognised by their `Irradiance Time Offset` comment) built from its satellite series are written about 0.7 h ahead of their hour: the irradiance is moved back by the offset measured from the daylight edges (`-v` prints it); its ERA5-based files (declared offset -0.5) are aligned and left alone. With `--lat/--lon` elsewhere the file keeps its own solar day: each row meets the site's sun at the same solar time | `read_epw` -> `Weather` |
| `--clear-sky` | flag | off | no weather file: synthesise an hourly clear-sky year (Ineichen-Perez model, pvlib formulation) for `--lat/--lon`; cloudless upper bound, typically 20-50 % above a TMY; exclusive with `--epw` | `clear_sky_weather(lat, lon, tz, TL, alt, w)` |
| `--turbidity TL` | Linke turbidity | `3` | atmosphere for `--clear-sky`: 2 very clear mountain, 3 clear, 4-5 hazy / urban, 6-7 polluted | argument of `clear_sky_weather` |
| `--altitude M` | metres | `0` | site elevation for `--clear-sky` (air pressure, model coefficients) | argument of `clear_sky_weather` |
| `--lat DEG` `--lon DEG` | degrees, N and E positive | EPW header | location; required without `--epw` (sun hours, clear sky); overrides the EPW header | `lat`, `lon` |
| `--tz H` | hours, E positive | EPW header, else `round(lon / 15)` | the clock of `--time`, `--from`, `--to` and `--hours`: standard time, no DST. An EPW keeps its own clock; the timestamps are moved into it, so the weather of an hour stays with the sun of that hour whatever `--tz` says | `tz` |
| `--time TS` | `[YYYY-]MM-DDTHH:MM` | | evaluate one instant (year ignored; Feb 29 only with a leap EPW); DNI/DHI interpolated between the hours around it | `t0` |
| `--from TS --to TS` | timestamps | | evaluate the range `[from, to)` in `--step` increments, the sun at the middle of each step (the last step of the range, or of a day's `--hours` window, may be shorter) | `t0`, `t1` |
| `--step MIN` | minutes | `60` | step of `--from/--to` (DNI/DHI interpolated between the hour centres; 60-min steps on whole hours read the hours exactly) | `stepMin` |
| `--hours HH:MM-HH:MM` | times of day, end up to `24:00` | whole day | with `--from/--to`: keep only the steps in `[start, end)` of every day, e.g. 1 Jun to 31 Aug 09:00 to 17:00 (an analysis period) | `dayFrom`, `dayTo` (minutes) |
| (no time flags) | | | annual: one step per weather row at the mid-hour | `t0 = t1 = NAN` |
| `--sky-density` | `1` \| `2` \| `4` | `1` | sky patches 145 / 577 / 2305 (Reinhart MF); ladybug mode: 1 or 2 | `mf` |
| `--patch-samples K` | square number | `1` | physical mode: K stratified rays per patch (4, 9, 16 smooth the patch discretisation) | `patchSamples` |
| `--ground RHO` | 0..1 | `0.2` | ground reflectance (Lambertian ground = mirrored sky) | `ground` |
| `--north DEG` | degrees, counter-clockwise from +Y | `0` | model north (ladybug convention; 90 = west) | `north` |
| `--offset M` | metres | `0` | sensors moved along their normal (Grasshopper uses 0.1) | `offset` |
| `--out FILE[.npy]` | path | stdout | values: text (one line per element, `%.10g` columns) or NumPy float64 `[N x C]`; manifest at `FILE.json` | `Result` |
| `--timeseries F.npy` | path | off | per-step values, float32 `[N x T]` (W/m2, or lux for illuminance) for the non-night steps; `F.npy.hoys.txt` lists the hours of year of the evaluated instants, in the `--tz` clock | `seriesPath` |
| `--dump-bits F.npy` | path | off | validation: visibility bits uint32 `[N x words]`, direction order sky \| ground \| sun | `dumpBitsPath` |
| `--dump-sky F` | path | off | validation (ladybug mode): the two `gendaymtx -A` blocks | `dumpSky` |
| `--dump-sun F` | path | off | validation: per step hoy, date, NOAA and sun.c positions, DNI, DHI | `dumpSun` |
| `--bench [NSENS NTRIS]` | counts | `1000000 10000000` | synthetic grid + boxes benchmark, prints timings, no output file | |
| `-v` | flag | off | timings and sizes on stderr (`info: read ...`, `info: run ...`, `info: output ...`) | `verbose` |
| `-h` `--help` | | | usage | |

Environment: `CUDA_VISIBLE_DEVICES` selects the GPU. Exit codes: 0 ok, 1 usage error (message + usage on stderr),
2 runtime error (file, weather, scene, CUDA/OptiX). Weather is needed for every quantity except `sunhours`; without
`--epw` or `--clear-sky` only `--quantity sunhours` runs (`--lat --lon` required, `--tz` optional).

## Output

- Rows: one per element of every sensor mesh, meshes in input order, elements in mesh order (`--on faces`: one row
  per input polygon in `counts` order, a quad is one row; `--on vertices`: one row per point in point order). Nothing
  is reordered or dropped, so `values[first:first+count]` from the manifest's `meshes` entry is exactly that mesh's
  face or point attribute.
- Columns: `columns` in the manifest, `<quantity>_<side>` (`radiation_front`, `sunhours_back`, ...), in `--quantity`
  order, sides inner.
- Manifest `FILE.json`: `{"columns": [...], "duration_h": H, "dirs": {"sky": P*K, "ground": P*K, "sun": S},
  "meshes": [{"name": "...", "first": 0, "count": N0}, ...]}`. Mesh names: `.dmesh` file stem, OBJ `o`/`g` group, USD
  prim path (C++ reader) or sanitised prim path (Python converter, see its `meshes.json`).
- Units: `radiation`/`direct`/`diffuse` kWh/m2 summed over the evaluated steps (x1000 = Wh/m2), `irradiance` W/m2
  averaged over the evaluated hours, `illuminance` lux averaged, `sunhours` hours.
- Library: `Result::values` row-major `[N x C]`, `Result::columns`, `Result::ranges` = `(name, count)` per sensor mesh
  in input order.
