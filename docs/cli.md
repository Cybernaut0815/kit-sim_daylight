# CLI reference

Complete CLI reference (`daylight --help` prints the short form). The C++ `daylight::Options` field for each flag
is given for in-process callers; the Python and TypeScript `run(..., args=[...])` pass flags through verbatim.

| flag | value | default | meaning | `Options` |
|---|---|---|---|---|
| `FILE...` | `.dmesh` `.obj` `.usd*` | required (or `--bench`) | sensor meshes: analysed, and blocking unless `--no-self-occlude` | `Mesh::sensor = true` |
| `--context FILE` | mesh file, repeatable | none | blocks rays only, never analysed | `Mesh::sensor = false` |
| `--on` | `faces` \| `vertices` | `faces` | element = polygon (sensor at the centroid, face normal) or point (area-weighted vertex normal) | `onVertices` |
| `--mode` | `physical` \| `ladybug` | `physical` | sky/sun model, see [modes.md](modes.md) | `mode` |
| `--quantity` | `Q[,Q...]` | `radiation` | one output column per quantity: `radiation` kWh/m2 over the run, `direct` (sun only: beam x transmittance), `diffuse` (sky + ground, x transmittance; with `--bounces` also the light the model reflects, its reflected sun included), `irradiance` mean W/m2 (instantaneous for a single `--time`), `illuminance` mean lux (physical mode), `sunhours` equivalent full-sun hours (an hour behind a surface transmitting 0.6 counts 0.6; with every surface opaque, the default, an exact count of sunlit hours). Transmittance is 1 for open sky and 0 behind an opaque surface; between only with `--transmission on`. Reflected light (`--bounces`) adds to radiation, diffuse, irradiance and illuminance, never to direct or sun hours | `quantities` |
| `--sides` | `front` \| `back` \| `both` | `front` | receiving hemisphere of each element; `both` = two columns per quantity (`_front`, `_back`) | `sides` |
| `--cull-backfaces` | flag | off | occluder triangles block only from their front (winding) side; default both sides. With `--transmission on` a closed translucent solid is then crossed once, through its front faces (a crown counts its tau once, not twice) | `cullBackfaces` |
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
| `--optics SPEC[;SPEC...]` | `NAME[/SLOT]=RHO[,TAU[,glass\|porous]]`, repeatable | every slot `rho 0.2`, `tau 0` | numbers per mesh or face slot (`.dmesh` slot block, see [input.md](input.md)): RHO diffuse reflectance, TAU transmittance at normal incidence, `rho + tau <= 1`. Kind `porous` when TAU > 0 (tau at every angle: screens, trees) unless `glass` (Radiance glass, n 1.52: RHO 0, 0 < TAU <= 0.918; tau 0.64 = double low-e). NAME is the core mesh name (`.dmesh` stem, OBJ `o`/`g` group, USD prim path; for TypeScript callers the `dmeshStems()` output); the SPEC splits at its last `=`, `/SLOT` only as a trailing `/` + digits (`Facade/East` is a name, `/World/Roof/2` is slot 2 of `/World/Roof`); `*` = every mesh; `;` packs several SPECs into one token (a name containing `;` cannot be addressed). Precedence NAME/SLOT > NAME > `*` > built-in. Values out of range: exit 1 at parse; an unknown or ambiguous name, a slot out of range or a row given twice: exit 1 after loading (library: `run()` fails, server: a job error). No effect by itself: TAU needs `--transmission on`, RHO needs `--bounces` (without it every surface is black) | `optics` (`daylight::Optics`) |
| `--transmission` | `off` \| `on` | `off` | on (physical mode): light passes glass / porous slots, weighted by their transmittance at each crossing (a closed thin solid, e.g. a pane modelled as a 1 cm box, counts once). Free without a translucent slot (TAU > 0 on faces that block: a sensor mesh's slots under `--no-self-occlude` never count); with one, rays continue through it (one any-hit per crossing) and visibility takes 8 bit per direction instead of 1 (8x memory, chunked). Off: every surface opaque, as before | `transmission` |
| `--fresnel` | `on` \| `off` | `on` | glass transmittance by angle (Radiance `glass`: both faces, internal reflections, the longer path in the glass; 0.64 at normal incidence passes 0.54 at 60 deg, 0.12 at 85 deg); off: glass passes tau at every angle | `fresnel` |
| `--bounces N` | `0`..`20` | `0` | reflected light (physical mode; ladybug mode refuses N > 0): what reaches a sensor after up to N diffuse reflections off the model, each by its slot's RHO. Glass mirrors its Radiance R(theta) (`--fresnel on`; not counted in N), translucent slots pass light with `--transmission on` (otherwise every slot is opaque: glass, RHO 0, absorbs), a thin solid's exit face passes it. Paths leave each sensor cosine-weighted and are binned where they escape: a sky patch of `--sky-density` (with the sun binned into the patches as `gendaymtx` does) or the virtual ground (`--ground`), as `rcontrib -ab N+1` daylight coefficients. 1 is the common outdoor setting (Honeybee's exterior studies use `-ab 2`); Russian roulette from the second reflection on keeps higher N cheap outdoors; interiors need 5-8. A ground mesh hides the virtual ground from the rays it stops (without bounces those see a black surface; with them its RHO reflects the sky and sun it receives, shaded by the model). 0: no reflection pass, no glass mirror, today's numbers | `bounces` |
| `--bounce-samples M` | `64`..`65536` | `1024` | paths per sensor for `--bounces`, stratified (Hammersley points, rotated per sensor) and deterministic: a run repeats its numbers, chunking and launch sizes change nothing. Noise falls as 1 / sqrt(M); annual and long-range values average it out, a single `--time` instant does not (the sun sits in a few patches): raise M (4096-16384) for instants and for small differences between variants | `bounceSamples` |
| `--north DEG` | degrees, counter-clockwise from +Y | `0` | model north (ladybug convention; 90 = west) | `north` |
| `--offset M` | metres | `0` | sensors moved along their normal (Grasshopper uses 0.1) | `offset` |
| `--out FILE[.npy]` | path | stdout | values: text (one line per element, `%.10g` columns) or NumPy float64 `[N x C]`; manifest at `FILE.json` | `Result` |
| `--timeseries F.npy` | path | off | per-step values, float32 `[N x T]` (W/m2, or lux for illuminance) for the non-night steps; `F.npy.hoys.txt` lists the hours of year of the evaluated instants, in the `--tz` clock | `seriesPath` |
| `--dump-bits F.npy` | path | off | validation: visibility bits uint32 `[N x words]`, direction order sky \| ground \| sun; with a translucent slot (`--transmission on`) the transmittance uint8 `[N x directions]` instead (x255, 0 = culled or blocked) | `dumpBitsPath` |
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
