# daylight (Python binding)

USD stage -> raw mesh arrays for the agnostic `daylight` core, a `.dmesh` writer for numpy arrays, and a thin
`subprocess` wrapper around the CLI. uv project (Python >= 3.10, `usd-core==26.8`, `numpy`); no C++ build is
needed for the conversion itself. Everything stays binary: `meshes()` is vectorised numpy, meshes go to the core as
`.dmesh`, values come back as `.npy` (1M quads: `meshes()` 0.1 s + `to_dmesh` 0.15 s after `Stage.Open`).

## Usage

Raw arrays -> values (one quad sensor; `values` is `[N x C]`, N = faces by default, C = `manifest['columns']`):

```python
import numpy as np, daylight
P = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)        # world metres, Z-up
values, manifest = daylight.run_meshes([(P, [4], [0, 1, 2, 3])], epw='site.epw')
print(manifest['columns'], manifest['meshes'], values)
```

USD stage -> `meshes()` / `split()` / `run_meshes()` (sensors analysed and blocking, the rest blocking only):

```python
import daylight
sens, ctx = daylight.split(daylight.meshes('scene.usd'), sensors=['/World/Bldg'])
values, m = daylight.run_meshes([x[1:] for x in sens], [x[1:] for x in ctx], epw='site.epw',
                                args=['--quantity', 'radiation'])
print([x[0] for x in sens], m['meshes'], values.shape)                  # prim paths in row order
```

Results back on the stage (`write_results`): a new layer that sublayers `scene.usd` and overrides each sensor prim
with float primvars `daylight_<column>` + `primvars:displayColor` (viridis via `colormap`), so opening the new file
alone in usdview / Kit shows the coloured scene and `scene.usd` stays untouched:

```python
import daylight
sens, ctx = daylight.split(daylight.meshes('scene.usd'), sensors=['/World/Bldg'])
values, m = daylight.run_meshes([x[1:] for x in sens], [x[1:] for x in ctx], epw='site.epw',
                                args=['--quantity', 'radiation'])
daylight.write_results('scene.usd', values, m, [x[0] for x in sens], 'daylight.usda')   # -> usdview daylight.usda
```

One-liner (writes `out/<stem>.dmesh` per mesh + `out/meshes.json`; flags after `--` go to the core ->
`out/values.npy(.json)`, rows in `meshes.json` order; `--write-usd PATH` adds the sublayer file):

```
cd bindings/python && uv sync
uv run python -m daylight scene.usd --sensors /World/Bldg -o out --write-usd out/daylight.usda -- --epw x.epw --quantity radiation
uv run python -m daylight.check          # to_dmesh / masking / converter / colormap / write_results self-checks + run_meshes end to end (needs the exe)
```

`--sensors`/`--context` are prim-path prefixes (repeatable): no `--sensors` = every mesh is a sensor, no `--context`
= every non-sensor mesh blocks only, both = the rest is dropped. `--time T` picks the USD time code (default: the
stage start time). File stem = prim path with `[^A-Za-z0-9._-]` replaced by `_`, leading `_` stripped, made unique
with `_2`, `_3` ... (`/World/Bldg/roof` -> `World_Bldg_roof.dmesh`); `meshes.json` is the ordered list
`[{"file", "prim", "role": "sensor" | "context"}]`, sensors first.

## API

- `meshes(stage, time=None)` -> `[(name, points [N,3] float64, counts int32 [F], indices int32 [sum(counts)])]`
  (the `.dmesh` arrays, 0-based flat indices); `split(meshes, sensors, context)`; `to_obj(meshes, path)` (interop).
- `to_dmesh(points, counts, indices, path)`: one mesh as raw arrays -> `.dmesh` (layout below). `ValueError` unless
  `sum(counts) == len(indices)`, every count >= 3 and every index in `[0, n_points)`.
- `run(sensors, context=(), epw=None, args=(), out=None)` -> `(values [N x C], manifest)`; `sensors`/`context` are
  a mesh file (`.obj` / `.dmesh`) or an iterable of them, passed positional / as repeated `--context`. `out` keeps
  the value file: `.npy` (default, `np.load`) or any other suffix for text (`np.loadtxt`).
- `run_meshes(sensors, context=(), **run kwargs)`: each argument an iterable of `(points, counts, indices)` triples
  (a single triple is accepted too); written as `sensor_<i>.dmesh` / `context_<i>.dmesh` (= the manifest names).
- `write_results(stage, values, manifest, prims, out, on='faces', name='daylight', vmin=None, vmax=None)` -> written
  prim paths. `stage` = path or `Usd.Stage`; `values` `[N x C]` (1-D = one column) with rows per `manifest['meshes']`,
  `prims` the sensor prim paths in that order; `out` (.usda/.usdc) is created / overwritten with `subLayerPaths =
  [relative path to the stage's root layer]` (absolute when on another drive). Per column one primvar `<name>_<column>` (`float[]`, `uniform` for
  faces / `vertex` for points) plus `displayColor` from `colormap(first column, vmin, vmax)` (default range: finite
  min/max over all written sensors). Faces `meshes()` dropped (holes, < 3 vertices) get `nan` and grey; any other
  count mismatch raises `ValueError` naming the prim; instance proxies and `<instancer>[k]...` names are skipped with a
  warning (no prim to override). Uses the stage start time.
- `colormap(values, vmin=None, vmax=None)` -> float64 `[N, 3]` RGB in 0..1: viridis with 9 stops, linear in
  between, `t = clip((v - vmin) / (vmax - vmin), 0, 1)` (`vmax == vmin` -> first stop), non-finite -> 0.5 grey. The
  same table as the TypeScript binding, so colours match across tools.

## `.dmesh` layout (one mesh per file, little-endian, name = file stem)

```
offset  0  char[4]   "DLM1"                 magic + version
offset  4  uint32    flags                  0 here (this binding writes no slot block); bit 0 = uint16 slot[n_faces] after the indices, see docs/input.md
offset  8  uint64    n_points
offset 16  uint64    n_faces
offset 24  uint64    n_indices
offset 32  float64   xyz[3 * n_points]      world metres, Z-up
           int32     counts[n_faces]        vertices per face (>= 3), right-handed winding
           int32     indices[n_indices]     0-based, sum(counts) == n_indices
```

Conversion: composed stage, instance proxies and PointInstancers expanded (names `<instancer>[k]<proto mesh path>`),
invisible / `guide` / `proxy` / inactive prims dropped, `metersPerUnit` and `upAxis` applied (Y-up -> Z-up as
`(x, -z, y)`), winding flipped for `leftHanded` meshes and mirroring transforms, holes and faces with < 3 vertices
dropped, authored normals ignored, subdivision control cages used as-is (warning on stderr). `ValueError` names the
prim when `faceVertexCounts` / `faceVertexIndices` are inconsistent or out of range.

OBJ contract of the core (`to_obj`, interop only): world metres, Z-up, right-handed (counter-clockwise = front)
winding, one `o` group per mesh, 1-based global vertex indices.

Core executable: `DAYLIGHT_EXE` if set, else `<repo>/build/RelWithDebInfo/daylight.exe`.
