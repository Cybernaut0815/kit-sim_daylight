# daylight-mesh (TypeScript binding)

Turns three.js scenes or `.glb` files into the raw `.dmesh` arrays (or OBJ) the daylight core CLI reads, and (Node only) runs the core.

Node (three.js scene -> values):

```ts
import { fromThree, run } from 'daylight-mesh';

const sensors = fromThree([scene]);                 // every descendant Mesh, world metres, Y-up -> Z-up
const { values, columns } = await run('daylight.exe', { sensors, args: ['--epw', 'x.epw'] });
console.log(columns, values[0]);                    // one Float64Array per sensor element
```

glb:

```ts
import { fromGLB, run } from 'daylight-mesh';
import fs from 'node:fs/promises';

const sensors = await fromGLB(await fs.readFile('a.glb'));            // { scale } multiplies metres
const context = await fromGLB(await fs.readFile('city.glb'));
const { values } = await run('daylight.exe', { sensors, context, args: ['--epw', 'x.epw', '--quantity', 'radiation'] });
```

Browser (or Node) against a `daylight_server` (the GPU box runs the core in-process; meshes are uploaded once and
referenced by `id` afterwards, so a re-run with other flags sends no geometry):

```ts
import { fromThree, runRemote } from 'daylight-mesh';

const sensors = fromThree([scene]);                                   // give each ObjMesh a stable `id` to reuse the server cache
const { values, columns, cache } = await runRemote('http://gpu-box:8080', { sensors, epw: epwFile, args: ['--quantity', 'radiation,sunhours'] });
```

Colour a three.js mesh by its results (`applyToThree`):

```ts
import { applyToThree, fromThree, run } from 'daylight-mesh';

const { values, columns, manifest } = await run('daylight.exe', { sensors: fromThree([mesh]), args: ['--epw', 'x.epw'] });
const { first, count } = (manifest as { meshes: { name: string; first: number; count: number }[] }).meshes[0]; // rows of this mesh
const col = columns.indexOf('radiation_front'); // columns are <quantity>_<side>
await applyToThree(mesh, values.slice(first, first + count).map((r) => r[col])); // one value per triangle -> flat face colours
```

## API

- `colormap(values, { vmin, vmax, stops })`: viridis (9 stops, linear) -> `Float32Array` RGB triples; range defaults to the
  finite min/max of `values` (`vmax == vmin` -> first stop); non-finite values -> grey `[0.5, 0.5, 0.5]`; `stops` =
  any evenly spaced RGB list for another gradient. Default identical to `daylight.colormap` in the Python binding.
- `applyToThree(mesh, values, { vmin, vmax, stops, rgb, material })`: `values.length` must be the triangle count (per-face: the
  geometry is converted to non-indexed via `toNonIndexed()` so each face gets one flat colour) or the vertex count
  (per-vertex, geometry unchanged). Colours from `colormap()` or from `rgb` (3 floats per value). Sets the `color`
  attribute and, unless `material: false`, `mesh.material = new MeshStandardMaterial({ vertexColors: true })`.
  Imports `three` lazily; returns the RGB array.
- `ObjMesh { name, positions /* xyz, metres, Z-up */, indices /* 0-based triangles */, slots?, id? }`. `slots`: optional
  `Uint16Array`, one optics slot per triangle (the core's `Mesh::slot`; absent = every triangle slot 0), written by
  `toDmesh` (so by `run` and `runRemote`: part of the bytes the server caches under `id`); `toObj` drops it.
- `fromThree(objects, { scale })`: `updateMatrixWorld(true)`, applies `matrixWorld`, converts `(x, y, z) -> (x, -z, y)`,
  reverses winding when `det(matrixWorld) < 0`. Triangle `BufferGeometry` only (indexed or not); no strips, fans or lines.
- `fromGLB(data, { scale })`: via `@gltf-transform/core`; `TRIANGLES` primitives of every node with a mesh,
  name `node/mesh`. Same frame conversion and winding rule.
- `toDmesh(mesh)`: one mesh -> `Uint8Array` in the `.dmesh` layout below (flags 1 + the slot block when `mesh.slots`
  is set, else flags 0); throws on bad indices or a `slots` length other than the triangle count.
- `toObj(meshes)`: one `o` group per mesh, 1-based global indices; throws on bad indices (interop with other OBJ tools).
- `run(exe, { sensors, context?, args })`: temp dir, `exe <sensors...> [--context <file>]... ...args --out out.npy`,
  returns `{ values: Float64Array[] (one per element), columns, manifest }`; rejects with the core's stderr.
  `sensors`/`context` accept `ObjMesh[]` (written as one `<name>.dmesh` per mesh, name sanitised to `[A-Za-z0-9._-]`
  and made unique; the manifest's mesh names are the file stems) or OBJ text (written as one `.obj`).
  Needs Node >= 20 and the GPU build of the core.
- `runRemote(url, { sensors, context?, epw?, args?, signal?, clientId?, onStatus? })`: submits a job
  (`POST url/jobs`) with the meshes' `id`s (minted as UUIDs when absent), `PUT url/mesh/<id>?name=<stem>` once for
  every id the server reports missing, then one retry; long-polls `GET url/jobs/<id>?wait=25` (`onStatus` sees each
  queued / running state, with the queue position); aborting `signal` cancels the job on the server; `clientId` is
  sent as `X-Client-Id` for the per-client job limit; `epw` is a `Blob` (uploaded) or `{ url }` (fetched by the server
  through its allowlisted proxy cache); returns `{ values, columns, manifest, cache }` (`values_b64` decoded).
  Browser-safe. The full API is self-documented by the server at `/docs` and `/openapi.json`.
- `dmeshStems(names, fallback?)`: the stem / mesh-name sanitiser both runners use (`[^A-Za-z0-9._-]` -> `_`, unique
  case-insensitively).
- `parseNpy(bytes, cols)`: parses a core `--out FILE.npy` (NumPy v1.0 `<f8`, C order) into rows of `cols`.
- `parseValues(text)`: parses a core `--out FILE` text file.

Browser bundles use `toDmesh`, `toObj`, `fromThree`, `fromGLB`, `colormap`, `applyToThree`, `runRemote`; `run` imports Node builtins lazily and must stay on Node.

## Performance (2M triangles, RTX 2060)

1000x1000 grid as one glb (1 002 001 Float32 positions, 2M Uint32-indexed triangles), best of 2:

| step | ms |
| --- | --- |
| `fromGLB(bytes)` | 43 |
| `toDmesh(mesh)` | 32 |
| `fs.writeFileSync(.dmesh)` (56 MB) | 181 |
| `run()` end to end (write, core sunhours one timestep, read `.npy`) | 1724 |

`run()` reads the values back from `--out out.npy` (one `Float64Array` over the raw bytes) instead of text:
`parseValues` on a 1M-row text file takes ~0.9 s, the `.npy` read is I/O bound. `toObj` (text, ~1.5 s) stays interop only.

## The `.dmesh` format

One mesh per file, little-endian, no library semantics. Mesh name = file stem.

```text
offset  0  char[4]   "DLM1"                 magic + version
offset  4  uint32    flags                  bit 0: slot block present; other bits must be 0 (keeps the arrays 8-byte aligned)
offset  8  uint64    n_points
offset 16  uint64    n_faces
offset 24  uint64    n_indices
offset 32  float64   xyz[3 * n_points]      world metres, Z-up
           int32     counts[n_faces]        vertices per face (>= 3), right-handed winding; toDmesh writes 3
           int32     indices[n_indices]     0-based, sum(counts) == n_indices
           uint16    slot[n_faces]          only with flags bit 0: optics slot of each face (ObjMesh.slots)
```

The core rejects other flag bits and a short slot block ([docs/input.md](../../docs/input.md)).

```sh
npm install && npm run build && npm test
DAYLIGHT_EXE=../../build/RelWithDebInfo/daylight.exe npm test    # also runs the box through the core via .dmesh (12 rows)
```
