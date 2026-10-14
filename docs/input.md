# Input and semantics

## Formats

| input | role | availability |
|---|---|---|
| `.dmesh` | native: raw arrays, one mesh per file, name = file stem (layout below) | always; the bindings write it |
| `.obj` | interop text: world metres, Z-up, right-handed winding, one `o`/`g` group per mesh, 1-based global `v` indices (negative = relative); `vt`/`vn` ignored | always |
| `.usd` `.usda` `.usdc` `.usdz` | interop via LightUSD (`read_usd`), every mesh of the stage, name = prim path | `-DDAYLIGHT_USD=ON`; the Python binding converts USD without it |

```
offset  0  char[4]   "DLM1"                 magic + version
offset  4  uint32    0                      reserved (keeps the arrays 8-byte aligned)
offset  8  uint64    n_points
offset 16  uint64    n_faces
offset 24  uint64    n_indices
offset 32  float64   xyz[3 * n_points]      world metres, Z-up
           int32     counts[n_faces]        vertices per face (>= 3), right-handed winding
           int32     indices[n_indices]     0-based, sum(counts) == n_indices
```

Positional files are analysed (sensors) and, by default, also block rays; `--context FILE` files block only (roles
are per file). Location and time: `--epw FILE` gives weather and location, `--clear-sky` synthesises weather for
`--lat --lon`, `--lat --lon --tz` alone serve sun-hour runs; `--time`, `--from/--to/--step` (plus `--hours` for the
same hours of every day) select instants, the default is annual.

## Semantics

- **Separate arrays per mesh in, one scene inside.** Every mesh (one `.dmesh` / `daylight::Mesh`, one OBJ group,
  one USD prim) keeps its own points and indices. Internally all occluding triangles go into one OptiX acceleration
  structure and all sensors into one array; results come back in input order and the manifest `meshes` ranges
  (`Result::ranges`) map rows back to meshes.
- **Sensor meshes block by default.** A sensor mesh is also an occluder (self-shading, like the analysed geometry in
  a Radiance scene); `--no-self-occlude` leaves sensor meshes out of the acceleration structure. An element never
  blocks its own rays: the ray origin is nudged along the normal (Waechter-Binder) and hits closer than 1e-5 m are
  ignored.
- **Hemisphere cull.** Per sensor, directions with a non-positive cosine against its normal are never traced and
  contribute nothing. No sensor is dropped; every element gets a row, even one looking into a wall (then 0).
- **`--cull-backfaces`.** Occluder triangles whose front side faces away from the ray are ignored, decided per ray
  inside the RT-core traversal (OptiX ray flag); nothing is pre-filtered on the host. Default: triangles block from
  both sides, as Radiance black plastic.
- **`--on faces` vs `--on vertices`.** Faces: one element per polygon (quads and n-gons are one element each),
  sensor at the centroid, direction the face normal. Vertices: one element per point, direction the area-weighted
  mean of the adjacent face normals (a point shared by faces of several meshes belongs to each mesh's output).
- **`--sides both`** evaluates the front and the back hemisphere of each element (two columns per quantity).
- **C++ callers need no files**: build `daylight::Mesh` objects and call `daylight::run`; `.dmesh` exists so other
  languages and the CLI pass the same raw arrays without parsing.
