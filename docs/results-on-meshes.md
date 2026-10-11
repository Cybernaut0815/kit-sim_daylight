# Writing results back onto meshes

The values are plain per-face or per-point attributes, so writing them back is index arithmetic. The bindings do it:

- **USD (Python)**: `daylight.write_results(stage, values, manifest, prims, out, on='faces', name='daylight',
  vmin=None, vmax=None)` writes an `over` layer `out` (`.usda`/`.usdc`) with one `float[] primvars:<name>_<column>`
  per column (`interpolation = "uniform"` for faces, `"vertex"` for points) and `primvars:displayColor` (viridis of
  the first column, `vmin`/`vmax` default to the data range) on every sensor prim. The source stage is untouched; add
  the layer as a sublayer (Omniverse Kit, usdview, `Usd.Stage.Open('daylight.usda')`) and the meshes are coloured.
  `prims` = the prim paths of the sensor meshes in run order (what `split` gave you); the CLI one-liner does the same
  with `--write-usd out/daylight.usda` using its `meshes.json`. Faces dropped by the converter (USD `holeIndices`,
  faces with < 3 vertices) get `nan`; instance proxies and point-instancer prototypes are skipped with a warning
  (they cannot be overridden per instance).
- **three.js (TypeScript)**: `await applyToThree(mesh, values, {vmin, vmax, material})` sets a `color` attribute
  (per triangle when `values.length` equals the triangle count: the geometry becomes non-indexed; per vertex when it
  equals the point count) and a `MeshStandardMaterial({vertexColors: true})` unless `material: false`. Use the rows
  of the mesh from the manifest range; `fromThree`/`fromGLB` meshes are triangle lists, so faces map 1:1.
- **Anything else**: `daylight.colormap(values, vmin, vmax)` (Python, `[N x 3]` float RGB in 0..1) and
  `colormap(values, {vmin, vmax})` (TypeScript, `Float32Array` RGB) are the same 9-stop viridis with linear
  interpolation, `nan` -> grey; feed them to glTF `COLOR_0`, PLY vertex colours, a Kit extension or a shader. For a
  custom scale keep the raw numbers and colour them yourself.
