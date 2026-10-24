# Modes and weather

- `--mode physical` (default): patch-centre sky vectors (`--sky-density 1|2|4` = 145/577/2305 patches,
  `--patch-samples K` stratified rays per patch), exact sun vector per step (never binned), Lambertian ground
  (`--ground RHO`), NOAA sun position with refraction (Ladybug Sunpath semantics).
- `--mode ladybug`: emulates `ladybug_radiance.study.RadiationStudy` / `DirectSunStudy` bit for bit: Ladybug's
  trapezoid-normal patch vectors for rays and cosines, `gendaymtx -A -O1 -d/-s` float accumulation with `%.3g`
  rounding, sun spread over the 4 nearest patches, sun.c positions, ground = reflectance x mean sky patch.
  Use with `--no-self-occlude --offset 0` for library parity (the Grasshopper component uses 0.1 m and self-shading).

Surfaces, both modes by default: every triangle is opaque and black; the only reflection is the infinite Lambertian
ground (`--ground`) seen by rays that leave the model. `--optics` gives numbers per mesh or face slot (no material
names: the caller derives them); `--transmission on` (physical mode only, refused in ladybug mode, where `--optics` is
a no-op) lets glass and porous slots pass light, with `--fresnel` choosing angle-dependent glass (Radiance `glass`,
n 1.52) or a flat tau. Ray bending is not modelled: a pane's faces are parallel, light leaves in its own direction.
Sun hours with transmission count equivalent full-sun hours (an hour behind tau 0.6 counts 0.6). For EN 17037 / BRE
sunlight checks on a window, leave the window's own glazing out of the model, or give its slot an Opening, tau 1
(`--optics Facade/2=0,1 --transmission on`): the count then stays exact.
`--bounces N` (physical mode only) adds the light the model reflects: up to N diffuse reflections by each slot's rho,
glass mirroring its angle-dependent reflectance, binned into the sky patches (the sun binned as gendaymtx does, as
`rcontrib` daylight coefficients) and the virtual ground. It goes into radiation, diffuse, irradiance and illuminance,
never into direct or sun hours. Facades gain most: in a 1:1 street canyon with a ground mesh (everything rho 0.2) one
bounce raises the annual radiation of the north-facing wall by about a third and of the south-facing one by 6 %; a
modelled ground with one bounce gives a facade about what the virtual ground gave it without the mesh, minus the
model's shadow on it.

Weather sources, both modes: an EPW (`--epw`, measured/typical hourly DNI and DHI) or `--clear-sky`: for every hour
of a 365-day year the Ineichen-Perez clear-sky model (as in pvlib `clearsky.ineichen`: Kasten-Young airmass,
Spencer extraterrestrial irradiance, Linke turbidity `--turbidity`, site `--altitude`) at the mid-hour sun
position gives DNI and DHI, which then feed the same Perez sky. It is deterministic and needs no download, but it
has no clouds: use it for relative comparisons (which facade, which overhang), not for absolute yields.
