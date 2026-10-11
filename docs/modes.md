# Modes and weather

- `--mode physical` (default): patch-centre sky vectors (`--sky-density 1|2|4` = 145/577/2305 patches,
  `--patch-samples K` stratified rays per patch), exact sun vector per step (never binned), Lambertian ground
  (`--ground RHO`), NOAA sun position with refraction (Ladybug Sunpath semantics).
- `--mode ladybug`: emulates `ladybug_radiance.study.RadiationStudy` / `DirectSunStudy` bit for bit: Ladybug's
  trapezoid-normal patch vectors for rays and cosines, `gendaymtx -A -O1 -d/-s` float accumulation with `%.3g`
  rounding, sun spread over the 4 nearest patches, sun.c positions, ground = reflectance x mean sky patch.
  Use with `--no-self-occlude --offset 0` for library parity (the Grasshopper component uses 0.1 m and self-shading).

Weather sources, both modes: an EPW (`--epw`, measured/typical hourly DNI and DHI) or `--clear-sky`: for every hour
of a 365-day year the Ineichen-Perez clear-sky model (as in pvlib `clearsky.ineichen`: Kasten-Young airmass,
Spencer extraterrestrial irradiance, Linke turbidity `--turbidity`, site `--altitude`) at the mid-hour sun
position gives DNI and DHI, which then feed the same Perez sky. It is deterministic and needs no download, but it
has no clouds: use it for relative comparisons (which facade, which overhang), not for absolute yields.
