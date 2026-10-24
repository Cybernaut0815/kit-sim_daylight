// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// Internal contracts between the translation units. Not part of the public API.
#pragma once
#include "daylight.h"
#include "params.h"         // Slot
#include <cstdio>
#include <vector_types.h>   // float3, uint3 (CUDA toolkit, header-only)

namespace daylight {

// ------------------------------------------------------------------ sky.cpp (Radiance-derived; see THIRD_PARTY_NOTICES.md)
struct Patches {
  int P = 0;                         // sky patches (145 / 577 / 2305), excluding the gendaymtx ground row
  float alpha = 0;                   // row height (rad) = (pi/2)/(7*mf + 0.5)
  std::vector<float> alt, azi;       // patch centre altitude / azimuth (rad), float as in gendaymtx; azi clockwise from +Y
  std::vector<float> omega;          // solid angle (sr); sum over sky patches = 2 pi
  std::vector<int> row, ninrow;      // row index of patch (0 = horizon row, 7*mf = zenith cap), patches in that row
  std::vector<float3> centre;        // gendaymtx centre vectors (cos alt sin azi, cos alt cos azi, sin alt)
  std::vector<float3> lb;            // ladybug trapezoid-normal vectors: same azimuth, tan(altLB) = cos(pi/ninrow) tan(alt)
};
Patches make_patches(int mf);
// Deterministic stratified sample k of K (K = m*m) inside patch p, uniform in solid angle; K == 1 -> exact centre.
float3 patch_sample(const Patches& pa, int p, int k, int K);
// gendaymtx.c ComputeSky + AddDirect for one timestep. out: 3*(P+1) floats RGB; index 0 = ground row, 1..P = sky
// patches in gendaymtx order (= Patches order). alt/azi radians, azi clockwise from north. sun / sky emulate -d / -s
// (both true = total). visible = true: -O0 path (Perez luminous efficacy, dew point 11 C, output = lux/179 per unit);
// false: -O1 radiometric (W/m2/sr). dni/dhi in W/m2. Night (dni+dhi <= 1e-4) -> all zero.
void gendaymtx_step(const Patches& pa, int jd, double alt, double azi, double dni, double dhi, double grefl,
                    bool sun, bool sky, bool visible, float* out);
// Radiance sun.c solar position. lonWestDeg / meridianWestDeg are west-positive degrees (meridian = -15*tz).
// hour = local standard decimal hour. Returns altitude (rad) and azimuth clockwise from north (rad) = sazi + pi.
void sun_radiance(double latDeg, double lonWestDeg, double meridianWestDeg, int jd, double hour, double& alt, double& azi);
double round_fmt(double x, const char* fmt);    // strtod(snprintf(fmt, x)): ASCII round trip, e.g. "%.3g" (gendaymtx), "%.2f" (.wea)
float3 dir_from_altaz(double altRad, double aziRad);   // (cos alt sin azi, cos alt cos azi, sin alt)
float3 rotate_z(float3 v, double deg);                  // counterclockwise about +Z (ladybug north)

// ------------------------------------------------------------------ solar.cpp
struct Sun { double alt = 0, azi = 0; bool up = false; float3 vec{0.f, 0.f, 0.f}; };  // degrees; azi clockwise from north; vec towards the sun (true north; run.cpp rotates)
// NOAA/Meeus solar position with ladybug Sunpath semantics (fixed year 2017 / 2016 leap, 0.01-day rounding of the
// day fraction, NOAA refraction added to altitude, azimuth from the unrefracted zenith, up = refracted alt >= 0).
Sun sun_noaa(double lat, double lon, double tz, int month, int day, double hourFrac, bool leap);
// Evaluation steps from Options (+ Weather). EPW-driven annual: one step per row at row%24 + 0.5 h, dt 1.
// User range / time: stamps and the --hours window in the user's clock, dt = stepMin/60; the sun is evaluated at the
// instant (--time) or the step middle (ranges), moved into the weather's clock by shift = user tz - weather tz (hours),
// with dni/dhi interpolated there (0 without weather).
bool make_steps(const Options& opt, const Weather* w, std::vector<Step>& out, std::string& err, double shift = 0);
// Parse "[YYYY-]MM-DDTHH:MM" to hour of year (leap-aware). Year ignored.
bool parse_hoy(const std::string& s, bool leap, double& hoy, std::string& err);
void hoy_to_date(double hoy, bool leap, int& month, int& day, double& hour);
double extraterrestrial_irradiance(int doy);   // Spencer 1971, W/m2 (pvlib get_extra_radiation default)
// Ineichen-Perez clear sky (pvlib 0.16.1 formulation, no Perez enhancement); apparent zenith clamped to [0, 90] deg,
// Linke turbidity TL, site altitude in metres (standard-atmosphere pressure). Outputs W/m2.
void ineichen(double apparentZenithDeg, double turbidity, double altitudeM, double dniExtra, double& ghi, double& dni, double& dhi);

// ------------------------------------------------------------------ mesh.cpp
struct Tris { std::vector<float3> v; std::vector<uint3> i; std::vector<uint32_t> slot; };   // slot: global slot per triangle
struct Sensors { std::vector<float3> p, n; };              // recentred, offset applied; sides already expanded
void bbox_centre(const std::vector<Mesh>& meshes, double c[3]);
// Append the triangulated faces of m (recentred by c) to out, each triangle's global slot = slotBase + its face slot;
// skipped counts degenerate / < 3-vertex faces.
void triangulate(const Mesh& m, const double c[3], Tris& out, int& skipped, uint32_t slotBase = 0);
// Global slots: every mesh in input order takes slot_count(m) = max face slot + 1 (1 without slots); mesh base = the sum
// over the meshes before it.
size_t slot_count(const Mesh& m);
// Optics rows within range (0..1, rho + tau <= 1; glass rho 0, 0 < tau <= 0.918) and the mode rules (ladybug: no
// transmission). False + err naming the first bad row.
bool optics_ok(const Options& opt, std::string& err);
// One Slot per global slot from the rows (NAME/SLOT > NAME > * > rho 0.2 tau 0; glass: tn_from_Tn(tau), kSlotGlass).
// Errors: unknown or ambiguous mesh name, slot out of range, duplicate row, a Mesh::slot not one per face.
bool resolve_optics(const std::vector<Mesh>& meshes, const std::vector<Optics>& rows, std::vector<Slot>& out, std::string& err);
// Faces of slot s in m (recentred by c) as one closed thin solid: positions welded on a 1e-5 m grid, every edge used
// as often in each direction (closed, consistently wound), 2 |V| / S < 0.1 m -> kSlotThin, | kSlotInward when the signed
// volume is negative; else 0. ponytail: per slot, not per connected component (a pane box sharing its slot with an open
// sheet is not thin, so it passes tau twice); split the slot's faces into components if such slots matter.
unsigned thin_solid(const Mesh& m, unsigned s, const double c[3]);
// Sensor points/normals per face or per point per opt (mode rules, offset, sides). Appends to out: one entry per
// element, or two consecutive entries (front, back) when sides == Both. Returns the element count (faces or points).
size_t make_sensors(const Mesh& m, const Options& opt, const double c[3], Sensors& out, std::string& warn);

// ------------------------------------------------------------------ run.cpp
bool npy_write_header(std::FILE* f, size_t rows, size_t cols, const char* descr);   // .npy v1.0 [rows x cols] C order, descr "<f8" / "<f4" / "<u4" / "<u1"

// ------------------------------------------------------------------ gpu.cu
struct Gpu;
Gpu* gpu_init(std::string& err);
void gpu_destroy(Gpu* g);
// GAS, compacted, inputs freed (t.slot < slots.size()). When a slot transmits (tau > 0): one SBT record per slot
// (within the device's records per GAS), any-hit on the transmitting ones; else one opaque record, the layout of an
// all-opaque scene. Geometry flags are fixed at build: a slot that starts or stops transmitting rebuilds.
bool gpu_build(Gpu& g, const Tris& t, const std::vector<Slot>& slots, std::string& err);
bool gpu_set_slots(Gpu& g, const std::vector<Slot>& slots, std::string& err);   // hit records (every run; same count as the build)
// A built acceleration structure moved out of the Gpu, kept in VRAM for a later run of the same scene.
struct GasSlot { void* dGas = nullptr; unsigned long long gas = 0; size_t bytes = 0; };
GasSlot gpu_detach(Gpu& g);              // the current GAS moves out; the Gpu then has none
void gpu_attach(Gpu& g, GasSlot s);      // frees the current GAS and installs s
void gpu_free(GasSlot& s);
size_t gpu_free_bytes();                 // free device memory (cudaMemGetInfo), 0 on error
bool gpu_set_scene(Gpu& g, const Sensors& s, const std::vector<float3>& dirs, std::string& err);   // resident sensors + directions
// Trace sensors [i0, i0+nc) against all directions; bits stay on device, word-major bits[w*nc + i]. Ray flags with
// any-hit on (gpu_ray_flags(.., true)): transmittance bytes[d*nc + i] instead (x255, 255 = 1), which reduce / series
// then weight by. Adaptive TDR-safe sub-launches inside (first 4096 sensors timed, then scaled to ~150 ms per launch;
// byte mode: 2^13 threads first and at least, then 30 ms, at most doubling).
bool gpu_trace(Gpu& g, size_t i0, size_t nc, unsigned rayFlags, std::string& err);
// out[i] = sum_d T(i,d) * (useCos ? max(0, N_i . D_d) : 1) * W[d], double accumulation; W has nDir host floats.
// T = the bit, or byte / 255.
bool gpu_reduce(Gpu& g, size_t nc, const float* W, bool useCos, double* out, std::string& err);
struct SeriesIn {
  unsigned nSky = 0;          // sky directions are 0..nSky-1
  const int* row = nullptr;   // [nSky] patch index per sky direction (host)
  const float* wsky = nullptr;// [nSky] weight per sky direction (omega/K physical, omega ladybug) (host)
  const float* S = nullptr;   // [P x T] patch radiance per step (host), row-major
  int P = 0, T = 0;
  const float* g = nullptr;   // [T] ground term per step (host)
  const float* G = nullptr;   // [nc] ground visibility per sensor (host, from gpu_reduce with ground weights)
  unsigned sun0 = 0, nSun = 0;// sun direction range in the direction table
  const int* sunStep = nullptr;    // [nSun] step (column) index per sun direction
  const float3* sunVec = nullptr;  // [nSun]
  const float* sunW = nullptr;     // [nSun] DNI (W/m2) or direct illuminance (lux) per sun direction
};
// E[i*T + t] (host, nc x T floats) for the current chunk: sum_sky T*max(0,cos)*wsky*S[row,t] + G*g[t] + sun term.
bool gpu_series(Gpu& g, size_t nc, const SeriesIn& in, float* E, std::string& err);
bool gpu_download_bits(Gpu& g, size_t nc, std::vector<uint32_t>& out, std::string& err);   // [nWords x nc] as on device
bool gpu_download_bytes(Gpu& g, size_t nc, std::vector<uint8_t>& out, std::string& err);   // byte mode: [nDir x nc]
struct GpuStats { double buildMs = 0, traceMs = 0; size_t rays = 0, gasBytes = 0, launches = 0; };
GpuStats gpu_stats(const Gpu& g);
// TERMINATE_ON_FIRST_HIT [| CULL_BACK_FACING_TRIANGLES], | DISABLE_ANYHIT unless a slot transmits (translucent)
unsigned gpu_ray_flags(bool cullBackfaces, bool translucent);

}  // namespace daylight
