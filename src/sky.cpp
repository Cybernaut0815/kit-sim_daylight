// SPDX-License-Identifier: MIT
// Ports of Radiance rad5R4 code: gendaymtx.c (ComputeSky, AddDirect, rh_init, Perez model), sun.c (sdec,
// stadj, salt, sazi), tcos.c (table sine used by rh_cos) and the color.h / fvect.h macros they rely on.
// Radiance is distributed under the Radiance Software License, Version 2.0:
//   Copyright (c) 1990-2022 The Regents of the University of California, through Lawrence Berkeley National
//   Laboratory. All rights reserved. See THIRD_PARTY_NOTICES.md for the full licence text and notices.
// Floating-point types and evaluation order mirror the C source (float patch angles / solid angles / matrix
// values, double elsewhere) so that results match gendaymtx bit for bit after float rounding.
#include "internal.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace daylight {
namespace {

constexpr double PI = 3.14159265358979323846;
constexpr double WHTEFFICACY = 179.;       // color.h
constexpr double FTINY = 1e-6;             // fvect.h (double FVECT build)
constexpr double DC_SolarConstantE = 1367.0;
constexpr double dew_point = 11.0;         // gendaymtx default (no -O0 dew point input in .wea)
inline double DegToRad(double d) { return d * (PI / 180.); }
inline double RadToDeg(double r) { return r * (180. / PI); }

/*********************************************************************
 *
 *  H32_gendaylit.CPP - Perez Sky Model Calculation
 *
 *  Version:    1.00A
 *
 *  History:    09/10/01 - Created.
 *				11/10/08 - Modified for Unix compilation.
 *				11/10/12 - Fixed conditional __max directive.
 *				1/11/13 - Tweaks and optimizations (G.Ward)
 *
 *  Compilers:  Microsoft Visual C/C++ Professional V10.0
 *
 *  Author:     Ian Ashdown, P.Eng.
 *              byHeart Consultants Limited
 *              620 Ballantree Road
 *              West Vancouver, B.C.
 *              Canada V7S 1W3
 *              e-mail: ian_ashdown@helios32.com
 *
 *  References:	Perez, R., P. Ineichen, R. Seals, J. Michalsky, and R.
 *				Stewart. 1990. ìModeling Daylight Availability and
 *				Irradiance Components from Direct and Global
 *				Irradiance,î Solar Energy 44(5):271-289.
 *
 *				Perez, R., R. Seals, and J. Michalsky. 1993.
 *				ìAll-Weather Model for Sky Luminance Distribution -
 *				Preliminary Configuration and Validation,î Solar Energy
 *				50(3):235-245.
 *
 *				Perez, R., R. Seals, and J. Michalsky. 1993. "ERRATUM to
 *				All-Weather Model for Sky Luminance Distribution -
 *				Preliminary Configuration and Validation,î Solar Energy
 *				51(5):423.
 *
 *  NOTE:		This program is a completely rewritten version of
 *				gendaylit.c written by Jean-Jacques Delaunay (1994).
 *
 *  Copyright 2009-2012 byHeart Consultants Limited. All rights
 *  reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted for personal and commercial purposes
 *  provided that redistribution of source code must retain the above
 *  copyright notice, this list of conditions and the following
 *  disclaimer:
 *
 *    THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESSED OR IMPLIED
 *    WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 *    OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *    DISCLAIMED. IN NO EVENT SHALL byHeart Consultants Limited OR
 *    ITS EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 *    USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 *    AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *    POSSIBILITY OF SUCH DAMAGE.
 *
 *********************************************************************/

/* Perez sky model coefficients */

/* Reference:	Perez, R., R. Seals, and J. Michalsky, 1993. "All- */
/*				Weather Model for Sky Luminance Distribution - */
/*				Preliminary Configuration and Validation," Solar */
/*				Energy 50(3):235-245, Table 1. */

static const double PerezCoeff[8][20] =
{
	/* Sky clearness (epsilon): 1.000 to 1.065 */
	{   1.3525,  -0.2576,  -0.2690,  -1.4366,   -0.7670,
	    0.0007,   1.2734,  -0.1233,   2.8000,    0.6004,
	    1.2375,   1.0000,   1.8734,   0.6297,    0.9738,
	    0.2809,   0.0356,  -0.1246,  -0.5718,    0.9938 },
    /* Sky clearness (epsilon): 1.065 to 1.230 */
	{  -1.2219,  -0.7730,   1.4148,   1.1016,   -0.2054,
	    0.0367,  -3.9128,   0.9156,   6.9750,    0.1774,
		6.4477,  -0.1239,  -1.5798,  -0.5081,   -1.7812,
		0.1080,   0.2624,   0.0672,  -0.2190,   -0.4285 },
    /* Sky clearness (epsilon): 1.230 to 1.500 */
	{  -1.1000,  -0.2515,   0.8952,   0.0156,    0.2782,
	   -0.1812, - 4.5000,   1.1766,  24.7219,  -13.0812,
	  -37.7000,  34.8438,  -5.0000,   1.5218,    3.9229,
	   -2.6204,  -0.0156,   0.1597,   0.4199,   -0.5562 },
    /* Sky clearness (epsilon): 1.500 to 1.950 */
	{  -0.5484,  -0.6654,  -0.2672,   0.7117,   0.7234,
	   -0.6219,  -5.6812,   2.6297,  33.3389, -18.3000,
	  -62.2500,  52.0781,  -3.5000,   0.0016,   1.1477,
	    0.1062,   0.4659,  -0.3296,  -0.0876,  -0.0329 },
    /* Sky clearness (epsilon): 1.950 to 2.800 */
	{  -0.6000,  -0.3566,  -2.5000,   2.3250,   0.2937,
	    0.0496,  -5.6812,   1.8415,  21.0000,  -4.7656 ,
	  -21.5906,   7.2492,  -3.5000,  -0.1554,   1.4062,
	    0.3988,   0.0032,   0.0766,  -0.0656,  -0.1294 },
    /* Sky clearness (epsilon): 2.800 to 4.500 */
	{  -1.0156,  -0.3670,   1.0078,   1.4051,   0.2875,
	   -0.5328,  -3.8500,   3.3750,  14.0000,  -0.9999,
	   -7.1406,   7.5469,  -3.4000,  -0.1078,  -1.0750,
	    1.5702,  -0.0672,   0.4016,   0.3017,  -0.4844 },
    /* Sky clearness (epsilon): 4.500 to 6.200 */
	{  -1.0000,   0.0211,   0.5025,  -0.5119,  -0.3000,
	    0.1922,   0.7023,  -1.6317,  19.0000,  -5.0000,
		1.2438,  -1.9094,  -4.0000,   0.0250,   0.3844,
		0.2656,   1.0468,  -0.3788,  -2.4517,   1.4656 },
    /* Sky clearness (epsilon): 6.200 to ... */
	{  -1.0500,   0.0289,   0.4260,   0.3590,  -0.3250,
	    0.1156,   0.7781,   0.0025,  31.0625, -14.5000,
	  -46.1148,  55.3750,  -7.2312,   0.4050,  13.3500,
	    0.6234,   1.5000,  -0.6426,   1.8564,   0.5636 }
};

/* Perez irradiance component model coefficients */

/* Reference:	Perez, R., P. Ineichen, R. Seals, J. Michalsky, and R. */
/*				Stewart. 1990. ìModeling Daylight Availability and */
/*				Irradiance Components from Direct and Global */
/*				Irradiance,î Solar Energy 44(5):271-289. */

typedef struct
{
	double lower;	/* Lower bound */
	double upper;	/* Upper bound */
} CategoryBounds;

/* Perez sky clearness (epsilon) categories (Table 1) */
static const CategoryBounds SkyClearCat[8] =
{
	{ 1.000, 1.065 },	/* Overcast */
	{ 1.065, 1.230 },
	{ 1.230, 1.500 },
	{ 1.500, 1.950 },
	{ 1.950, 2.800 },
	{ 2.800, 4.500 },
	{ 4.500, 6.200 },
	{ 6.200, 12.01 }	/* Clear */
};

/* Luminous efficacy model coefficients */
typedef struct
{
	double a;
	double b;
	double c;
	double d;
} ModelCoeff;

/* Diffuse luminous efficacy model coefficients (Table 4, Eqn. 7) */
static const ModelCoeff DiffuseLumEff[8] =
{
	{  97.24, -0.46,  12.00,  -8.91 },
	{ 107.22,  1.15,   0.59,  -3.95 },
	{ 104.97,  2.96,  -5.53,  -8.77 },
	{ 102.39,  5.59, -13.95, -13.90 },
	{ 100.71,  5.94, -22.75, -23.74 },
	{ 106.42,  3.83, -36.15, -28.83 },
	{ 141.88,  1.90, -53.24, -14.03 },
	{ 152.23,  0.35, -45.27,  -7.98 }
};

/* Direct luminous efficacy model coefficients (Table 4, Eqn. 8) */
static const ModelCoeff DirectLumEff[8] =
{
	{  57.20, -4.55, -2.98, 117.12 },
	{  98.99, -3.46, -1.21,  12.38 },
	{ 109.83, -4.90, -1.71,  -8.81 },
	{ 110.34, -5.84, -1.99,  -4.56 },
	{ 106.36, -3.97, -1.75,  -6.16 },
	{ 107.19, -1.25, -1.51, -26.73 },
	{ 105.75,  0.77, -1.26, -34.44 },
	{ 101.18,  1.58, -1.10,  -8.29 }
};

// tcos.c: table-based cosine, NCOSENTRY 1024, no interpolation. Active in every rad5R4 build that does not define
// __FAST_MATH__ (MSVC never does; Radiance's CMake adds no -ffast-math), so gendaymtx's rh_cos() uses it.
double tcos(double x)
{
	constexpr int NCOSENTRY = 1024;
	static double	costab[NCOSENTRY+1];
	int	i;

	if (costab[0] < 0.5)		/* initialize table */
		for (i = 0; i <= NCOSENTRY; i++)
			costab[i] = std::cos((PI/2./NCOSENTRY)*i);
					/* normalize angle */
	if (x < 0.)
		x = -x;
	i = (int)((NCOSENTRY*2./PI) * x  +  0.5);
	while (i >= 4*NCOSENTRY)
		i -= 4*NCOSENTRY;
	switch (i / NCOSENTRY) {
	case 0:
		return(costab[i]);
	case 1:
		return(-costab[(2*NCOSENTRY)-i]);
	case 2:
		return(-costab[i-(2*NCOSENTRY)]);
	default:
		return(costab[(4*NCOSENTRY)-i]);
	}
}
inline double tsin(double x) { return tcos(x - (PI/2.)); }   // rtmath.h

// gendaymtx vector(v, alt, azi) macro: (cos alt sin azi, cos alt cos azi, sin alt) in double.
inline void rad_vector(double v[3], double alt, double azi) {
  v[1] = std::cos(alt); v[0] = v[1] * std::sin(azi); v[1] *= std::cos(azi); v[2] = std::sin(alt);
}

}  // namespace

double tsin_radiance(double x) { return tsin(x); }   // test hook (check_sky.cpp): the normalisation sine

/* Initialize Reinhart sky patch positions (GW) -- rh_init(), minus the ground row (gendaymtx index 0) */
Patches make_patches(int mf) {
  static const int tnaz[7] = {30, 30, 24, 24, 18, 12, 6};
  const double alpha = (PI/2.)/(7*mf + .5);
  Patches pa;
  pa.alpha = (float)alpha;
  for (int i = 0; i < 7*mf; i++) {
    const float ralt = (float)(alpha*(i + .5));
    const int ninrow = tnaz[i/mf]*mf;
    const float dom = (float)(2.*PI*(std::sin(alpha*(i+1)) - std::sin(alpha*i)) / (double)ninrow);
    for (int j = 0; j < ninrow; j++) {
      pa.alt.push_back(ralt);
      pa.azi.push_back((float)(2.*PI * j / (double)ninrow));
      pa.omega.push_back(dom);
      pa.row.push_back(i);
      pa.ninrow.push_back(ninrow);
    }
  }
  pa.alt.push_back((float)(PI/2.));                              // zenith cap
  pa.azi.push_back(0.f);
  pa.omega.push_back((float)(2.*PI*(1. - std::cos(alpha*.5))));
  pa.row.push_back(7*mf);
  pa.ninrow.push_back(1);
  pa.P = (int)pa.alt.size();
  for (int p = 0; p < pa.P; p++) {
    pa.centre.push_back(dir_from_altaz(pa.alt[p], pa.azi[p]));
    // ladybug: unit normal of the patch trapezoid, tan(altLB) = cos(pi/ninrow) tan(altC); zenith exactly +Z
    const double altLB = p == pa.P - 1 ? PI/2 : std::atan(std::cos(PI/pa.ninrow[p]) * std::tan((double)pa.alt[p]));
    pa.lb.push_back(p == pa.P - 1 ? float3{0.f, 0.f, 1.f} : dir_from_altaz(altLB, pa.azi[p]));
  }
  return pa;
}

float3 patch_sample(const Patches& pa, int p, int k, int K) {
  if (K <= 1) return pa.centre[p];
  const int m = (int)std::lround(std::sqrt((double)K));
  const double u = ((k % m) + 0.5) / m, v = ((k / m) + 0.5) / m, a = pa.alpha;
  if (p == pa.P - 1) {                     // zenith cap: cos(theta) uniform in [cos(alpha/2), 1]
    const double ct = 1 - v * (1 - std::cos(a / 2)), st = std::sqrt(1 - ct * ct), az = 2 * PI * u;
    return float3{(float)(st * std::sin(az)), (float)(st * std::cos(az)), (float)ct};
  }
  const int i = pa.row[p];
  const double s0 = std::sin(a * i), s1 = std::sin(a * (i + 1));
  return dir_from_altaz(std::asin(s0 + v * (s1 - s0)), pa.azi[p] + (u - 0.5) * 2 * PI / pa.ninrow[p]);
}

float3 dir_from_altaz(double altRad, double aziRad) {
  double v[3];
  rad_vector(v, altRad, aziRad);
  return float3{(float)v[0], (float)v[1], (float)v[2]};
}

float3 rotate_z(float3 v, double deg) {
  const double t = DegToRad(deg), c = std::cos(t), s = std::sin(t);
  return float3{(float)(v.x * c - v.y * s), (float)(v.x * s + v.y * c), v.z};
}

/* ComputeSky + AddDirect for one time step. parr = mtx_data row: [0] ground, [1..P] sky (float, as mtx_data). */
void gendaymtx_step(const Patches& pa, int jd, double altitude, double azimuth, double dni, double dhi, double grefl_in,
                    bool sun, bool sky, bool visible, float* parr) {
  const int nskypatch = pa.P + 1;
  if (dni + dhi <= 1e-4) {              /* effectively nighttime? */
    std::memset(parr, 0, sizeof(float)*3*nskypatch);
    return;
  }
  // -d (sun only) zeroes skycolor AND grefl; -s (sky only) zeroes suncolor. COLOR is float in color.h.
  const float skycolor[3] = {sky ? .96f : 0.f, sky ? 1.004f : 0.f, sky ? 1.118f : 0.f};
  const float suncolor = sun ? 1.f : 0.f;
  const float grefl = sky ? (float)grefl_in : 0.f;
  const double dir_irrad = dni, diff_irrad = dhi;

  /* Calculate atmospheric precipitable water content */
  const double apwc = std::exp(0.07 * dew_point - 0.075);

  /* Calculate sun zenith angle (don't let it dip below horizon) */
  /* Also limit minimum angle to keep circumsolar off zenith */
  double sun_zenith;
  if (altitude <= 0.0)
    sun_zenith = DegToRad(90.0);
  else if (altitude >= DegToRad(87.0))
    sun_zenith = DegToRad(3.0);
  else
    sun_zenith = DegToRad(90.0) - altitude;

  /* CalcSkyBrightness: CalcAirMass (Kasten 1966) and CalcEccentricity (Sen 2008) */
  const double day_angle = (jd - 1.0) * (2.0 * PI / 365.0);
  const double E0 = 1.00011 + 0.034221 * std::cos(day_angle) + 0.00128 * std::sin(day_angle)
      + 0.000719 * std::cos(2.0 * day_angle) + 0.000077 * std::sin(2.0 * day_angle);
  const double air_mass = 1.0 / (std::cos(sun_zenith) + 0.15 * std::pow(93.885 - RadToDeg(sun_zenith), -1.253));
  double sky_brightness = diff_irrad * air_mass / (DC_SolarConstantE * E0);
  /* CalcSkyClearness: diff_irrad == 0 -> inf -> clamped to 11.9 below (host code is /fp:precise, never fast-math) */
  const double sz_cubed = sun_zenith*sun_zenith*sun_zenith;
  double sky_clearness = ((diff_irrad + dir_irrad) / diff_irrad + 1.041 * sz_cubed) / (1.0 + 1.041 * sz_cubed);

  /* Limit sky clearness */
  if (sky_clearness > 11.9)
    sky_clearness = 11.9;
  /* Limit sky brightness */
  if (sky_brightness < 0.01)
    sky_brightness = 0.01;

  /* GetCategoryIndex */
  int index;
  for (index = 0; index < 8; index++)
    if ((sky_clearness >= SkyClearCat[index].lower) && (sky_clearness < SkyClearCat[index].upper))
      break;
  if (index > 7) index = 7;   // no category (eps < 1 from negative DNI, or NaN): the C code reads past the table

  /* Calculate illuminance */
  double diff_illum, dir_illum;
  if (visible) {                       /* -O0: CalcDiffuseIllumRatio / CalcDirectIllumRatio */
    const ModelCoeff* pnle = &DiffuseLumEff[index];
    diff_illum = diff_irrad * (pnle->a + pnle->b * apwc + pnle->c * std::cos(sun_zenith) + pnle->d * std::log(sky_brightness));
    pnle = &DirectLumEff[index];
    const double r = pnle->a + pnle->b * apwc + pnle->c * std::exp(5.73 * sun_zenith - 5.0) + pnle->d * sky_brightness;
    dir_illum = dir_irrad * (r > 0.0 ? r : 0.0);
  } else {                             /* -O1: hack for solar radiance */
    diff_illum = diff_irrad * WHTEFFICACY;
    dir_illum = dir_irrad * WHTEFFICACY;
  }

  /* Compute ground radiance (include solar contribution if any); float/double mixing as in the C code */
  parr[0] = (float)diff_illum;
  if (altitude > 0)
    parr[0] = (float)(parr[0] + dir_illum * std::sin(altitude));
  parr[2] = parr[1] = parr[0] = (float)(parr[0] * (1./PI/WHTEFFICACY));
  parr[0] *= grefl; parr[1] *= grefl; parr[2] *= grefl;      /* multcolor(parr, grefl) */

  if (!sky) {                                                  /* bright(skycolor) <= 1e-4: 0 sky component */
    std::memset(parr+3, 0, sizeof(float)*3*(nskypatch-1));
  } else {
    /* CalcPerezParam(sun_zenith, sky_clearness, sky_brightness, index) */
    const double sz = sun_zenith, epsilon = sky_clearness;
    double delta = sky_brightness;
    double perez_param[5];
    /* Limit sky brightness */
    if (epsilon > 1.065 && epsilon < 2.8)
    {
      if (delta < 0.2)
        delta = 0.2;
    }
    const double (*x)[4] = reinterpret_cast<const double (*)[4]>(PerezCoeff[index]);   /* x[i][j] = PerezCoeff[index][4*i+j] */
    if (index != 0)
    {
      /* Calculate parameter a, b, c, d and e (Eqn. 6) */
      for (int i = 0; i < 5; i++)
        perez_param[i] = x[i][0] + x[i][1] * sz + delta * (x[i][2] + x[i][3] * sz);
    }
    else
    {
      /* Parameters a, b and e (Eqn. 6) */
      perez_param[0] = x[0][0] + x[0][1] * sz + delta * (x[0][2] + x[0][3] * sz);
      perez_param[1] = x[1][0] + x[1][1] * sz + delta * (x[1][2] + x[1][3] * sz);
      perez_param[4] = x[4][0] + x[4][1] * sz + delta * (x[4][2] + x[4][3] * sz);
      /* Parameter c (Eqn. 7) */
      perez_param[2] = std::exp(std::pow(delta * (x[2][0] + x[2][1] * sz), x[2][2])) - x[2][3];
      /* Parameter d (Eqn. 8) */
      perez_param[3] = -std::exp(delta * (x[3][0] + x[3][1] * sz)) + x[3][2] + delta * x[3][3];
    }

    /* CalcSkyPatchLumin: Perez All-Weather relative luminance per patch (acos argument unclamped, as in the C code) */
    for (int i = 1; i < nskypatch; i++)
    {
      const double aas = std::fabs(pa.azi[i-1] - azimuth);             /* sun-sky point azimuthal angle */
      const double zsa = PI * 0.5 - pa.alt[i-1];                        /* zenithal sun angle */
      const double sspa = std::acos(std::cos(sun_zenith) * std::cos(zsa) + std::sin(sun_zenith) *
          std::sin(zsa) * std::cos(aas));                              /* sun-sky point angle */
      parr[3*i] = (float)((1.0 + perez_param[0] * std::exp(perez_param[1] / std::cos(zsa))) *
          (1.0 + perez_param[2] * std::exp(perez_param[3] * sspa) + perez_param[4] * std::cos(sspa) * std::cos(sspa)));
      if (parr[3*i] < 0) parr[3*i] = 0;
      parr[3*i+2] = parr[3*i+1] = parr[3*i];
    }

    /* CalcRelHorzIllum (rh_cos(i) = tsin(rh_palt[i]), the table sine) */
    double norm_diff_illum = 0.0;
    for (int i = 1; i < nskypatch; i++)
      norm_diff_illum += parr[3*i+1] * tsin(pa.alt[i-1]) * pa.omega[i-1];

    /* Check for zero sky -- make uniform in that case */
    if (norm_diff_illum <= FTINY) {
      for (int i = 1; i < nskypatch; i++)
        parr[3*i] = parr[3*i+1] = parr[3*i+2] = 1.f;
      norm_diff_illum = PI;
    }
    /* Normalization coefficient */
    norm_diff_illum = diff_illum / norm_diff_illum;

    /* Apply to sky patches to get absolute radiance values: scalecolor (double factor) then multcolor (float) */
    const double sf = norm_diff_illum*(1./WHTEFFICACY);
    for (int i = 1; i < nskypatch; i++) {
      parr[3*i] = (float)(parr[3*i] * sf) * skycolor[0];
      parr[3*i+1] = (float)(parr[3*i+1] * sf) * skycolor[1];
      parr[3*i+2] = (float)(parr[3*i+2] * sf) * skycolor[2];
    }
  }

  /* AddDirect: add in solar direct to nearest sky patches (GW) */
  constexpr int NSUNPATCH = 4;
  if (dir_illum <= 1e-4 || !sun)        /* bright(suncolor) <= 1e-4 */
    return;
  double near_dprod[NSUNPATCH];
  int near_patch[NSUNPATCH] = {0, 0, 0, 0};
  double wta[NSUNPATCH], wtot;
  int i, j;
  for (i = NSUNPATCH; i--; )
    near_dprod[i] = -1.;
  double svec[3];
  rad_vector(svec, altitude, azimuth);
  for (int p = 1; p < nskypatch; p++) {
    double pvec[3];
    rad_vector(pvec, pa.alt[p-1], pa.azi[p-1]);
    const double dprod = pvec[0]*svec[0] + pvec[1]*svec[1] + pvec[2]*svec[2];
    for (i = 0; i < NSUNPATCH; i++)
      if (dprod > near_dprod[i]) {
        for (j = NSUNPATCH; --j > i; ) {
          near_dprod[j] = near_dprod[j-1];
          near_patch[j] = near_patch[j-1];
        }
        near_dprod[i] = dprod;
        near_patch[i] = p;
        break;
      }
  }
  wtot = 0;                             /* weight by proximity (summation order as in the C code) */
  for (i = NSUNPATCH; i--; )
    wtot += wta[i] = 1./(1.002 - near_dprod[i]);
                                        /* add to nearest patch radiances */
  for (i = NSUNPATCH; i--; ) {
    float* pdest = parr + 3*near_patch[i];
    float val_add = (float)(wta[i] * dir_illum / (WHTEFFICACY * wtot));
    val_add /= pa.omega[near_patch[i]-1];
    pdest[0] += val_add*suncolor;
    pdest[1] += val_add*suncolor;
    pdest[2] += val_add*suncolor;
  }
}

/* sun.c (rad5R4): sdec, stadj, salt, sazi; site angles in radians, longitude / meridian west-positive */
void sun_radiance(double latDeg, double lonWestDeg, double meridianWestDeg, int jd, double hour, double& alt, double& azi) {
  const double s_latitude = DegToRad(latDeg), s_longitude = DegToRad(lonWestDeg), s_meridian = DegToRad(meridianWestDeg);
  const double sd = 0.4093 * std::sin((2.*PI/365.) * (jd - 81));                       /* sdec */
  const double sta = 0.170 * std::sin((4.*PI/373.) * (jd - 80)) -
      0.129 * std::sin((2.*PI/355.) * (jd - 8)) + (12./PI) * (s_meridian - s_longitude);  /* stadj */
  const double st = hour + sta;
  alt = std::asin(std::sin(s_latitude) * std::sin(sd) - std::cos(s_latitude) * std::cos(sd) * std::cos(st*(PI/12.)));
  azi = -std::atan2(std::cos(sd)*std::sin(st*(PI/12.)),
      -std::cos(s_latitude)*std::sin(sd) - std::sin(s_latitude)*std::cos(sd)*std::cos(st*(PI/12.))) + PI;
}

double round_fmt(double x, const char* fmt) {
  char buf[32];
#pragma warning(suppress : 4774)   // non-literal format: callers pass "%.3g" / "%.2f"
  std::snprintf(buf, sizeof buf, fmt, x);
  return std::strtod(buf, nullptr);
}

}  // namespace daylight
