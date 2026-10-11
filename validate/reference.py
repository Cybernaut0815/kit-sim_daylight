# SPDX-License-Identifier: MIT
"""Reference data for compare.py -> validate/_out/ref/. Needs setup.py and fixtures.py first.

  uv run reference.py [--epw STEM ...] [--scene NAME ...] [--all]      default: Boston + Chicago, all scenes

Per EPW (<ref>/epw/<stem>/): gendaymtx -m 1/2 -d/-s -O1 -A ASCII outputs (sky_m{1,2}_{d,s}_O1_A.txt), gendaymtx -D
suns (suns_D.npz), ladybug Sunpath (sunpath.npz), pvlib SPA (pvlib.npz), and ladybug SkyMatrix kWh/m2 per patch
(skymtx_m{mf}_n{north}.npz: dir, dif, rho).
Per scene (<ref>/scene/<name>/): intersection matrices (imtx_m{mf}_n{north}_{face,vertex}.npy, float cos*visible,
columns [sky | ground]) and per EPW radiation values (rad_<stem>_m{mf}_n{north}_{face,vertex}.npy); DirectSunStudy
per site (sun_imtx_<stem>_{face,vertex}.npy bool [N x n_up], sunhours_<stem>_{face,vertex}.npy).
index.json lists every file with its parameters and timings."""
import argparse
import json
import subprocess
import sys
import time
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from ladybug.epw import EPW
from ladybug.sunpath import Sunpath
from ladybug.wea import Wea
from ladybug_geometry.geometry3d import Mesh3D, Point3D, Vector3D
from ladybug_radiance.config import folders

HERE = Path(__file__).resolve().parent
DL, OUT = HERE / '_dl', HERE / '_out'
REF, FIX = OUT / 'ref', OUT / 'fixtures'
folders.radiance_path = (DL / 'radiance_path.txt').read_text().strip()   # before the modules that bake in the exe paths
from ladybug_radiance.skymatrix import SkyMatrix
from ladybug_radiance.study.directsun import DirectSunStudy
from ladybug_radiance.study.radiation import RadiationStudy
GENDAYMTX = str(Path(folders.radbin_path) / 'gendaymtx.exe')
DEFAULT_EPWS = ['USA_MA_Boston-City.WSO_TMY', 'USA_IL_Chicago-OHare.Intl.AP.725300_TMY3']
SCENES = ['flat', 'flat_tri', 'walls', 'box', 'tilt30', 'overhang']
index = {'epw': {}, 'scene': {}, 'timings': {}}


def timed(key, fn):
    t = time.time()
    r = fn()
    index['timings'][key] = round(time.time() - t, 2)
    print(f'  {key}: {index["timings"][key]} s')
    return r


def gendaymtx(args):
    r = subprocess.run([GENDAYMTX] + args, capture_output=True, check=False)
    assert r.returncode == 0, r.stderr.decode(errors='replace')[-500:]
    return r.stdout


def parse_suns(text):
    """gendaymtx -D file -> (hoy, vec[3], radiance) per written sun. Source ids are solar<minute-of-year>."""
    tok = text.split()
    hoy, vec, rad = [], [], []
    for i, t in enumerate(tok):
        if t == 'light':
            hoy.append(int(tok[i + 1][5:]) / 60.0)
            rad.append(float(tok[i + 5]))
        elif t == 'source':
            vec.append([float(x) for x in tok[i + 5:i + 8]])
    assert len(hoy) == len(vec), (len(hoy), len(vec))
    return np.array(hoy), np.array(vec), np.array(rad)


def epw_refs(stem):
    d = REF / 'epw' / stem
    d.mkdir(parents=True, exist_ok=True)
    epw_path = DL / 'epw' / f'{stem}.epw'
    epw = EPW(str(epw_path))
    wea = Wea.from_epw_file(str(epw_path))
    wea_path = wea.write(str(d / 'wea'))              # ladybug writes the x.5 hours; our wea emulation mirrors it
    n = len(wea.direct_normal_irradiance)
    files = {'wea': Path(wea_path).name, 'rows': n, 'lat': epw.location.latitude, 'lon': epw.location.longitude,
             'tz': epw.location.time_zone, 'elev': epw.location.elevation, 'leap': wea.is_leap_year}
    for mf in (1, 2):
        for flag in ('d', 's'):
            name = f'sky_m{mf}_{flag}_O1_A'
            (d / f'{name}.txt').write_bytes(gendaymtx(['-m', str(mf), f'-{flag}', '-O1', '-A', wea_path]))
            files[name] = f'{name}.txt'
    # (c) gendaymtx -D suns (matrix on stdout discarded)
    gendaymtx(['-m', '1', '-O1', '-D', str(d / 'suns.rad'), '-M', str(d / 'suns.mod'), wea_path])
    hoy, vec, rad = parse_suns((d / 'suns.rad').read_text())
    np.savez(d / 'suns_D.npz', hoy=hoy, vec=vec, rad=rad)
    files['suns_D'] = 'suns_D.npz'
    # (b) ladybug Sunpath at x.5 hoys (north 0) + pvlib SPA
    sp = Sunpath.from_location(epw.location)
    sp.is_leap_year = wea.is_leap_year
    suns = timed(f'{stem}/sunpath', lambda: [sp.calculate_sun_from_hoy(h + 0.5) for h in range(n)])
    np.savez(d / 'sunpath.npz', hoy=np.arange(n) + 0.5, alt=[s.altitude for s in suns], azi=[s.azimuth for s in suns],
             up=[s.is_during_day for s in suns], vec=[list(s.sun_vector) for s in suns])
    files['sunpath'] = 'sunpath.npz'
    year = 2016 if wea.is_leap_year else 2017
    tz = epw.location.time_zone
    times = pd.date_range(f'{year}-01-01 00:30', periods=n, freq='h', tz=timezone(timedelta(hours=tz)))
    spa = pvlib.solarposition.spa_python(times, epw.location.latitude, epw.location.longitude, epw.location.elevation)
    np.savez(d / 'pvlib.npz', hoy=np.arange(n) + 0.5, alt=spa['apparent_elevation'].values, azi=spa['azimuth'].values)
    files['pvlib'] = 'pvlib.npz'
    # ladybug SkyMatrix values (kWh/m2 per patch, after %.3g and the row coefficients) for the W reconstruction
    for mf in (1, 2):
        for north in (0, 30):
            sm = SkyMatrix(wea, north=north, high_density=(mf == 2))
            sm.compute_sky()
            np.savez(d / f'skymtx_m{mf}_n{north}.npz', dir=sm.data[1], dif=sm.data[2], rho=sm.ground_reflectance)
            files[f'skymtx_m{mf}_n{north}'] = f'skymtx_m{mf}_n{north}.npz'
    index['epw'][stem] = files


def load_scene(name):
    js = json.loads((FIX / name / 'fixture.json').read_text())
    mk = lambda m: Mesh3D([Point3D(*p) for p in m['vertices']], [tuple(f) for f in m['faces']])
    return [mk(m) for m in js['sensors']], [mk(m) for m in js['context']]


def scene_refs(name, epws):
    d = REF / 'scene' / name
    d.mkdir(parents=True, exist_ok=True)
    sensors, context = load_scene(name)
    mesh = Mesh3D.join_meshes(sensors) if len(sensors) > 1 else sensors[0]
    if name == 'overhang':
        context = context + [mesh]            # self-shading: the library never adds the study mesh by itself
    sim = d / 'sim'
    files = {'self_shading': name == 'overhang', 'n_faces': len(mesh.faces), 'n_vertices': len(mesh.vertices)}
    # (d) RadiationStudy: the intersection matrix depends on (mf, north, by_vertex) only; values per EPW
    for mf in (1, 2):
        for north in (0, 30):
            for by_vertex in (False, True):
                tag = f'm{mf}_n{north}_{"vertex" if by_vertex else "face"}'
                study = None
                for stem in epws:
                    wea = Wea.from_epw_file(str(DL / 'epw' / f'{stem}.epw'))
                    sm = SkyMatrix(wea, north=north, high_density=(mf == 2))
                    if study is None:
                        study = RadiationStudy(sm, mesh, context, 0, by_vertex, str(sim))
                        timed(f'{name}/imtx_{tag}', study.compute)
                        np.save(d / f'imtx_{tag}.npy', np.asarray(study.intersection_matrix, dtype=np.float64))
                        files[f'imtx_{tag}'] = f'imtx_{tag}.npy'
                    else:
                        study.sky_matrix = sm      # keeps the intersection matrix, recomputes the values
                    np.save(d / f'rad_{stem}_{tag}.npy', np.asarray(study.radiation_values))
                    files[f'rad_{stem}_{tag}'] = f'rad_{stem}_{tag}.npy'
    # (e) DirectSunStudy with Sunpath vectors at x.5 for sun-up hours; one per distinct site
    sites = {}
    for stem in epws:
        loc = EPW(str(DL / 'epw' / f'{stem}.epw')).location
        sites.setdefault((loc.latitude, loc.longitude, loc.time_zone), stem)
    for stem in sites.values():
        sp = np.load(REF / 'epw' / stem / 'sunpath.npz')
        up = sp['up']
        vecs = [Vector3D(*v) for v in sp['vec'][up]]
        for by_vertex in (False, True):
            tag = f'{stem}_{"vertex" if by_vertex else "face"}'
            study = DirectSunStudy(vecs, mesh, context, 1, 0, by_vertex, str(sim))
            timed(f'{name}/sun_{tag}', study.compute)
            np.save(d / f'sun_imtx_{tag}.npy', np.asarray(study.intersection_matrix, dtype=bool))
            np.save(d / f'sunhours_{tag}.npy', np.asarray(study.direct_sun_hours))
            files[f'sun_imtx_{tag}'] = f'sun_imtx_{tag}.npy'
            files[f'sunhours_{tag}'] = f'sunhours_{tag}.npy'
    index['scene'][name] = files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epw', nargs='*', default=DEFAULT_EPWS)
    ap.add_argument('--scene', nargs='*', default=SCENES)
    ap.add_argument('--all', action='store_true', help='every EPW in _dl/epw')
    a = ap.parse_args()
    epws = sorted(p.stem for p in (DL / 'epw').glob('*.epw')) if a.all else a.epw
    REF.mkdir(parents=True, exist_ok=True)
    if (REF / 'index.json').exists():
        index.update(json.loads((REF / 'index.json').read_text()))
    for stem in epws:
        print(stem)
        epw_refs(stem)
        (REF / 'index.json').write_text(json.dumps(index, indent=1))
    for name in a.scene:
        print(name)
        scene_refs(name, epws)
        (REF / 'index.json').write_text(json.dumps(index, indent=1))


if __name__ == '__main__':
    sys.exit(main())
