# SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
"""Fetch the validation inputs into validate/_dl (gitignored): Radiance rad5R4 portable zip and EPW files.

Run from validate/: uv run setup.py. Idempotent: existing downloads are kept; sha256.txt is rewritten.
"""
import hashlib
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DL = HERE / '_dl'
EPW_DIR = DL / 'epw'
RAD_DIR = DL / 'radiance'
RAD_URL = 'https://github.com/LBNL-ETA/Radiance/releases/download/rad5R4/Radiance_4ee32974_Windows.zip'
RAD_EXES = ['gendaymtx.exe', 'rcontrib.exe', 'oconv.exe', 'rad.exe', 'obj2mesh.exe']
EPW_URLS = [
    'https://github.com/NREL/EnergyPlus/raw/develop/weather/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw',
    'https://climate.onebuilding.org/WMO_Region_5_Southwest_Pacific/SGP_Singapore/SGP_SG_Singapore-Changi.Intl.AP.486980_TMYx.zip',
    'https://climate.onebuilding.org/WMO_Region_6_Europe/NOR_Norway/TR_Troms/NOR_TR_Tromso.AP-Langnes.010250_TMYx.zip',
    'https://climate.onebuilding.org/WMO_Region_2_Asia/IND_India/DL_Delhi/IND_DL_New.Delhi-Safdarjung.AP.421820_TMYx.zip',
]
LOCAL_EPWS = ['C:/DAYSIM/wea/USA_MA_Boston-City.WSO_TMY.epw', 'C:/DAYSIM/wea/CAN_ON_Ottawa_CWEC.epw']
CHICAGO = 'USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw'
BOSTON = 'USA_MA_Boston-City.WSO_TMY.epw'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url, dest):
    if dest.exists():
        print(f'  kept {dest.name}')
        return
    print(f'  GET {url}')
    req = urllib.request.Request(url, headers={'User-Agent': 'daylight-validate/0.1'})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, 'wb') as f:
        shutil.copyfileobj(r, f)


def setup_radiance(shas):
    zpath = DL / Path(RAD_URL).name
    download(RAD_URL, zpath)
    shas[zpath.name] = sha256(zpath)
    if not any(RAD_DIR.rglob('gendaymtx.exe')):
        RAD_DIR.mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(zpath).extractall(RAD_DIR)
    bin_dir = next(RAD_DIR.rglob('gendaymtx.exe')).parent
    missing = [e for e in RAD_EXES if not (bin_dir / e).exists()]
    assert not missing, f'Radiance bin {bin_dir} lacks {missing}'
    root = bin_dir.parent
    (DL / 'radiance_path.txt').write_text(str(root) + '\n')
    # rcontrib -version must say 5.4 (rad5R4); 6.x adds NAVERAGED= to the gendaymtx header -> ladybug misparses.
    ver = subprocess.run([str(bin_dir / 'rcontrib.exe'), '-version'], capture_output=True, text=True, check=False).stdout
    assert '5.4' in ver, f'rcontrib -version: {ver!r}'
    tmp = DL / 'tmp'
    tmp.mkdir(exist_ok=True)
    wea = tmp / 'tiny.wea'
    wea.write_text('place tiny\nlatitude 42.35\nlongitude 71.07\ntime_zone 75\nsite_elevation 0\n'
                   'weather_data_file_units 1\n6 21 11.500 800 100\n6 21 12.500 850 110\n')
    out = subprocess.run([str(bin_dir / 'gendaymtx.exe'), '-m', '1', '-O1', '-A', str(wea)],
                         capture_output=True, text=True, check=True).stdout
    lines = out.split('\n')
    assert 'NAVERAGED' not in out, 'gendaymtx prints NAVERAGED=: not rad5R4'
    assert lines[7] == '' and lines[8] != '', 'expected the 8-line gendaymtx header (7 lines + blank)'
    assert len(lines) == 8 + 146 + 1, f'{len(lines)} lines for -m 1 -A'
    from ladybug_radiance.config import folders   # in-process check that ladybug accepts the folder
    folders.radiance_path = str(root)
    assert folders.radbin_path and Path(folders.radbin_path).samefile(bin_dir)
    print(f'  Radiance {ver.strip()} at {root}')


def leap_epw(src, dst):
    """Feb 28 rows duplicated as Feb 29; header leap flag Yes; 8784 rows."""
    lines = src.read_text().splitlines()
    head, rows = lines[:8], lines[8:]
    assert len(rows) == 8760, len(rows)
    hol = head[4].split(',')
    assert hol[0] == 'HOLIDAYS/DAYLIGHT SAVINGS', head[4]
    hol[1] = 'Yes'
    head[4] = ','.join(hol)
    md = [tuple(r.split(',')[1:3]) for r in rows]
    feb29 = [r.replace(',2,28,', ',2,29,', 1) for r, d in zip(rows, md) if d == ('2', '28')]
    i = md.index(('3', '1'))
    rows = rows[:i] + feb29 + rows[i:]
    assert len(rows) == 8784 and len(feb29) == 24
    dst.write_text('\n'.join(head + rows) + '\n')


def missing_epw(src, dst):
    """Rows 4000..4005 (mid June) get 9999 in GHI/DNI/DHI (EPW columns 14-16, 1-based)."""
    lines = src.read_text().splitlines()
    for k in range(4000, 4006):
        f = lines[8 + k].split(',')
        f[13] = f[14] = f[15] = '9999'
        lines[8 + k] = ','.join(f)
    dst.write_text('\n'.join(lines) + '\n')


def setup_epws(shas):
    EPW_DIR.mkdir(parents=True, exist_ok=True)
    for url in EPW_URLS:
        dest = DL / Path(url).name
        download(url, dest)
        shas[dest.name] = sha256(dest)
        if dest.suffix == '.zip':
            with zipfile.ZipFile(dest) as z:
                members = [m for m in z.namelist() if m.lower().endswith('.epw')]
                assert len(members) == 1, members
                target = EPW_DIR / Path(members[0]).name
                if not target.exists():
                    target.write_bytes(z.read(members[0]))
        else:
            shutil.copyfile(dest, EPW_DIR / dest.name)
    for p in LOCAL_EPWS:
        lines = Path(p).read_text().splitlines()
        if len(lines[1].split(',')) < 60:   # Ottawa CWEC: old short DESIGN CONDITIONS line crashes ladybug's parser
            lines[1] = 'DESIGN CONDITIONS,0'
        (EPW_DIR / Path(p).name).write_text('\n'.join(lines) + '\n')
    leap_epw(EPW_DIR / CHICAGO, EPW_DIR / 'chicago_leap.epw')
    missing_epw(EPW_DIR / BOSTON, EPW_DIR / 'missing.epw')
    from ladybug.wea import Wea
    for p in sorted(EPW_DIR.glob('*.epw')):
        w = Wea.from_epw_file(str(p))
        n = len(w.direct_normal_irradiance)
        assert n == (8784 if p.name == 'chicago_leap.epw' else 8760), (p.name, n)
        print(f'  {p.name}: {n} rows, lat {w.location.latitude} lon {w.location.longitude} tz {w.location.time_zone}')


def main():
    DL.mkdir(exist_ok=True)
    shas = {}
    print('Radiance')
    setup_radiance(shas)
    print('EPW')
    setup_epws(shas)
    (DL / 'sha256.txt').write_text(''.join(f'{v}  {k}\n' for k, v in sorted(shas.items())))
    print((DL / 'sha256.txt').read_text())


if __name__ == '__main__':
    sys.exit(main())
