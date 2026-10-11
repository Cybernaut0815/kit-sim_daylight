// SPDX-License-Identifier: MIT
import { existsSync } from 'node:fs';
import { Document, NodeIO } from '@gltf-transform/core';
import { BoxGeometry, Mesh } from 'three';
import { describe, expect, it } from 'vitest';
import { applyToThree, colormap, fromGLB, fromThree, parseNpy, parseValues, run, toDmesh, toObj } from '../src/index.js';

const box = (x = 1) => { const m = new Mesh(new BoxGeometry(1, 1, 1)); m.position.set(0, 2, 0); m.scale.x = x; m.name = 'box'; return m; };

describe('fromThree', () => {
  it('converts a translated BoxGeometry to Z-up', () => {
    const [m] = fromThree([box()]);
    expect(m.name).toBe('box');
    expect(m.positions.length).toBe(24 * 3);
    expect(m.indices.length).toBe(12 * 3);
    const p = m.positions, g = box().geometry.attributes.position; // three (x, y=2.5, z) -> core (x, -z, 2.5)
    expect([p[0], p[1], p[2]]).toEqual([g.getX(0), -g.getZ(0), g.getY(0) + 2]);
    expect(p[2]).toBe(2.5);
  });
  it('reverses winding for a mirrored transform', () => {
    const [a] = fromThree([box()]), [b] = fromThree([box(-1)]);
    expect([b.indices[0], b.indices[1], b.indices[2]]).toEqual([a.indices[0], a.indices[2], a.indices[1]]);
    expect(b.positions[0]).toBe(-a.positions[0]);
  });
});

describe('toObj', () => {
  it('writes one group per mesh with global 1-based indices', () => {
    const tri = { name: 't', positions: [0, 0, 0, 1, 0, 0, 0, 1, 0], indices: [0, 1, 2] };
    const obj = toObj([tri, { ...tri, name: 'u' }]);
    const lines = obj.trim().split('\n');
    expect(lines[0]).toBe('o t');
    expect(lines.filter((l) => l.startsWith('v ')).length).toBe(6);
    expect(lines.filter((l) => l.startsWith('f ')).length).toBe(2);
    expect(lines.filter((l) => l.startsWith('f '))).toEqual(['f 1 2 3', 'f 4 5 6']);
    expect(() => toObj([{ ...tri, indices: [0, 1, 3] }])).toThrow(/out of range/);
  });
});

describe('toDmesh', () => {
  it('writes the DLM1 header and arrays little-endian', () => {
    const [m] = fromThree([box()]);
    const bytes = toDmesh(m);
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    expect(bytes.byteLength).toBe(32 + 24 * 24 + 4 * 12 + 4 * 36);
    expect(String.fromCharCode(...bytes.subarray(0, 4))).toBe('DLM1');
    expect(dv.getUint32(4, true)).toBe(0);
    expect([dv.getBigUint64(8, true), dv.getBigUint64(16, true), dv.getBigUint64(24, true)]).toEqual([24n, 12n, 36n]);
    expect([dv.getFloat64(32, true), dv.getFloat64(40, true), dv.getFloat64(48, true)]).toEqual([m.positions[0], m.positions[1], m.positions[2]]);
    expect(dv.getInt32(32 + 24 * 24, true)).toBe(3);
    expect(dv.getInt32(bytes.byteLength - 4, true)).toBe(m.indices[35]);
    expect(() => toDmesh({ name: 't', positions: [0, 0, 0, 1, 0, 0, 0, 1, 0], indices: [0, 1, 3] })).toThrow(/out of range/);
    expect(() => toDmesh({ name: 't', positions: [0, 0, 0, 1, 0, 0, 0, 1, 0], indices: [0, 1, 1.5] })).toThrow(/out of range/);
  });
});

describe('fromGLB', () => {
  it('round-trips a generated glb', async () => {
    const doc = new Document();
    const buf = doc.createBuffer();
    const pos = doc.createAccessor().setType('VEC3').setArray(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0])).setBuffer(buf);
    const idx = doc.createAccessor().setType('SCALAR').setArray(new Uint16Array([0, 1, 2])).setBuffer(buf);
    const mesh = doc.createMesh('tri').addPrimitive(doc.createPrimitive().setAttribute('POSITION', pos).setIndices(idx));
    doc.createScene().addChild(doc.createNode('n').setMesh(mesh).setTranslation([1, 2, 3]));
    const glb = await new NodeIO().writeBinary(doc);
    const meshes = await fromGLB(glb);
    expect(meshes.length).toBe(1);
    expect(meshes[0].name).toBe('n/tri');
    expect(Array.from(meshes[0].positions)).toEqual([1, -3, 2, 2, -3, 2, 1, -3, 3]);
    expect(Array.from(meshes[0].indices)).toEqual([0, 1, 2]);
  });
});

it('parseValues', () => {
  expect(parseValues('1 2.5\n\n 3 4 \n').map((r) => Array.from(r))).toEqual([[1, 2.5], [3, 4]]);
});

it('parseNpy', () => {
  const header = "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 2), }".padEnd(117) + '\n'; // 10 + 118 = 128 bytes (64-aligned like NumPy)
  const buf = new Uint8Array(128 + 32), dv = new DataView(buf.buffer);
  buf.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, 118, 0]); // \x93NUMPY v1.0, header length 118 LE
  buf.set(Array.from(header, (c) => c.charCodeAt(0)), 10);
  [1, 2.5, 3, 4].forEach((v, i) => dv.setFloat64(128 + 8 * i, v, true));
  expect(parseNpy(buf, 2).map((r) => Array.from(r))).toEqual([[1, 2.5], [3, 4]]);
  expect(parseNpy(Buffer.concat([Buffer.alloc(8), Buffer.from(buf)]).subarray(8), 2).map((r) => Array.from(r))).toEqual([[1, 2.5], [3, 4]]); // Buffer view with byteOffset (fs.readFile pool)
  expect(parseNpy(buf.subarray(0, 128 + 24), 2).length).toBe(1); // partial trailing row dropped
  expect(() => parseNpy(new Uint8Array(16), 2)).toThrow(/NumPy/);
  expect(() => parseNpy(buf, 0)).toThrow(/cols/);
});

it('colormap', () => {
  const near = (rgb: Float32Array, i: number, want: number[]) => Array.from(rgb.subarray(3 * i, 3 * i + 3)).forEach((c, k) => expect(c).toBeCloseTo(want[k], 5));
  const rgb = colormap([0, 0.3, 1, NaN], { vmin: 0, vmax: 1 });
  near(rgb, 0, [0.267004, 0.004874, 0.329415]);
  near(rgb, 1, [0.206931, 0.372933, 0.550578]);
  near(rgb, 2, [0.993248, 0.906157, 0.143936]);
  near(rgb, 3, [0.5, 0.5, 0.5]);
  near(colormap([10, 20, NaN]), 1, [0.993248, 0.906157, 0.143936]); // default range = finite min/max
  near(colormap([7, 7]), 1, [0.267004, 0.004874, 0.329415]); // vmin == vmax -> t = 0
});

describe('applyToThree', () => {
  const vals = (n: number) => Float64Array.from({ length: n }, (_, i) => i);
  it('per face: non-indexed, 3 equal colours per triangle', async () => {
    const m = new Mesh(new BoxGeometry());
    const rgb = await applyToThree(m, vals(12));
    expect(m.geometry.index).toBeNull();
    expect(m.geometry.getAttribute('color').count).toBe(36);
    expect(rgb.length).toBe(108);
    expect(Array.from(rgb.subarray(3, 9))).toEqual([rgb[0], rgb[1], rgb[2], rgb[0], rgb[1], rgb[2]]);
    expect((m.material as { vertexColors: boolean }).vertexColors).toBe(true);
  });
  it('per vertex: stays indexed; bad length rejects; material:false keeps the material', async () => {
    const m = new Mesh(new BoxGeometry()), mat = m.material;
    await applyToThree(m, vals(24), { material: false });
    expect(m.geometry.index).not.toBeNull();
    expect(m.geometry.getAttribute('color').count).toBe(24);
    expect(m.material).toBe(mat);
    await expect(applyToThree(new Mesh(new BoxGeometry()), vals(7))).rejects.toThrow(/triangle count/);
  });
});

const exe = process.env.DAYLIGHT_EXE && existsSync(process.env.DAYLIGHT_EXE) ? process.env.DAYLIGHT_EXE : '';
it.skipIf(!exe)('run() on a box through the core via .dmesh in, .npy out (set DAYLIGHT_EXE to enable)', async () => {
  const r = await run(exe, { sensors: fromThree([box()]), args: ['--lat', '42.35', '--lon', '-71.07', '--tz', '-5', '--quantity', 'sunhours', '--time', '06-21T12:00'] });
  expect(r.values.length).toBe(12);
  expect(r.columns).toEqual(['sunhours_front']); // <quantity>_<side>, as the README snippet relies on
  expect(r.values[0].length).toBe(r.columns.length);
  for (const row of r.values) for (const v of row) expect(v).toBeGreaterThanOrEqual(0), expect(v).toBeLessThanOrEqual(1);
});
