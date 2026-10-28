// SPDX-License-Identifier: Apache-2.0 AND LicenseRef-Commons-Clause-1.0
// daylight-mesh: three.js / glTF meshes -> .dmesh / OBJ for the daylight core CLI, plus a Node runner.
// Core contract: world metres, Z-up, right-handed winding; .dmesh = one mesh per file, OBJ = one `o` group per mesh.
import { NodeIO, Primitive } from '@gltf-transform/core';
import type { Mesh, Object3D } from 'three';

export interface ObjMesh {
  name: string;
  positions: ArrayLike<number>; // xyz interleaved, world metres, Z-up
  indices: ArrayLike<number>; // triangle indices, 0-based
  slots?: Uint16Array; // optics slot per triangle (core Mesh::slot; toDmesh writes it, toObj does not); absent = all slot 0
  id?: string; // stable identity for runRemote(): the server caches the mesh under it, so a re-run sends no bytes
}

/**
 * `.dmesh` file stems (= core mesh names) for `names`: sanitised to `[A-Za-z0-9._-]`, empty -> `fallback`, made unique
 * case-insensitively (`_2`, `_3`, ... suffixes; Windows/macOS file systems ignore case). Shared by run() and runRemote().
 */
export function dmeshStems(names: ArrayLike<string>, fallback = 'mesh'): string[] {
  const used = new Set<string>(), out: string[] = [];
  for (let k = 0; k < names.length; k++) {
    const base = names[k].replace(/[^A-Za-z0-9._-]/g, '_') || fallback;
    let stem = base;
    for (let i = 2; used.has(stem.toLowerCase()); i++) stem = `${base}_${i}`;
    used.add(stem.toLowerCase());
    out.push(stem);
  }
  return out;
}

export function toObj(meshes: ObjMesh[]): string {
  let out = '';
  let base = 0;
  for (const m of meshes) {
    const nv = m.positions.length / 3;
    if (!Number.isInteger(nv) || m.indices.length % 3 !== 0) throw new Error(`${m.name}: positions must be xyz triples, indices triangle triples`);
    out += `o ${m.name}\n`;
    for (let i = 0; i < m.positions.length; i += 3) out += `v ${m.positions[i]} ${m.positions[i + 1]} ${m.positions[i + 2]}\n`;
    for (let i = 0; i < m.indices.length; i += 3) {
      const a = m.indices[i], b = m.indices[i + 1], c = m.indices[i + 2];
      if (!(a >= 0 && a < nv && b >= 0 && b < nv && c >= 0 && c < nv)) throw new Error(`${m.name}: index out of range in face ${i / 3}`);
      out += `f ${base + a + 1} ${base + b + 1} ${base + c + 1}\n`;
    }
    base += nv;
  }
  return out;
}

/**
 * One mesh -> `.dmesh` bytes (little-endian): "DLM1", uint32 flags, uint64 nPoints/nFaces/nIndices,
 * float64 xyz[3*nPoints], int32 counts[nFaces] (all 3), int32 indices[nIndices], then with `slots` (flags 1, else 0)
 * uint16 slot[nFaces]. Mesh name = file stem.
 */
export function toDmesh(mesh: ObjMesh): Uint8Array<ArrayBuffer> {
  const N = mesh.positions.length / 3, F = mesh.indices.length / 3, M = mesh.indices.length, S = mesh.slots;
  if (!Number.isInteger(N) || !Number.isInteger(F)) throw new Error(`${mesh.name}: positions must be xyz triples, indices triangle triples`);
  if (S && S.length !== F) throw new Error(`${mesh.name}: slots must hold one value per triangle (${F}), not ${S.length}`);
  const buf = new ArrayBuffer(32 + 24 * N + 4 * F + 4 * M + (S ? 2 * F : 0));
  const dv = new DataView(buf);
  new Uint8Array(buf, 0, 4).set([0x44, 0x4c, 0x4d, 0x31]); // "DLM1"
  dv.setUint32(4, S ? 1 : 0, true); // flags: bit 0 = slot block
  dv.setBigUint64(8, BigInt(N), true);
  dv.setBigUint64(16, BigInt(F), true);
  dv.setBigUint64(24, BigInt(M), true);
  new Float64Array(buf, 32, 3 * N).set(mesh.positions);
  new Int32Array(buf, 32 + 24 * N, F).fill(3);
  const idx = new Int32Array(buf, 32 + 24 * N + 4 * F, M);
  for (let i = 0; i < M; i++) {
    const v = mesh.indices[i];
    if (!(Number.isInteger(v) && v >= 0 && v < N)) throw new Error(`${mesh.name}: index out of range in face ${Math.floor(i / 3)}`);
    idx[i] = v;
  }
  if (S) new Uint16Array(buf, 32 + 24 * N + 4 * F + 4 * M, F).set(S);
  return new Uint8Array(buf);
}

// Column-major 4x4 affine `m` (three.js and glTF agree) -> world, * scale, Y-up -> Z-up as (x, -z, y);
// winding reversed when det(m) < 0 (mirrored transform) so the OBJ stays right-handed.
function toMesh(name: string, pos: ArrayLike<number>, idx: ArrayLike<number> | null, m: ArrayLike<number>, scale: number): ObjMesh {
  const positions = new Float64Array(pos.length);
  for (let i = 0; i < pos.length; i += 3) {
    const x = pos[i], y = pos[i + 1], z = pos[i + 2];
    const wx = m[0] * x + m[4] * y + m[8] * z + m[12];
    const wy = m[1] * x + m[5] * y + m[9] * z + m[13];
    const wz = m[2] * x + m[6] * y + m[10] * z + m[14];
    positions[i] = wx * scale;
    positions[i + 1] = -wz * scale;
    positions[i + 2] = wy * scale;
  }
  const indices = idx ? Uint32Array.from(idx) : Uint32Array.from({ length: pos.length / 3 }, (_, i) => i);
  const det = m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2]) + m[8] * (m[1] * m[6] - m[5] * m[2]);
  if (det < 0) for (let i = 0; i < indices.length; i += 3) { const t = indices[i + 1]; indices[i + 1] = indices[i + 2]; indices[i + 2] = t; }
  return { name, positions, indices };
}

/** Every descendant Mesh of `objects` (triangle BufferGeometry only; strips/fans/lines are not supported). */
export function fromThree(objects: Object3D[], opts: { scale?: number } = {}): ObjMesh[] {
  const out: ObjMesh[] = [];
  for (const root of objects) {
    root.updateMatrixWorld(true);
    root.traverse((o) => {
      const mesh = o as Mesh;
      if (!mesh.isMesh) return;
      const pos = mesh.geometry.attributes.position;
      if (!pos || pos.itemSize !== 3) throw new Error(`${o.name || o.uuid}: geometry has no xyz position attribute`);
      out.push(toMesh(o.name || o.uuid, pos.array, mesh.geometry.index?.array ?? null, mesh.matrixWorld.elements, opts.scale ?? 1));
    });
  }
  return out;
}

/** All TRIANGLES primitives of every node with a mesh in a .glb; name = node name + '/' + mesh name (or primitive index). */
export async function fromGLB(data: ArrayBuffer | Uint8Array, opts: { scale?: number } = {}): Promise<ObjMesh[]> {
  const doc = await new NodeIO().readBinary(new Uint8Array(data));
  const out: ObjMesh[] = [];
  for (const node of doc.getRoot().listNodes()) {
    const mesh = node.getMesh();
    if (!mesh) continue;
    const m = node.getWorldMatrix();
    mesh.listPrimitives().forEach((prim, i) => {
      if (prim.getMode() !== Primitive.Mode.TRIANGLES) return;
      const pos = prim.getAttribute('POSITION')?.getArray();
      if (!pos) return;
      const name = `${node.getName()}/${mesh.getName() || i}`;
      out.push(toMesh(name, pos, prim.getIndices()?.getArray() ?? null, m, opts.scale ?? 1));
    });
  }
  return out;
}

/** Core `--out FILE` text file -> one Float64Array of columns per element (blank lines skipped). */
export function parseValues(text: string): Float64Array[] {
  return text.split('\n').filter((l) => l.trim()).map((l) => Float64Array.from(l.trim().split(/\s+/), Number));
}

/** Core `--out FILE.npy` (NumPy v1.0, `<f8`, C order [N x cols]) -> one Float64Array per row. */
export function parseNpy(bytes: Uint8Array, cols: number): Float64Array[] {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (dv.getUint8(0) !== 0x93 || dv.getUint8(6) !== 1) throw new Error('not a NumPy v1.0 .npy file');
  if (!(cols >= 1)) throw new RangeError('cols must be >= 1');
  const start = 10 + dv.getUint16(8, true);
  const data = new Float64Array(Uint8Array.prototype.slice.call(bytes, start).buffer); // real copy (Buffer.slice is a view): 8-byte aligned, own buffer
  const rows: Float64Array[] = [];
  for (let i = 0; i + cols <= data.length; i += cols) rows.push(data.subarray(i, i + cols));
  return rows;
}

// viridis at t = 0, 0.125, ..., 1 (matplotlib); shared with bindings/python.
const VIRIDIS = [[0.267004, 0.004874, 0.329415], [0.278826, 0.175490, 0.483397], [0.229739, 0.322361, 0.545706], [0.172719, 0.448791, 0.557885], [0.127568, 0.566949, 0.550556], [0.157851, 0.683765, 0.501686], [0.369214, 0.788888, 0.382914], [0.678489, 0.863742, 0.189503], [0.993248, 0.906157, 0.143936]];

/**
 * Values -> RGB triples (Float32Array, 3 per value) over [vmin, vmax] (default: finite min/max); non-finite -> grey.
 * `stops`: evenly spaced RGB stops (0..1) of any other gradient, default viridis.
 */
export function colormap(values: ArrayLike<number>, opts: { vmin?: number; vmax?: number; stops?: ArrayLike<ArrayLike<number>> } = {}): Float32Array {
  const stops = opts.stops ?? VIRIDIS, n = stops.length - 1;
  if (n < 1) throw new RangeError('stops needs at least two colours');
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < values.length; i++) { const v = values[i]; if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
  const vmin = opts.vmin ?? lo, span = (opts.vmax ?? hi) - vmin;
  const rgb = new Float32Array(3 * values.length);
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) { rgb.fill(0.5, 3 * i, 3 * i + 3); continue; }
    const t = span ? Math.min(Math.max((v - vmin) / span, 0), 1) : 0;
    const k = Math.min(Math.floor(t * n), n - 1), f = t * n - k, a = stops[k], b = stops[k + 1];
    for (let c = 0; c < 3; c++) rgb[3 * i + c] = a[c] + f * (b[c] - a[c]);
  }
  return rgb;
}

/**
 * Colours `mesh` by `values`: one per triangle (geometry becomes non-indexed so each face gets a flat colour) or one per
 * vertex. Sets the `color` attribute and, unless `material: false`, a `MeshStandardMaterial({ vertexColors: true })`.
 * Colours come from colormap() (`vmin`, `vmax`, `stops`) or, with `rgb` (3 floats per value), from the caller.
 * Imports three lazily (peer dependency). Returns the RGB attribute array.
 */
export async function applyToThree(mesh: Mesh, values: ArrayLike<number>, opts: { vmin?: number; vmax?: number; stops?: ArrayLike<ArrayLike<number>>; rgb?: Float32Array; material?: boolean } = {}): Promise<Float32Array> {
  const { BufferAttribute, MeshStandardMaterial } = await import('three');
  let geometry = mesh.geometry;
  const triangles = (geometry.index ? geometry.index.count : geometry.attributes.position.count) / 3;
  const perFace = values.length === triangles;
  if (!perFace && values.length !== geometry.attributes.position.count) throw new Error('values.length must be the triangle count (N) or the vertex count (M)');
  if (opts.rgb && opts.rgb.length !== 3 * values.length) throw new Error('rgb must hold 3 floats per value');
  let rgb = opts.rgb ?? colormap(values, opts);
  if (perFace) {
    if (geometry.index) geometry = mesh.geometry = geometry.toNonIndexed();
    const face = rgb;
    rgb = new Float32Array(9 * triangles);
    for (let i = 0; i < rgb.length; i++) rgb[i] = face[3 * Math.floor(i / 9) + (i % 3)];
  }
  geometry.setAttribute('color', new BufferAttribute(rgb, 3));
  if (opts.material !== false) mesh.material = new MeshStandardMaterial({ vertexColors: true });
  return rgb;
}

/**
 * Node only: writes the inputs into a temp dir, runs `exe <sensor files> [--context <file>]... ...args --out out.npy`,
 * returns the values and the `out.npy.json` manifest. A string input is OBJ text (one .obj); an ObjMesh[] becomes
 * one `<name>.dmesh` per mesh via toDmesh() (name sanitised to [A-Za-z0-9._-], made unique).
 */
export async function run(exe: string, input: { sensors: string | ObjMesh[]; context?: string | ObjMesh[]; args?: string[] }): Promise<{ values: Float64Array[]; columns: string[]; manifest: unknown }> {
  const fs = await import('node:fs/promises'); // dynamic so the browser-safe functions above bundle without node builtins
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const { execFile } = await import('node:child_process');
  const dir = await fs.mkdtemp(join(tmpdir(), 'daylight-'));
  try {
    const sens = typeof input.sensors === 'string' ? [] : input.sensors;
    const ctx = !input.context || typeof input.context === 'string' ? [] : input.context;
    const stems = dmeshStems([...sens.map((m) => m.name || 'sensors'), ...ctx.map((m) => m.name || 'context')]);
    const write = async (kind: string, m: string | ObjMesh[], first: number) => {
      if (typeof m === 'string') { const p = join(dir, kind + '.obj'); await fs.writeFile(p, m); return [p]; }
      const paths: string[] = [];
      for (let k = 0; k < m.length; k++) {
        paths.push(join(dir, stems[first + k] + '.dmesh'));
        await fs.writeFile(paths[paths.length - 1], toDmesh(m[k]));
      }
      return paths;
    };
    const out = join(dir, 'out.npy');
    const args = await write('sensors', input.sensors, 0);
    if (input.context) for (const p of await write('context', input.context, sens.length)) args.push('--context', p);
    args.push(...(input.args ?? []), '--out', out);
    await new Promise<void>((res, rej) => execFile(exe, args, { maxBuffer: 1 << 26 }, (e, _o, stderr) => (e ? rej(new Error(`${exe} failed: ${stderr.trim() || e.message}`)) : res())));
    const manifest = JSON.parse(await fs.readFile(out + '.json', 'utf8')) as { columns: string[] };
    return { values: parseNpy(await fs.readFile(out), manifest.columns.length), columns: manifest.columns, manifest };
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
}

/** The core's `FILE.json` manifest as `daylight_server` returns it (plus the server's own fields). */
export interface RemoteManifest {
  columns: string[];
  duration_h: number;
  dirs: { sky: number; ground: number; sun: number };
  meshes: { name: string; first: number; count: number }[];
  run_s?: number;
  cache?: Record<string, string>;
}

/** A job on `daylight_server` (`/jobs/{id}`): position counts the jobs ahead of a queued one. */
export interface RemoteJob {
  id: string;
  state: 'queued' | 'running' | 'done' | 'error' | 'cancelled';
  position?: number;
  error?: string;  // state error / cancelled: the server's message
  status?: number; // state error: the HTTP status POST /run would have answered (400 usage, 500 runtime)
}

/** Mesh ids this process has put on each server: a job names them by id alone; any other is put first. */
const uploaded = new Map<string, Set<string>>();

/**
 * Browser or Node: runs the core on a `daylight_server` as a queued job, so several users share the GPU. Meshes travel
 * by id (`ObjMesh.id`; a UUID is minted here when absent): `POST <url>/jobs` names them. A mesh this process has not
 * sent to that server yet goes up first, once, via `PUT <url>/mesh/<id>?name=<stem>`, so a re-run with other settings
 * sends no geometry; when the server answers 404 `{missing}` anyway (restarted, or it evicted the mesh) those go up
 * again before one retry. The job is then long-polled (`GET /jobs/<id>?wait=25`); `onStatus` sees every
 * queued / running state. Aborting `signal` cancels the job on the server (`DELETE /jobs/<id>`), also one aborted
 * while the server was making it (the POST itself is not aborted, so its id is always known). `epw` is a Blob
 * (uploaded) or `{ url }` (the server fetches it through its allowlisted proxy cache). `args` are CLI flags verbatim
 * (no files). `clientId` (header `X-Client-Id`) identifies this client for the per-client job limit. Rejects with the
 * server's text on 4xx / 5xx or a failed job. One Float64Array per element.
 */
export async function runRemote(
  url: string,
  input: {
    sensors: ObjMesh[]; context?: ObjMesh[]; epw?: Blob | { url: string }; args?: string[];
    signal?: AbortSignal; clientId?: string; onStatus?: (job: RemoteJob) => void; fetch?: typeof globalThis.fetch;
  },
): Promise<{ values: Float64Array[]; columns: string[]; manifest: RemoteManifest; cache?: Record<string, string> }> {
  const f = input.fetch ?? globalThis.fetch;
  const base = url.replace(/\/$/, '');
  const headers: Record<string, string> = input.clientId ? { 'X-Client-Id': input.clientId } : {};
  const req = (path: string, init: RequestInit = {}) => f(`${base}${path}`, { ...init, headers, signal: input.signal });
  const meshes = [...input.sensors, ...(input.context ?? [])];
  for (const m of meshes) m.id ??= crypto.randomUUID();
  const stems = dmeshStems(meshes.map((m) => m.name));
  const sent = uploaded.get(base) ?? uploaded.set(base, new Set()).get(base)!;
  const put = (ids: string[]) => Promise.all([...new Set(ids)].map(async (id) => {
    const k = meshes.findIndex((m) => m.id === id);
    if (k < 0) throw new Error(`${base}/jobs asked for an unknown mesh id ${id}`);
    const res = await req(`/mesh/${encodeURIComponent(id)}?name=${encodeURIComponent(stems[k])}`, { method: 'PUT', body: toDmesh(meshes[k]) });
    if (!res.ok) throw new Error(`PUT ${base}/mesh/${id} failed (${res.status}): ${(await res.text()).trim()}`);
    sent.add(id);
  }));
  const body = () => {
    const fd = new FormData();
    for (const m of input.sensors) fd.append('sensors', m.id!);
    for (const m of input.context ?? []) fd.append('context', m.id!);
    if (input.epw instanceof Blob) fd.append('epw', input.epw, 'weather.epw');
    else if (input.epw) fd.append('epw_url', input.epw.url);
    for (const a of input.args ?? []) fd.append('args', a);
    return fd;
  };
  await put(meshes.map((m) => m.id!).filter((id) => !sent.has(id))); // new meshes first: no 404 round trip
  // Not aborted: once the server has made the job its id is known, so an abort meanwhile still deletes it below.
  const post = () => f(`${base}/jobs`, { method: 'POST', body: body(), headers });
  let res = await post();
  if (res.status === 404) { // the server lost some (restart, eviction): put them again, retry once
    const { missing } = (await res.json()) as { missing: string[] };
    for (const id of missing) sent.delete(id);
    await put(missing);
    res = await post();
  }
  if (!res.ok) throw new Error(`${base}/jobs failed (${res.status}): ${(await res.text()).trim()}`);
  let job = (await res.json()) as RemoteJob & Partial<RemoteManifest> & { values_b64?: string };
  try {
    input.signal?.throwIfAborted(); // aborted while the job was being made
    while (job.state === 'queued' || job.state === 'running') {
      input.onStatus?.(job);
      const poll = await req(`/jobs/${encodeURIComponent(job.id)}?wait=25`);
      if (!poll.ok) throw new Error(`${base}/jobs/${job.id} failed (${poll.status}): ${(await poll.text()).trim()}`);
      job = await poll.json();
    }
  } catch (e) {
    if (input.signal?.aborted) f(`${base}/jobs/${encodeURIComponent(job.id)}`, { method: 'DELETE', headers }).catch(() => undefined); // free the GPU for others
    throw e;
  }
  if (job.state !== 'done') throw new Error(`${base} job ${job.id} ${job.state}${job.status ? ` (${job.status})` : ''}: ${(job.error ?? '').trim()}`);
  const m = job as RemoteJob & RemoteManifest & { values_b64: string };
  const bin = atob(m.values_b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const data = new Float64Array(bytes.buffer), cols = m.columns.length, values: Float64Array[] = [];
  for (let i = 0; cols && i + cols <= data.length; i += cols) values.push(data.subarray(i, i + cols));
  return { values, columns: m.columns, manifest: m, cache: m.cache };
}
