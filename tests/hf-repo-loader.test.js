/**
 * Tests for src/hf/repo-loader.js — drive readShardHeader / resolveShards /
 * loadHfRepo with a mocked global fetch so we can exercise sharded and
 * single-file repo flows without touching the network.
 */
import { describe, it, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { readShardHeader, resolveShards, loadHfRepo } from '../src/hf/repo-loader.js';

const REF = { owner: 'o', repo: 'r', revision: 'main' };
const RESOLVE = (file) => `https://huggingface.co/o/r/resolve/main/${file}`;
const PROXY = (file) => `/api/hf-file?url=${encodeURIComponent(RESOLVE(file))}`;

/** Build a safetensors blob: 8-byte LE length + JSON header + N data bytes. */
function buildSafetensors(headerObj, dataBytes = 0) {
  const json = JSON.stringify(headerObj);
  const headerBuf = new TextEncoder().encode(json);
  const total = 8 + headerBuf.byteLength + dataBytes;
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  dv.setBigUint64(0, BigInt(headerBuf.byteLength), true);
  out.set(headerBuf, 8);
  return out;
}

/** Slice a buffer to satisfy a single Range header value (bytes=a-b, inclusive). */
function rangeSlice(buf, range) {
  const m = /bytes=(\d+)-(\d+)/.exec(range || '');
  if (!m) return buf;
  const start = parseInt(m[1], 10);
  const end = parseInt(m[2], 10);
  return buf.subarray(start, end + 1);
}

/** Make a Response-like result. */
function jsonResponse(obj, status = 200) {
  return new Response(JSON.stringify(obj), { status, headers: { 'content-type': 'application/json' } });
}
function bytesResponse(bytes, status = 206) {
  return new Response(bytes, { status });
}
function notFound() { return new Response('not found', { status: 404 }); }

let originalFetch;
beforeEach(() => { originalFetch = globalThis.fetch; });
afterEach(() => { globalThis.fetch = originalFetch; });

describe('readShardHeader', () => {
  it('reads the JSON header in two Range requests and returns parsed tensors', async () => {
    const blob = buildSafetensors({
      'embed.weight': { dtype: 'F32', shape: [4, 8], data_offsets: [0, 128] },
      __metadata__: { format: 'pt' },
    });
    const calls = [];
    globalThis.fetch = async (url, init = {}) => {
      calls.push({ url, range: init.headers?.Range });
      return bytesResponse(rangeSlice(blob, init.headers?.Range));
    };
    const header = await readShardHeader(RESOLVE('model.safetensors'));
    assert.equal(header.tensors.length, 1);
    assert.equal(header.tensors[0].name, 'embed.weight');
    assert.deepEqual(header.tensors[0].shape, [4, 8]);
    assert.equal(calls.length, 2);
    assert.match(calls[0].range, /^bytes=0-7$/);
    assert.match(calls[1].range, /^bytes=0-\d+$/);
  });

  it('throws for an implausibly large header length', async () => {
    const huge = new Uint8Array(8);
    new DataView(huge.buffer).setBigUint64(0, BigInt(512 * 1024 * 1024), true);
    globalThis.fetch = async () => bytesResponse(huge);
    await assert.rejects(() => readShardHeader(RESOLVE('m.safetensors')), /Implausible/);
  });

  it('surfaces a friendly auth error on 401', async () => {
    globalThis.fetch = async () => new Response('no', { status: 401 });
    await assert.rejects(() => readShardHeader(RESOLVE('m.safetensors')), /401/);
  });
});

describe('resolveShards', () => {
  it('returns sorted unique shard list from index weight_map', async () => {
    globalThis.fetch = async (url) => {
      if (url === PROXY('model.safetensors.index.json')) {
        return jsonResponse({ weight_map: {
          'a.weight': 'model-00002-of-00002.safetensors',
          'b.weight': 'model-00001-of-00002.safetensors',
          'c.weight': 'model-00001-of-00002.safetensors',
        }});
      }
      return notFound();
    };
    const { shards } = await resolveShards(REF);
    assert.deepEqual(shards, ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']);
  });

  it('falls back to single model.safetensors when no index.json', async () => {
    globalThis.fetch = async () => notFound();
    const { shards } = await resolveShards(REF);
    assert.deepEqual(shards, ['model.safetensors']);
  });

  it('falls back to single file when index.json has empty weight_map', async () => {
    globalThis.fetch = async (url) => {
      if (url === PROXY('model.safetensors.index.json')) return jsonResponse({ weight_map: {} });
      return notFound();
    };
    const { shards } = await resolveShards(REF);
    assert.deepEqual(shards, ['model.safetensors']);
  });
});

describe('loadHfRepo', () => {
  it('fetches config, resolves shards, and aggregates tensors with shard names', async () => {
    const shardA = buildSafetensors({ 't1': { dtype: 'F16', shape: [2], data_offsets: [0, 4] } });
    const shardB = buildSafetensors({ 't2': { dtype: 'F32', shape: [3], data_offsets: [0, 12] } });
    globalThis.fetch = async (url, init = {}) => {
      if (url === PROXY('config.json')) return jsonResponse({ model_type: 'llama', num_hidden_layers: 1 });
      if (url === PROXY('model.safetensors.index.json')) return jsonResponse({ weight_map: { t1: 'a.safetensors', t2: 'b.safetensors' } });
      if (url === PROXY('a.safetensors')) return bytesResponse(rangeSlice(shardA, init.headers?.Range));
      if (url === PROXY('b.safetensors')) return bytesResponse(rangeSlice(shardB, init.headers?.Range));
      return notFound();
    };
    const progress = [];
    const result = await loadHfRepo(REF, (m) => progress.push(m));
    assert.equal(result.config.model_type, 'llama');
    assert.deepEqual(result.shards, ['a.safetensors', 'b.safetensors']);
    assert.equal(result.tensors.length, 2);
    assert.equal(result.tensors[0].shard, 'a.safetensors');
    assert.equal(result.tensors[1].shard, 'b.safetensors');
    assert.ok(progress.some(m => /config\.json/.test(m)));
    assert.ok(progress.some(m => /shard 1\/2/.test(m)));
  });

  it('rejects when config.json is invalid JSON', async () => {
    globalThis.fetch = async (url) => {
      if (url === PROXY('config.json')) return new Response('not json', { status: 200 });
      return notFound();
    };
    await assert.rejects(() => loadHfRepo(REF), /config\.json is not valid JSON/);
  });

  it('wraps shard read failures with the shard filename', async () => {
    globalThis.fetch = async (url) => {
      if (url === PROXY('config.json')) return jsonResponse({ model_type: 'llama' });
      if (url === PROXY('model.safetensors.index.json')) return notFound();
      if (url === PROXY('model.safetensors')) return new Response('boom', { status: 500 });
      return notFound();
    };
    await assert.rejects(() => loadHfRepo(REF), /Failed to read model\.safetensors/);
  });
});
