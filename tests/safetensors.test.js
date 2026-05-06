import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  parseSafetensorsHeader,
  readSafetensorsHeaderLength,
  SAFETENSORS_DTYPES,
} from '../src/parsers/safetensors.js';

/** Build a minimal safetensors buffer with the given JSON header. */
function makeBuffer(headerObj, dataPad = 0) {
  const headerBytes = new TextEncoder().encode(JSON.stringify(headerObj));
  const buf = new ArrayBuffer(8 + headerBytes.length + dataPad);
  const view = new DataView(buf);
  // 8-byte LE header length
  view.setUint32(0, headerBytes.length & 0xFFFFFFFF, true);
  view.setUint32(4, Math.floor(headerBytes.length / 0x100000000), true);
  new Uint8Array(buf, 8, headerBytes.length).set(headerBytes);
  return buf;
}

describe('readSafetensorsHeaderLength', () => {
  it('reads a small LE u64 length', () => {
    const buf = new ArrayBuffer(8);
    new DataView(buf).setUint32(0, 1234, true);
    assert.equal(readSafetensorsHeaderLength(buf), 1234);
  });

  it('handles header lengths above 2^32', () => {
    const buf = new ArrayBuffer(8);
    const view = new DataView(buf);
    view.setUint32(0, 0, true);
    view.setUint32(4, 1, true); // hi=1 → 2^32
    assert.equal(readSafetensorsHeaderLength(buf), 0x100000000);
  });

  it('throws on a too-small buffer', () => {
    assert.throws(() => readSafetensorsHeaderLength(new ArrayBuffer(4)), /too small/i);
  });

  it('accepts a Uint8Array view', () => {
    const buf = new ArrayBuffer(16);
    new DataView(buf).setUint32(0, 42, true);
    assert.equal(readSafetensorsHeaderLength(new Uint8Array(buf, 0, 8)), 42);
  });
});

describe('parseSafetensorsHeader', () => {
  it('parses a single-tensor header', () => {
    const buf = makeBuffer({
      'embed.weight': { dtype: 'F32', shape: [4, 8], data_offsets: [0, 128] },
    });
    const result = parseSafetensorsHeader(buf);
    assert.equal(result.tensors.length, 1);
    const t = result.tensors[0];
    assert.equal(t.name, 'embed.weight');
    assert.equal(t.dtype, 'F32');
    assert.deepEqual(t.shape, [4, 8]);
    assert.deepEqual(t.dataOffsets, [0, 128]);
    assert.equal(t.numElements, 32);
    assert.equal(t.byteLength, 128);
    assert.equal(t.ggmlType, 0);
  });

  it('extracts __metadata__ separately from tensors', () => {
    const buf = makeBuffer({
      __metadata__: { format: 'pt', author: 'me' },
      'a.weight': { dtype: 'F16', shape: [2], data_offsets: [0, 4] },
    });
    const result = parseSafetensorsHeader(buf);
    assert.deepEqual(result.metadata, { format: 'pt', author: 'me' });
    assert.equal(result.tensors.length, 1);
    assert.equal(result.tensors[0].name, 'a.weight');
  });

  it('handles multiple tensors with different dtypes', () => {
    const buf = makeBuffer({
      't1': { dtype: 'BF16', shape: [10], data_offsets: [0, 20] },
      't2': { dtype: 'I8',   shape: [5],  data_offsets: [20, 25] },
      't3': { dtype: 'F32',  shape: [3],  data_offsets: [32, 44] }, // padded
    });
    const result = parseSafetensorsHeader(buf);
    assert.equal(result.tensors.length, 3);
    assert.equal(result.tensors.find(t => t.name === 't1').ggmlType, 30);
    assert.equal(result.tensors.find(t => t.name === 't2').ggmlType, 24);
    assert.equal(result.tensors.find(t => t.name === 't3').ggmlType, 0);
  });

  it('marks unknown dtypes with ggmlType: null', () => {
    const buf = makeBuffer({
      't': { dtype: 'WEIRD', shape: [1], data_offsets: [0, 4] },
    });
    const result = parseSafetensorsHeader(buf);
    assert.equal(result.tensors[0].ggmlType, null);
  });

  it('reports dataStart as 8 + headerLength', () => {
    const buf = makeBuffer({ x: { dtype: 'F32', shape: [1], data_offsets: [0, 4] } }, 16);
    const result = parseSafetensorsHeader(buf);
    assert.equal(result.dataStart, 8 + result.headerLength);
  });

  it('skips entries that are missing required fields', () => {
    const buf = makeBuffer({
      good: { dtype: 'F32', shape: [1], data_offsets: [0, 4] },
      bad1: { dtype: 'F32', shape: [1] },         // missing offsets
      bad2: { shape: [1], data_offsets: [0, 4] }, // missing dtype
      bad3: 'not an object',
    });
    const result = parseSafetensorsHeader(buf);
    assert.equal(result.tensors.length, 1);
    assert.equal(result.tensors[0].name, 'good');
  });

  it('throws on a truncated buffer', () => {
    const fullBuf = makeBuffer({ x: { dtype: 'F32', shape: [1], data_offsets: [0, 4] } });
    const truncated = fullBuf.slice(0, 12); // only 4 bytes of header JSON
    assert.throws(() => parseSafetensorsHeader(truncated), /truncated/i);
  });

  it('throws on malformed JSON', () => {
    const headerBytes = new TextEncoder().encode('{ not valid json');
    const buf = new ArrayBuffer(8 + headerBytes.length);
    new DataView(buf).setUint32(0, headerBytes.length, true);
    new Uint8Array(buf, 8).set(headerBytes);
    assert.throws(() => parseSafetensorsHeader(buf), /malformed/i);
  });

  it('throws on negative or zero header length', () => {
    const buf = new ArrayBuffer(16);
    new DataView(buf).setUint32(0, 0, true);
    assert.throws(() => parseSafetensorsHeader(buf), /Invalid.*header length/i);
  });
});

describe('SAFETENSORS_DTYPES', () => {
  it('covers standard floats and ints with correct byte sizes', () => {
    assert.equal(SAFETENSORS_DTYPES.F32.bytesPerElement, 4);
    assert.equal(SAFETENSORS_DTYPES.F16.bytesPerElement, 2);
    assert.equal(SAFETENSORS_DTYPES.BF16.bytesPerElement, 2);
    assert.equal(SAFETENSORS_DTYPES.F64.bytesPerElement, 8);
    assert.equal(SAFETENSORS_DTYPES.I8.bytesPerElement, 1);
    assert.equal(SAFETENSORS_DTYPES.I64.bytesPerElement, 8);
  });

  it('covers unsigned integer types', () => {
    assert.equal(SAFETENSORS_DTYPES.U8.bytesPerElement, 1);
    assert.equal(SAFETENSORS_DTYPES.U16.bytesPerElement, 2);
    assert.equal(SAFETENSORS_DTYPES.U32.bytesPerElement, 4);
    assert.equal(SAFETENSORS_DTYPES.U64.bytesPerElement, 8);
  });

  it('covers BOOL as a 1-byte type', () => {
    assert.equal(SAFETENSORS_DTYPES.BOOL.bytesPerElement, 1);
  });

  it('covers FP8 variants as 1-byte types', () => {
    assert.equal(SAFETENSORS_DTYPES.F8_E4M3.bytesPerElement, 1);
    assert.equal(SAFETENSORS_DTYPES.F8_E5M2.bytesPerElement, 1);
  });
});

describe('parseSafetensorsHeader — edge cases', () => {
  it('returns no tensors for a header containing only __metadata__', () => {
    const buf = makeBuffer({ __metadata__: { format: 'pt' } });
    const result = parseSafetensorsHeader(buf);
    assert.equal(result.tensors.length, 0);
    assert.deepEqual(result.metadata, { format: 'pt' });
  });

  it('handles a scalar tensor (empty shape)', () => {
    const buf = makeBuffer({
      scalar: { dtype: 'F32', shape: [], data_offsets: [0, 4] },
    });
    const result = parseSafetensorsHeader(buf);
    assert.equal(result.tensors.length, 1);
    assert.equal(result.tensors[0].numElements, 1);
    assert.equal(result.tensors[0].byteLength, 4);
    assert.deepEqual(result.tensors[0].shape, []);
  });

  it('handles a tensor with a zero-length dimension', () => {
    const buf = makeBuffer({
      empty: { dtype: 'F16', shape: [0, 16], data_offsets: [0, 0] },
    });
    const result = parseSafetensorsHeader(buf);
    assert.equal(result.tensors[0].numElements, 0);
    assert.equal(result.tensors[0].byteLength, 0);
  });

  it('handles FP8 dtype with a known ggmlType', () => {
    const buf = makeBuffer({
      x: { dtype: 'F8_E4M3', shape: [16], data_offsets: [0, 16] },
    });
    const result = parseSafetensorsHeader(buf);
    assert.equal(result.tensors[0].ggmlType, 24);
    assert.equal(result.tensors[0].byteLength, 16);
  });

  it('returns an empty metadata object when __metadata__ is absent', () => {
    const buf = makeBuffer({ a: { dtype: 'F32', shape: [1], data_offsets: [0, 4] } });
    const result = parseSafetensorsHeader(buf);
    assert.deepEqual(result.metadata, {});
  });

  it('preserves shape ordering for multi-dimensional tensors', () => {
    const buf = makeBuffer({
      m: { dtype: 'F16', shape: [2, 3, 5, 7], data_offsets: [0, 420] },
    });
    const result = parseSafetensorsHeader(buf);
    assert.deepEqual(result.tensors[0].shape, [2, 3, 5, 7]);
    assert.equal(result.tensors[0].numElements, 210);
  });
});
