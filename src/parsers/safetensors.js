/**
 * Safetensors header parser.
 *
 * On-disk layout:
 *   [ 8 bytes LE u64: header_length ]
 *   [ header_length bytes: JSON: { tensor_name: { dtype, shape, data_offsets: [start,end] }, "__metadata__": {...} } ]
 *   [ tensor data ]
 *
 * Spec: https://github.com/huggingface/safetensors
 *
 * This module only reads the JSON header — no weight data.
 */

/**
 * Map a safetensors dtype string to (a) the GGML type id used elsewhere in
 * the app and (b) the byte size per element.
 *
 * GGML doesn't distinguish signed/unsigned ints; sizes match so quant/memory
 * accounting stays correct. Display name is preserved as the safetensors
 * dtype so the UI shows "U8"/"BOOL" verbatim where applicable.
 */
export const SAFETENSORS_DTYPES = {
  F32:  { ggmlType: 0,  bytesPerElement: 4 },
  F16:  { ggmlType: 1,  bytesPerElement: 2 },
  BF16: { ggmlType: 30, bytesPerElement: 2 },
  F64:  { ggmlType: 28, bytesPerElement: 8 },
  I8:   { ggmlType: 24, bytesPerElement: 1 },
  U8:   { ggmlType: 24, bytesPerElement: 1 },
  BOOL: { ggmlType: 24, bytesPerElement: 1 },
  I16:  { ggmlType: 25, bytesPerElement: 2 },
  U16:  { ggmlType: 25, bytesPerElement: 2 },
  I32:  { ggmlType: 26, bytesPerElement: 4 },
  U32:  { ggmlType: 26, bytesPerElement: 4 },
  I64:  { ggmlType: 27, bytesPerElement: 8 },
  U64:  { ggmlType: 27, bytesPerElement: 8 },
  // FP8 variants — best-effort 1-byte mapping; not all tooling agrees on type id.
  'F8_E4M3': { ggmlType: 24, bytesPerElement: 1 },
  'F8_E5M2': { ggmlType: 24, bytesPerElement: 1 },
};

/**
 * Read the 8-byte little-endian header length from a buffer.
 * @param {ArrayBuffer|Uint8Array} buffer  must contain at least 8 bytes
 * @returns {number} header length in bytes
 */
export function readSafetensorsHeaderLength(buffer) {
  const view = buffer instanceof ArrayBuffer
    ? new DataView(buffer)
    : new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  if (view.byteLength < 8) throw new Error('Buffer too small for safetensors header length');
  const lo = view.getUint32(0, true);
  const hi = view.getUint32(4, true);
  // JS safe-int range easily covers any plausible header length (< 2^53).
  return lo + hi * 0x100000000;
}

/**
 * Parse a safetensors header from a buffer that contains at least the first
 * (8 + header_length) bytes of the file.
 *
 * Returns:
 *   {
 *     headerLength,           // int — bytes after the 8-byte length prefix
 *     dataStart,              // int — file offset where tensor data begins
 *     metadata,               // object from "__metadata__" key, or {}
 *     tensors: [{
 *       name, dtype, shape,   // strings/numbers from JSON
 *       dataOffsets: [s, e],  // raw values from JSON (relative to dataStart)
 *       numElements,
 *       byteLength,           // e - s
 *       ggmlType,             // GGML type id; null if dtype unknown
 *     }, ...]
 *   }
 */
export function parseSafetensorsHeader(buffer) {
  const u8 = buffer instanceof Uint8Array
    ? buffer
    : new Uint8Array(buffer);
  if (u8.byteLength < 8) throw new Error('Buffer too small for safetensors header');

  const headerLength = readSafetensorsHeaderLength(u8);
  if (headerLength <= 0) throw new Error(`Invalid safetensors header length: ${headerLength}`);
  if (u8.byteLength < 8 + headerLength) {
    throw new Error(`Buffer truncated: need ${8 + headerLength} bytes, got ${u8.byteLength}`);
  }

  const headerBytes = u8.subarray(8, 8 + headerLength);
  let json;
  try { json = JSON.parse(new TextDecoder('utf-8').decode(headerBytes)); }
  catch (err) { throw new Error(`Safetensors JSON header is malformed: ${err.message}`); }
  if (!json || typeof json !== 'object') throw new Error('Safetensors header is not a JSON object');

  const metadata = (json.__metadata__ && typeof json.__metadata__ === 'object') ? json.__metadata__ : {};

  const tensors = [];
  for (const [name, info] of Object.entries(json)) {
    if (name === '__metadata__') continue;
    if (!info || typeof info !== 'object') continue;
    const { dtype, shape, data_offsets } = info;
    if (typeof dtype !== 'string' || !Array.isArray(shape) || !Array.isArray(data_offsets)) continue;
    const [s, e] = data_offsets;
    const numElements = shape.reduce((acc, d) => acc * Number(d), 1);
    const dtypeInfo = SAFETENSORS_DTYPES[dtype] || null;
    tensors.push({
      name,
      dtype,
      shape: shape.map(Number),
      dataOffsets: [Number(s), Number(e)],
      numElements,
      byteLength: Number(e) - Number(s),
      ggmlType: dtypeInfo ? dtypeInfo.ggmlType : null,
    });
  }

  return {
    headerLength,
    dataStart: 8 + headerLength,
    metadata,
    tensors,
  };
}
