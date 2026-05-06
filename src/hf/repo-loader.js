/**
 * Browser-side helpers for loading a Hugging Face repo's metadata via the
 * /api/hf-file proxy. Uses HTTP Range requests to read safetensors headers
 * without downloading the full weight files.
 */

import { buildResolveUrl } from './client.js';
import { parseSafetensorsHeader, readSafetensorsHeaderLength } from '../parsers/safetensors.js';

const PROXY = (resolveUrl) => `/api/hf-file?url=${encodeURIComponent(resolveUrl)}`;

/** Fetch a small repo file (config.json, index.json) as text via the proxy. */
async function fetchRepoText(resolveUrl) {
  const res = await fetch(PROXY(resolveUrl));
  if (!res.ok) {
    const msg = await res.text().catch(() => '');
    const err = new Error(`HTTP ${res.status} ${resolveUrl}${msg ? `: ${msg}` : ''}`);
    err.status = res.status;
    throw err;
  }
  return await res.text();
}

/** Fetch a Range from the proxy. Returns an ArrayBuffer. */
async function fetchRange(resolveUrl, start, end) {
  const res = await fetch(PROXY(resolveUrl), { headers: { Range: `bytes=${start}-${end}` } });
  if (!res.ok && res.status !== 206) {
    const msg = await res.text().catch(() => '');
    throw new Error(`HTTP ${res.status} on ${resolveUrl}${msg ? `: ${msg}` : ''}`);
  }
  return await res.arrayBuffer();
}

/** Read just the safetensors JSON header for a shard. */
export async function readShardHeader(resolveUrl) {
  const lenBuf = await fetchRange(resolveUrl, 0, 7);
  const headerLength = readSafetensorsHeaderLength(lenBuf);
  if (!Number.isFinite(headerLength) || headerLength <= 0 || headerLength > 256 * 1024 * 1024) {
    throw new Error(`Implausible safetensors header length: ${headerLength}`);
  }
  const headerBuf = await fetchRange(resolveUrl, 0, 8 + headerLength - 1);
  return parseSafetensorsHeader(headerBuf);
}

/**
 * Resolve the list of safetensors shard filenames for a repo.
 * Returns { shards: [filename, ...] } — single-element array for unsharded models.
 */
export async function resolveShards(ref) {
  // Try the index first (sharded models)
  try {
    const indexText = await fetchRepoText(buildResolveUrl(ref, 'model.safetensors.index.json'));
    const index = JSON.parse(indexText);
    const map = index.weight_map || {};
    const set = new Set();
    for (const fname of Object.values(map)) {
      if (typeof fname === 'string') set.add(fname);
    }
    if (set.size > 0) return { shards: [...set].sort() };
  } catch (err) {
    if (err.status !== 404) {
      // Network/parse error is worth surfacing; missing index is normal for single-file repos.
      // Fall through and try the single-file path so the user still gets a useful error.
    }
  }
  // Fall back to single file
  return { shards: ['model.safetensors'] };
}

/**
 * Top-level: load a repo's config + safetensors metadata.
 * @param {{owner, repo, revision}} ref
 * @param {(msg: string) => void} [onProgress]
 * @returns {Promise<{config: object, tensors: Array, shards: string[]}>}
 */
export async function loadHfRepo(ref, onProgress = () => {}) {
  onProgress('Fetching config.json…');
  const configText = await fetchRepoText(buildResolveUrl(ref, 'config.json'));
  let config;
  try { config = JSON.parse(configText); }
  catch (err) { throw new Error(`config.json is not valid JSON: ${err.message}`); }

  onProgress('Discovering safetensors shards…');
  const { shards } = await resolveShards(ref);
  if (shards.length === 0) throw new Error('No safetensors files found in repo');

  const tensors = [];
  for (let i = 0; i < shards.length; i++) {
    const shard = shards[i];
    onProgress(`Reading shard ${i + 1}/${shards.length}: ${shard}…`);
    const url = buildResolveUrl(ref, shard);
    let header;
    try { header = await readShardHeader(url); }
    catch (err) { throw new Error(`Failed to read ${shard}: ${err.message}`); }
    for (const t of header.tensors) {
      tensors.push({ ...t, shard });
    }
  }

  if (tensors.length === 0) throw new Error('No tensors found in repo safetensors files');

  return { config, tensors, shards };
}
