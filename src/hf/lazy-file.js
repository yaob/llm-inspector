/**
 * HfLazyFile — a File-like wrapper that reads a Hugging Face hub file on
 * demand via HTTP Range requests to the dev server's /api/hf-file proxy.
 *
 * Implements the subset of the File/Blob API used by the GGUF parser and
 * tensor decoder: .name, .size, .slice(start, end).arrayBuffer().
 */
import { authHeaders, authErrorMessage, isAuthError } from './token.js';

export class HfLazyFile {
  /**
   * @param {string} resolveUrl - canonical https://huggingface.co/.../resolve/... URL
   * @param {string} name - display name (e.g. "hf:owner/repo/file.gguf")
   * @param {number} size - total file size in bytes
   */
  constructor(resolveUrl, name, size) {
    this._resolveUrl = resolveUrl;
    this.name = name;
    this.size = size;
  }

  /**
   * Create an HfLazyFile by probing the proxy for the file's size.
   * @param {string} resolveUrl
   * @param {string} name
   * @returns {Promise<HfLazyFile>}
   */
  static async create(resolveUrl, name) {
    const url = `/api/hf-file?url=${encodeURIComponent(resolveUrl)}`;
    const auth = authHeaders();

    // Probe with a small Range GET to verify access AND seed size.
    const probeRes = await fetch(url, { headers: { Range: 'bytes=0-3', ...auth } });
    if (isAuthError(probeRes.status)) {
      throw new Error(authErrorMessage(probeRes.status));
    }
    if (!probeRes.ok && probeRes.status !== 206) {
      const msg = await probeRes.text().catch(() => '');
      throw new Error(`Cannot access Hugging Face file (HTTP ${probeRes.status})${msg ? `: ${msg}` : ''}`);
    }

    // HEAD to get total file size
    const headRes = await fetch(url, { method: 'HEAD', headers: auth });
    if (headRes.ok) {
      const size = parseInt(headRes.headers.get('Content-Length') || '0', 10);
      if (size) return new HfLazyFile(resolveUrl, name, size);
    }
    // Fallback: parse Content-Range from the probe response
    const cr = probeRes.headers.get('Content-Range') || '';
    const m = cr.match(/\/(\d+)$/);
    const size = m ? parseInt(m[1], 10) : 0;
    if (!size) throw new Error(`Cannot determine size of Hugging Face file: ${resolveUrl}`);
    return new HfLazyFile(resolveUrl, name, size);
  }

  /**
   * Return a Blob-like slice that supports .arrayBuffer().
   * Compatible with how decodeTensorSlice calls file.slice(start, end).arrayBuffer().
   */
  slice(start, end) {
    const resolveUrl = this._resolveUrl;
    return {
      arrayBuffer() {
        const url = `/api/hf-file?url=${encodeURIComponent(resolveUrl)}`;
        return fetch(url, {
          headers: { Range: `bytes=${start}-${end - 1}`, ...authHeaders() },
        }).then(async res => {
          if (isAuthError(res.status)) throw new Error(authErrorMessage(res.status));
          if (!res.ok && res.status !== 206) {
            const msg = await res.text().catch(() => '');
            throw new Error(`Failed to read bytes ${start}-${end} from ${resolveUrl}${msg ? `: ${msg}` : ''}`);
          }
          return res.arrayBuffer();
        });
      },
    };
  }
}
