/**
 * Dev server: static file serving + local file proxy for Ollama GGUF blobs.
 *
 * Serves the project directory as static files (like http-server) and adds a
 * special /api/local-file endpoint that reads a local file by path with Range
 * header support. This lets the browser load Ollama's on-disk GGUF blobs
 * without symlinking or copying.
 *
 * Usage: node server.js [port]
 */

import { createServer } from 'node:http';
import { open, readFile, stat } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';

const PORT = parseInt(process.argv[2] || '8088', 10);
const ROOT = resolve(fileURLToPath(import.meta.url), '..');

const MIME = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.mjs': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
};

function cors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Range, Authorization');
  res.setHeader('Access-Control-Expose-Headers', 'Content-Range, Content-Length');
}

/** Serve a local file by path with Range support (for Ollama blobs). */
async function handleLocalFile(req, res) {
  cors(res);
  if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

  const url = new URL(req.url, `http://${req.headers.host}`);
  const filePath = url.searchParams.get('path');
  if (!filePath) { res.writeHead(400); res.end('Missing ?path='); return; }

  // Security: only allow files under ~/.ollama
  const home = process.env.HOME || process.env.USERPROFILE || '';
  const allowed = resolve(home, '.ollama');
  const resolved = resolve(filePath);
  // Windows paths are case-insensitive; normalise before comparing
  const norm = process.platform === 'win32'
    ? s => s.toLowerCase()
    : s => s;
  if (!norm(resolved).startsWith(norm(allowed))) {
    res.writeHead(403);
    res.end('Forbidden: only files under ~/.ollama are accessible');
    return;
  }

  let info;
  try { info = await stat(resolved); } catch { res.writeHead(404); res.end('Not found'); return; }
  if (!info.isFile()) { res.writeHead(400); res.end('Not a file'); return; }

  const total = info.size;

  // HEAD: return size only (stat already succeeded, no read needed)
  if (req.method === 'HEAD') {
    res.writeHead(200, {
      'Content-Type': 'application/octet-stream',
      'Content-Length': total,
      'Accept-Ranges': 'bytes',
    });
    res.end();
    return;
  }

  // Verify read access before sending success headers (macOS may EPERM on open)
  let fd;
  try {
    fd = await open(resolved, 'r');
  } catch (err) {
    const isPerm = err.code === 'EPERM' || err.code === 'EACCES';
    res.writeHead(isPerm ? 403 : 500);
    res.end(isPerm
      ? 'Permission denied: grant Full Disk Access to your terminal app in System Settings → Privacy & Security → Full Disk Access, then restart the server.'
      : `Cannot read file: ${err.message}`);
    return;
  } finally {
    if (fd) await fd.close();
  }

  const range = req.headers.range;

  if (range) {
    const m = range.match(/bytes=(\d+)-(\d*)/);
    if (!m) { res.writeHead(416); res.end(); return; }
    const start = parseInt(m[1], 10);
    const end = m[2] ? parseInt(m[2], 10) : total - 1;
    if (start >= total || end >= total) { res.writeHead(416); res.end(); return; }
    res.writeHead(206, {
      'Content-Type': 'application/octet-stream',
      'Content-Range': `bytes ${start}-${end}/${total}`,
      'Content-Length': end - start + 1,
      'Accept-Ranges': 'bytes',
    });
    const stream = createReadStream(resolved, { start, end });
    stream.on('error', (err) => {
      console.error('Stream error:', err.message);
      res.end();
    });
    stream.pipe(res);
  } else {
    res.writeHead(200, {
      'Content-Type': 'application/octet-stream',
      'Content-Length': total,
      'Accept-Ranges': 'bytes',
    });
    const stream = createReadStream(resolved);
    stream.on('error', (err) => {
      console.error('Stream error:', err.message);
      res.end();
    });
    stream.pipe(res);
  }
}

const HF_UPSTREAM_TIMEOUT_MS = 30_000;

/**
 * Build the Hugging Face proxy handler. Exported as a factory so tests can
 * inject a mock fetchImpl and exercise the upstream-success path without
 * touching the network.
 */
export function createHfFileHandler({ fetchImpl = fetch, timeoutMs = HF_UPSTREAM_TIMEOUT_MS } = {}) {
  return async function handleHfFile(req, res) {
    cors(res);
    if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }

    const url = new URL(req.url, `http://${req.headers.host}`);
    const target = url.searchParams.get('url');
    if (!target) { res.writeHead(400); res.end('Missing ?url='); return; }

    let parsed;
    try { parsed = new URL(target); }
    catch { res.writeHead(400); res.end('Invalid url'); return; }

    // Security: only allow https://huggingface.co/<owner>/<repo>/resolve/...
    if (parsed.protocol !== 'https:' || parsed.hostname !== 'huggingface.co') {
      res.writeHead(403); res.end('Forbidden: only https://huggingface.co/ is allowed'); return;
    }
    if (!/^\/[^/]+\/[^/]+\/resolve\/[^/]+\/.+/.test(parsed.pathname)) {
      res.writeHead(403); res.end('Forbidden: only /<owner>/<repo>/resolve/<rev>/<file> paths are allowed'); return;
    }

    const fwdHeaders = {};
    if (req.headers.range) fwdHeaders['Range'] = req.headers.range;
    // Forward Authorization to huggingface.co only — the allowlist above guarantees
    // no token can leak to any other host.
    if (req.headers.authorization) fwdHeaders['Authorization'] = req.headers.authorization;

    const method = req.method === 'HEAD' ? 'HEAD' : 'GET';
    let upstream;
    try {
      upstream = await fetchImpl(parsed.toString(), {
        method,
        headers: fwdHeaders,
        redirect: 'follow',
        signal: AbortSignal.timeout(timeoutMs),
      });
    } catch (err) {
      const isTimeout = err.name === 'TimeoutError' || err.name === 'AbortError';
      res.writeHead(isTimeout ? 504 : 502);
      res.end(`Upstream ${isTimeout ? 'timeout' : 'error'}: ${err.message}`);
      return;
    }

    const out = { 'Content-Type': 'application/octet-stream' };
    const cl = upstream.headers.get('content-length');
    const cr = upstream.headers.get('content-range');
    const ar = upstream.headers.get('accept-ranges');
    const et = upstream.headers.get('etag');
    if (cl) out['Content-Length'] = cl;
    if (cr) out['Content-Range'] = cr;
    if (ar) out['Accept-Ranges'] = ar;
    if (et) out['ETag'] = et;

    res.writeHead(upstream.status, out);

    if (method === 'HEAD' || !upstream.body) { res.end(); return; }

    const nodeStream = Readable.fromWeb(upstream.body);
    nodeStream.on('error', (err) => {
      console.error('HF stream error:', err.message);
      res.end();
    });
    nodeStream.pipe(res);
  };
}

const handleHfFile = createHfFileHandler();

/** Serve static files from the project root. */
async function handleStatic(req, res) {
  const url = new URL(req.url, `http://${req.headers.host}`);
  let filePath = join(ROOT, decodeURIComponent(url.pathname));
  if (filePath.endsWith('/') || filePath.endsWith('\\')) filePath = join(filePath, 'index.html');

  const ext = extname(filePath);
  const mime = MIME[ext] || 'application/octet-stream';

  try {
    const data = await readFile(filePath);
    res.writeHead(200, {
      'Content-Type': mime,
      'Content-Length': data.length,
      'Cache-Control': 'no-cache',
    });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end('Not found');
  }
}

const server = createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);
  const handler = url.pathname === '/api/local-file' ? handleLocalFile
    : url.pathname === '/api/hf-file' ? handleHfFile
    : handleStatic;
  handler(req, res).catch((err) => {
    console.error('Request handler error:', err);
    if (!res.headersSent) { res.writeHead(500); res.end('Internal server error'); }
  });
});

// Only start listening when run as a script, not when imported by tests.
const isMain = import.meta.url === `file://${process.argv[1]}`;
if (isMain) {
  server.listen(PORT, '127.0.0.1', () => {
    console.log(`LLM Inspector running at http://127.0.0.1:${PORT}`);
    console.log(`Serving static files from ${ROOT}`);
    console.log(`Local file proxy at /api/local-file?path=...`);
    console.log(`Hugging Face proxy at /api/hf-file?url=...`);
  });
}

