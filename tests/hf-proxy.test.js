/**
 * Tests for the /api/hf-file proxy handler — drives createHfFileHandler with
 * a mocked fetchImpl so we can exercise the success path (header/body
 * forwarding) and the timeout path without any network access.
 */
import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { createHfFileHandler } from '../server.js';

/** Spin up a one-off HTTP server backed by the given handler. */
async function startTestServer(handler) {
  const server = createServer((req, res) => {
    handler(req, res).catch((err) => {
      if (!res.headersSent) { res.writeHead(500); res.end(String(err)); }
    });
  });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  const { port } = server.address();
  return { server, base: `http://127.0.0.1:${port}` };
}

const HF_URL = 'https://huggingface.co/o/r/resolve/main/file.bin';
const PROXY = (base, target = HF_URL) => `${base}/?url=${encodeURIComponent(target)}`;

describe('hf-file proxy upstream-success path', () => {
  let server;
  let base;
  let lastCall;

  before(async () => {
    const fetchImpl = async (url, init) => {
      lastCall = { url, init };
      // Echo back a 206 with a body and a few canonical HF response headers.
      return new Response(new Uint8Array([0x47, 0x47, 0x55, 0x46]), {
        status: 206,
        headers: {
          'content-length': '4',
          'content-range': 'bytes 0-3/100',
          'accept-ranges': 'bytes',
          'etag': '"abc123"',
        },
      });
    };
    const handler = createHfFileHandler({ fetchImpl, timeoutMs: 5_000 });
    ({ server, base } = await startTestServer(handler));
  });

  after(async () => {
    if (server) await new Promise(r => server.close(r));
  });

  it('forwards Range and pipes the upstream body through unchanged', async () => {
    const res = await fetch(PROXY(base), { headers: { Range: 'bytes=0-3' } });
    assert.equal(res.status, 206);
    assert.equal(lastCall.init.method, 'GET');
    assert.equal(lastCall.init.headers.Range, 'bytes=0-3');
    const buf = new Uint8Array(await res.arrayBuffer());
    assert.deepEqual([...buf], [0x47, 0x47, 0x55, 0x46]);
  });

  it('forwards Authorization header to the upstream fetch', async () => {
    await fetch(PROXY(base), { headers: { Authorization: 'Bearer hf_test_token' } });
    assert.equal(lastCall.init.headers.Authorization, 'Bearer hf_test_token');
  });

  it('does not forward Authorization when the client did not send one', async () => {
    await fetch(PROXY(base));
    assert.equal(lastCall.init.headers.Authorization, undefined);
  });

  it('mirrors Content-Length, Content-Range, Accept-Ranges, and ETag from upstream', async () => {
    const res = await fetch(PROXY(base), { headers: { Range: 'bytes=0-3' } });
    assert.equal(res.headers.get('content-length'), '4');
    assert.equal(res.headers.get('content-range'), 'bytes 0-3/100');
    assert.equal(res.headers.get('accept-ranges'), 'bytes');
    assert.equal(res.headers.get('etag'), '"abc123"');
  });

  it('uses HEAD upstream when the client sends HEAD', async () => {
    await fetch(PROXY(base), { method: 'HEAD' });
    assert.equal(lastCall.init.method, 'HEAD');
  });

  it('still emits CORS headers on a successful response', async () => {
    const res = await fetch(PROXY(base));
    assert.equal(res.headers.get('access-control-allow-origin'), '*');
  });
});

describe('hf-file proxy timeout', () => {
  let server;
  let base;

  before(async () => {
    // fetchImpl that hangs until aborted by the upstream signal.
    const fetchImpl = (url, init) => new Promise((_, reject) => {
      const sig = init?.signal;
      if (sig) {
        sig.addEventListener('abort', () => {
          const err = new Error('aborted');
          err.name = 'AbortError';
          reject(err);
        });
      }
    });
    const handler = createHfFileHandler({ fetchImpl, timeoutMs: 50 });
    ({ server, base } = await startTestServer(handler));
  });

  after(async () => {
    if (server) await new Promise(r => server.close(r));
  });

  it('returns 504 when the upstream fetch is aborted by the timeout', async () => {
    const res = await fetch(PROXY(base));
    assert.equal(res.status, 504);
    const body = await res.text();
    assert.match(body, /timeout/i);
  });
});

describe('hf-file proxy upstream error', () => {
  let server;
  let base;

  before(async () => {
    const fetchImpl = async () => { throw new Error('connection refused'); };
    const handler = createHfFileHandler({ fetchImpl, timeoutMs: 5_000 });
    ({ server, base } = await startTestServer(handler));
  });

  after(async () => {
    if (server) await new Promise(r => server.close(r));
  });

  it('returns 502 with the upstream error message', async () => {
    const res = await fetch(PROXY(base));
    assert.equal(res.status, 502);
    const body = await res.text();
    assert.match(body, /connection refused/);
  });
});
