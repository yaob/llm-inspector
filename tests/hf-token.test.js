import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import {
  getHfToken, setHfToken, clearHfToken, authHeaders,
  isAuthError, authErrorMessage, _setStorageForTest,
} from '../src/hf/token.js';

/** Minimal localStorage stub for tests. */
function makeStorage() {
  const m = new Map();
  return {
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => { m.set(k, String(v)); },
    removeItem: (k) => { m.delete(k); },
    _map: m,
  };
}

describe('hf token storage', () => {
  beforeEach(() => {
    _setStorageForTest(makeStorage());
  });

  it('getHfToken returns "" when nothing is stored', () => {
    assert.equal(getHfToken(), '');
  });

  it('setHfToken trims and stores the value, then getHfToken returns it', () => {
    setHfToken('   hf_abcdef123456   ');
    assert.equal(getHfToken(), 'hf_abcdef123456');
  });

  it('setHfToken("") clears the stored token', () => {
    setHfToken('hf_xxx');
    assert.equal(getHfToken(), 'hf_xxx');
    setHfToken('');
    assert.equal(getHfToken(), '');
  });

  it('setHfToken with whitespace-only clears the token', () => {
    setHfToken('hf_xxx');
    setHfToken('   ');
    assert.equal(getHfToken(), '');
  });

  it('setHfToken returns the canonical stored value', () => {
    assert.equal(setHfToken('  hf_yyy  '), 'hf_yyy');
    assert.equal(setHfToken(''), '');
  });

  it('setHfToken ignores non-string input by treating it as clear', () => {
    setHfToken('hf_xxx');
    setHfToken(undefined);
    assert.equal(getHfToken(), '');
  });

  it('clearHfToken removes the stored token', () => {
    setHfToken('hf_zzz');
    clearHfToken();
    assert.equal(getHfToken(), '');
  });

  it('returns "" gracefully when no storage is configured', () => {
    _setStorageForTest(null);
    assert.equal(getHfToken(), '');
    // Setters become no-ops, not throws
    assert.doesNotThrow(() => setHfToken('hf_x'));
    assert.doesNotThrow(() => clearHfToken());
  });

  it('survives a storage that throws on getItem', () => {
    _setStorageForTest({
      getItem: () => { throw new Error('quota'); },
      setItem: () => {},
      removeItem: () => {},
    });
    assert.equal(getHfToken(), '');
  });
});

describe('authHeaders', () => {
  beforeEach(() => { _setStorageForTest(makeStorage()); });

  it('returns {} when no token is set', () => {
    assert.deepEqual(authHeaders(), {});
  });

  it('returns Bearer header when a token is set', () => {
    setHfToken('hf_abc');
    assert.deepEqual(authHeaders(), { Authorization: 'Bearer hf_abc' });
  });

  it('uses an explicit token override when provided', () => {
    setHfToken('hf_stored');
    assert.deepEqual(authHeaders('hf_override'), { Authorization: 'Bearer hf_override' });
  });

  it('returns {} for an explicit empty/whitespace override', () => {
    setHfToken('hf_stored');
    assert.deepEqual(authHeaders(''), {});
    assert.deepEqual(authHeaders('   '), {});
  });
});

describe('isAuthError', () => {
  it('matches 401 and 403', () => {
    assert.equal(isAuthError(401), true);
    assert.equal(isAuthError(403), true);
  });
  it('rejects other status codes', () => {
    for (const s of [200, 206, 400, 404, 500, 502]) {
      assert.equal(isAuthError(s), false);
    }
  });
});

describe('authErrorMessage', () => {
  it('401 with no token suggests adding one', () => {
    const msg = authErrorMessage(401, { hasToken: false });
    assert.match(msg, /401/);
    assert.match(msg, /token/i);
    assert.match(msg, /add/i);
  });

  it('401 with a token suggests it may be invalid', () => {
    const msg = authErrorMessage(401, { hasToken: true });
    assert.match(msg, /invalid|expired/i);
  });

  it('403 with no token mentions gated repos', () => {
    const msg = authErrorMessage(403, { hasToken: false });
    assert.match(msg, /403/);
    assert.match(msg, /gated/i);
  });

  it('403 with a token mentions accepting repo terms', () => {
    const msg = authErrorMessage(403, { hasToken: true });
    assert.match(msg, /access|terms/i);
  });

  it('falls back to a generic message for unrelated codes', () => {
    assert.equal(authErrorMessage(500, { hasToken: false }), 'HTTP 500');
  });
});
