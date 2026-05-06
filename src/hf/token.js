/**
 * Hugging Face access-token storage.
 *
 * The token is kept in localStorage under a single key. Browser code reads it
 * via getHfToken() and attaches it to /api/hf-file requests; the dev-server
 * proxy then forwards it to huggingface.co (and only there).
 *
 * Tests can inject a stub storage via _setStorageForTest().
 */

const KEY = 'llm-inspector:hf-token';

let _storage = (typeof localStorage !== 'undefined') ? localStorage : null;

/** @internal — used by tests only. */
export function _setStorageForTest(storage) { _storage = storage; }

/** Return the saved token (trimmed) or an empty string. Never throws. */
export function getHfToken() {
  if (!_storage) return '';
  try {
    const t = _storage.getItem(KEY);
    return typeof t === 'string' && t.trim() ? t.trim() : '';
  } catch { return ''; }
}

/**
 * Save (or clear) the HF token. Trimmed; an empty/whitespace string clears.
 * Returns the canonical (trimmed) value that was stored, or '' if cleared.
 */
export function setHfToken(token) {
  if (!_storage) return '';
  const t = (typeof token === 'string') ? token.trim() : '';
  try {
    if (t) _storage.setItem(KEY, t);
    else _storage.removeItem(KEY);
  } catch { /* ignore */ }
  return t;
}

/** Remove any saved token. */
export function clearHfToken() {
  if (!_storage) return;
  try { _storage.removeItem(KEY); } catch { /* ignore */ }
}

/**
 * Build the Authorization header object for an HF request, or {} if no token.
 * @param {string} [token] — override; defaults to the stored token.
 */
export function authHeaders(token) {
  const t = (token != null) ? String(token).trim() : getHfToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

/**
 * Return true if the response status indicates the request was rejected because
 * authentication is required or insufficient.
 */
export function isAuthError(status) {
  return status === 401 || status === 403;
}

/**
 * Build a user-facing message for an auth failure, tailored to whether a token
 * is currently saved.
 */
export function authErrorMessage(status, { hasToken = !!getHfToken() } = {}) {
  if (status === 401) {
    return hasToken
      ? 'Authentication failed (HTTP 401). Your Hugging Face token may be invalid or expired — update it at the top of the Hugging Face tab.'
      : 'This repo requires authentication (HTTP 401). Add a Hugging Face access token at the top of the Hugging Face tab.';
  }
  if (status === 403) {
    return hasToken
      ? 'Access denied (HTTP 403). Your Hugging Face token does not have access to this repo. If it is gated, accept its terms on huggingface.co first.'
      : 'This repo is gated (HTTP 403). Add a Hugging Face access token (with access to this repo) at the top of the Hugging Face tab.';
  }
  return `HTTP ${status}`;
}
