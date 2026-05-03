/**
 * Hugging Face hub helpers (Phase 1: URL-only input).
 *
 * Phase 1 accepts canonical resolve URLs of the form:
 *   https://huggingface.co/<owner>/<repo>/resolve/<revision>/<filePath>
 *
 * Future phases may add <owner>/<repo>[:<file>] shorthand and tree listing.
 */

/**
 * Parse and validate a Hugging Face resolve URL.
 * Returns { owner, repo, revision, filePath, resolveUrl }.
 * Throws an Error with a short, user-facing message on invalid input.
 */
export function parseHfRef(input) {
  if (typeof input !== 'string') throw new Error('Input must be a string');
  const trimmed = input.trim();
  if (!trimmed) throw new Error('Input is empty');

  let url;
  try { url = new URL(trimmed); }
  catch { throw new Error('Input is not a valid URL'); }

  if (url.protocol !== 'https:') {
    throw new Error('URL must use https');
  }
  if (url.hostname !== 'huggingface.co') {
    throw new Error('URL must be on huggingface.co');
  }

  const parts = url.pathname.split('/').filter(Boolean);
  if (parts.length < 4 || parts[2] !== 'resolve') {
    throw new Error('URL must look like https://huggingface.co/<owner>/<repo>/resolve/<rev>/<file>');
  }
  const [owner, repo, , revision, ...rest] = parts;
  const filePath = rest.join('/');
  if (!filePath) throw new Error('URL is missing the file path');

  // Canonicalise: drop query/hash so the proxy allowlist sees a clean URL.
  const resolveUrl = `https://huggingface.co/${owner}/${repo}/resolve/${revision}/${filePath}`;
  return { owner, repo, revision, filePath, resolveUrl };
}

/** True for paths whose final segment ends in .gguf (case-insensitive). */
export function isGgufPath(filePath) {
  if (typeof filePath !== 'string' || !filePath) return false;
  const noQuery = filePath.split('?')[0].split('#')[0];
  return /\.gguf$/i.test(noQuery);
}
