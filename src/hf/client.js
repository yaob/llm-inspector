/**
 * Hugging Face hub helpers.
 *
 * Two input shapes are supported:
 *   - File ref (parseHfRef):  https://huggingface.co/<owner>/<repo>/resolve/<rev>/<file>
 *   - Repo ref (parseHfRepoRef):
 *       https://huggingface.co/<owner>/<repo>           (optionally /tree/<rev>)
 *       <owner>/<repo>[@<rev>]                          (shorthand)
 */

const REPO_NAME_RE = /^[A-Za-z0-9._-]+$/;

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

/**
 * Parse a Hugging Face repo reference (URL or shorthand).
 * Returns { owner, repo, revision, repoUrl, resolveBaseUrl }.
 * Throws on input that looks like a file URL (callers should try parseHfRef first).
 */
export function parseHfRepoRef(input) {
  if (typeof input !== 'string') throw new Error('Input must be a string');
  const trimmed = input.trim();
  if (!trimmed) throw new Error('Input is empty');

  let owner, repo, revision = 'main';

  if (/^https?:\/\//i.test(trimmed)) {
    let url;
    try { url = new URL(trimmed); }
    catch { throw new Error('Input is not a valid URL'); }
    if (url.protocol !== 'https:') throw new Error('URL must use https');
    if (url.hostname !== 'huggingface.co') throw new Error('URL must be on huggingface.co');

    const parts = url.pathname.split('/').filter(Boolean);
    if (parts.length < 2) throw new Error('URL must include <owner>/<repo>');
    [owner, repo] = parts;
    if (parts[2] === 'resolve' || parts[2] === 'blob') {
      throw new Error('Input looks like a file URL, not a repo URL');
    }
    if (parts[2] === 'tree' && parts[3]) revision = parts[3];
  } else {
    // Shorthand: <owner>/<repo>[@<rev>]
    const m = trimmed.match(/^([^\/\s@]+)\/([^\/\s@]+)(?:@([^\/\s]+))?$/);
    if (!m) throw new Error('Expected <owner>/<repo>[@<revision>] shorthand or a huggingface.co URL');
    owner = m[1];
    repo = m[2];
    if (m[3]) revision = m[3];
  }

  if (!REPO_NAME_RE.test(owner) || !REPO_NAME_RE.test(repo)) {
    throw new Error('Owner and repo names contain invalid characters');
  }

  const repoUrl = `https://huggingface.co/${owner}/${repo}`;
  const resolveBaseUrl = `https://huggingface.co/${owner}/${repo}/resolve/${revision}`;
  return { owner, repo, revision, repoUrl, resolveBaseUrl };
}

/**
 * Build a canonical resolve URL for a file inside a repo.
 * @param {{owner: string, repo: string, revision: string}} ref
 * @param {string} filePath
 */
export function buildResolveUrl(ref, filePath) {
  if (!filePath) throw new Error('filePath is required');
  return `https://huggingface.co/${ref.owner}/${ref.repo}/resolve/${ref.revision}/${filePath}`;
}
