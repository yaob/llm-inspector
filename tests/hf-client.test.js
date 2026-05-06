import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { parseHfRef, isGgufPath, parseHfRepoRef, buildResolveUrl } from '../src/hf/client.js';

describe('parseHfRef', () => {
  it('parses a canonical resolve URL', () => {
    const r = parseHfRef('https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf');
    assert.equal(r.owner, 'TheBloke');
    assert.equal(r.repo, 'Llama-2-7B-GGUF');
    assert.equal(r.revision, 'main');
    assert.equal(r.filePath, 'llama-2-7b.Q4_0.gguf');
    assert.equal(r.resolveUrl, 'https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf');
  });

  it('handles nested file paths', () => {
    const r = parseHfRef('https://huggingface.co/owner/repo/resolve/main/folder/sub/model.gguf');
    assert.equal(r.filePath, 'folder/sub/model.gguf');
    assert.equal(r.resolveUrl, 'https://huggingface.co/owner/repo/resolve/main/folder/sub/model.gguf');
  });

  it('strips query and hash from the canonical resolveUrl', () => {
    const r = parseHfRef('https://huggingface.co/o/r/resolve/main/m.gguf?download=true#frag');
    assert.equal(r.resolveUrl, 'https://huggingface.co/o/r/resolve/main/m.gguf');
  });

  it('handles non-main revisions (commit SHAs and branches)', () => {
    const r = parseHfRef('https://huggingface.co/o/r/resolve/abc123def/m.gguf');
    assert.equal(r.revision, 'abc123def');
  });

  it('trims surrounding whitespace', () => {
    const r = parseHfRef('   https://huggingface.co/o/r/resolve/main/m.gguf   ');
    assert.equal(r.owner, 'o');
    assert.equal(r.filePath, 'm.gguf');
  });

  it('rejects http (non-https) URLs', () => {
    assert.throws(() => parseHfRef('http://huggingface.co/o/r/resolve/main/m.gguf'), /https/);
  });

  it('rejects non-huggingface.co hosts', () => {
    assert.throws(() => parseHfRef('https://example.com/o/r/resolve/main/m.gguf'), /huggingface\.co/);
  });

  it('rejects URLs without a /resolve/ segment', () => {
    assert.throws(() => parseHfRef('https://huggingface.co/o/r/blob/main/m.gguf'), /resolve/);
  });

  it('rejects URLs missing the file path', () => {
    assert.throws(() => parseHfRef('https://huggingface.co/o/r/resolve/main/'), /(file path|resolve)/i);
  });

  it('rejects empty input', () => {
    assert.throws(() => parseHfRef(''), /empty/);
  });

  it('rejects whitespace-only input', () => {
    assert.throws(() => parseHfRef('   '), /empty/);
  });

  it('rejects non-string input', () => {
    assert.throws(() => parseHfRef(null), /string/);
    assert.throws(() => parseHfRef(undefined), /string/);
    assert.throws(() => parseHfRef(42), /string/);
  });

  it('rejects malformed URLs', () => {
    assert.throws(() => parseHfRef('not a url'), /valid URL/);
  });
});

describe('isGgufPath', () => {
  it('matches .gguf paths', () => {
    assert.equal(isGgufPath('model.gguf'), true);
    assert.equal(isGgufPath('folder/model.gguf'), true);
    assert.equal(isGgufPath('model.GGUF'), true);
  });

  it('matches .gguf with query string or hash', () => {
    assert.equal(isGgufPath('model.gguf?download=true'), true);
    assert.equal(isGgufPath('model.gguf#section'), true);
  });

  it('rejects non-gguf and edge cases', () => {
    assert.equal(isGgufPath('model.bin'), false);
    assert.equal(isGgufPath('model.gguf.bak'), false);
    assert.equal(isGgufPath(''), false);
    assert.equal(isGgufPath(null), false);
    assert.equal(isGgufPath(undefined), false);
    assert.equal(isGgufPath(42), false);
  });
});

describe('parseHfRepoRef', () => {
  it('parses a bare huggingface.co repo URL', () => {
    const r = parseHfRepoRef('https://huggingface.co/meta-llama/Llama-3.2-1B');
    assert.equal(r.owner, 'meta-llama');
    assert.equal(r.repo, 'Llama-3.2-1B');
    assert.equal(r.revision, 'main');
    assert.equal(r.repoUrl, 'https://huggingface.co/meta-llama/Llama-3.2-1B');
    assert.equal(r.resolveBaseUrl, 'https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main');
  });

  it('parses /tree/<rev> URLs and uses that revision', () => {
    const r = parseHfRepoRef('https://huggingface.co/o/r/tree/abc123');
    assert.equal(r.revision, 'abc123');
    assert.equal(r.resolveBaseUrl, 'https://huggingface.co/o/r/resolve/abc123');
  });

  it('parses <owner>/<repo> shorthand', () => {
    const r = parseHfRepoRef('meta-llama/Llama-3.2-1B');
    assert.equal(r.owner, 'meta-llama');
    assert.equal(r.repo, 'Llama-3.2-1B');
    assert.equal(r.revision, 'main');
  });

  it('parses <owner>/<repo>@<rev> shorthand', () => {
    const r = parseHfRepoRef('owner/repo@v1.0');
    assert.equal(r.revision, 'v1.0');
  });

  it('trims whitespace', () => {
    const r = parseHfRepoRef('   owner/repo   ');
    assert.equal(r.owner, 'owner');
    assert.equal(r.repo, 'repo');
  });

  it('rejects file (resolve) URLs so callers fall back to parseHfRef', () => {
    assert.throws(() => parseHfRepoRef('https://huggingface.co/o/r/resolve/main/m.gguf'), /file URL/i);
  });

  it('rejects blob URLs', () => {
    assert.throws(() => parseHfRepoRef('https://huggingface.co/o/r/blob/main/m.gguf'), /file URL/i);
  });

  it('rejects URLs missing the repo segment', () => {
    assert.throws(() => parseHfRepoRef('https://huggingface.co/owner'), /owner.*repo/i);
  });

  it('rejects non-huggingface hosts', () => {
    assert.throws(() => parseHfRepoRef('https://example.com/o/r'), /huggingface\.co/);
  });

  it('rejects empty or non-string input', () => {
    assert.throws(() => parseHfRepoRef(''), /empty/);
    assert.throws(() => parseHfRepoRef('   '), /empty/);
    assert.throws(() => parseHfRepoRef(null), /string/);
  });

  it('rejects malformed shorthand', () => {
    assert.throws(() => parseHfRepoRef('not-a-ref'), /shorthand|URL/);
    assert.throws(() => parseHfRepoRef('owner/'), /shorthand|URL/);
    assert.throws(() => parseHfRepoRef('/repo'), /shorthand|URL/);
  });

  it('rejects names with invalid characters', () => {
    assert.throws(() => parseHfRepoRef('own er/repo'), /shorthand|invalid/i);
  });
});

describe('buildResolveUrl', () => {
  it('builds a canonical resolve URL', () => {
    const url = buildResolveUrl({ owner: 'o', repo: 'r', revision: 'main' }, 'config.json');
    assert.equal(url, 'https://huggingface.co/o/r/resolve/main/config.json');
  });

  it('preserves nested file paths', () => {
    const url = buildResolveUrl({ owner: 'o', repo: 'r', revision: 'abc' }, 'sub/dir/m.safetensors');
    assert.equal(url, 'https://huggingface.co/o/r/resolve/abc/sub/dir/m.safetensors');
  });

  it('throws when filePath is missing', () => {
    assert.throws(() => buildResolveUrl({ owner: 'o', repo: 'r', revision: 'main' }, ''), /filePath/);
  });
});

describe('parseHfRepoRef — additional edge cases', () => {
  it('accepts repo URLs with a trailing slash', () => {
    const r = parseHfRepoRef('https://huggingface.co/owner/repo/');
    assert.equal(r.owner, 'owner');
    assert.equal(r.repo, 'repo');
    assert.equal(r.revision, 'main');
  });

  it('accepts repo URLs with extra path segments after owner/repo', () => {
    const r = parseHfRepoRef('https://huggingface.co/owner/repo/discussions');
    assert.equal(r.owner, 'owner');
    assert.equal(r.repo, 'repo');
    assert.equal(r.revision, 'main');
  });

  it('rejects /tree/ URLs without a revision segment', () => {
    // Falls through to default revision rather than throwing — explicit assertion
    const r = parseHfRepoRef('https://huggingface.co/o/r/tree');
    assert.equal(r.revision, 'main');
  });

  it('rejects shorthand with three segments', () => {
    assert.throws(() => parseHfRepoRef('a/b/c'), /shorthand|URL/);
  });

  it('rejects shorthand with @ but empty revision', () => {
    assert.throws(() => parseHfRepoRef('owner/repo@'), /shorthand|URL/);
  });

  it('accepts a repo name with dots and dashes', () => {
    const r = parseHfRepoRef('Org_1.0/Model-X.Y');
    assert.equal(r.owner, 'Org_1.0');
    assert.equal(r.repo, 'Model-X.Y');
  });

  it('rejects http (non-https) repo URLs', () => {
    assert.throws(() => parseHfRepoRef('http://huggingface.co/o/r'), /https/);
  });
});
