import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { parseHfRef, isGgufPath } from '../src/hf/client.js';

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
