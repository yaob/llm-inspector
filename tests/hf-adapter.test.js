import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { adaptHfRepo, mapHfTensorName } from '../src/model/hf-adapter.js';
import { analyzeModel } from '../src/model/analyzer.js';

/** Build a minimal Llama-style tensor list for a tiny model. */
function llamaTensors(numLayers, hidden, ffn, vocab) {
  const tensors = [
    { name: 'model.embed_tokens.weight', dtype: 'F16', shape: [vocab, hidden], dataOffsets: [0, 0], numElements: vocab * hidden, byteLength: vocab * hidden * 2, ggmlType: 1 },
    { name: 'model.norm.weight',         dtype: 'F32', shape: [hidden],         dataOffsets: [0, 0], numElements: hidden,          byteLength: hidden * 4,          ggmlType: 0 },
    { name: 'lm_head.weight',            dtype: 'F16', shape: [vocab, hidden], dataOffsets: [0, 0], numElements: vocab * hidden, byteLength: vocab * hidden * 2, ggmlType: 1 },
  ];
  for (let i = 0; i < numLayers; i++) {
    const add = (sub, dtype, shape) => {
      const n = shape.reduce((a, b) => a * b, 1);
      const bpe = dtype === 'F32' ? 4 : 2;
      tensors.push({ name: `model.layers.${i}.${sub}.weight`, dtype, shape, dataOffsets: [0, 0], numElements: n, byteLength: n * bpe, ggmlType: dtype === 'F32' ? 0 : 1 });
    };
    add('input_layernorm', 'F32', [hidden]);
    add('self_attn.q_proj', 'F16', [hidden, hidden]);
    add('self_attn.k_proj', 'F16', [hidden, hidden]);
    add('self_attn.v_proj', 'F16', [hidden, hidden]);
    add('self_attn.o_proj', 'F16', [hidden, hidden]);
    add('post_attention_layernorm', 'F32', [hidden]);
    add('mlp.gate_proj', 'F16', [ffn, hidden]);
    add('mlp.up_proj',   'F16', [ffn, hidden]);
    add('mlp.down_proj', 'F16', [hidden, ffn]);
  }
  return tensors;
}

describe('mapHfTensorName', () => {
  it('maps LLaMA block tensors to GGUF naming', () => {
    assert.equal(mapHfTensorName('model.layers.0.self_attn.q_proj.weight', 'llama'), 'blk.0.attn_q.weight');
    assert.equal(mapHfTensorName('model.layers.7.self_attn.k_proj.weight', 'llama'), 'blk.7.attn_k.weight');
    assert.equal(mapHfTensorName('model.layers.3.self_attn.v_proj.weight', 'llama'), 'blk.3.attn_v.weight');
    assert.equal(mapHfTensorName('model.layers.0.self_attn.o_proj.weight', 'llama'), 'blk.0.attn_output.weight');
    assert.equal(mapHfTensorName('model.layers.0.input_layernorm.weight', 'llama'), 'blk.0.attn_norm.weight');
    assert.equal(mapHfTensorName('model.layers.0.post_attention_layernorm.weight', 'llama'), 'blk.0.ffn_norm.weight');
    assert.equal(mapHfTensorName('model.layers.0.mlp.gate_proj.weight', 'llama'), 'blk.0.ffn_gate.weight');
    assert.equal(mapHfTensorName('model.layers.0.mlp.up_proj.weight',   'llama'), 'blk.0.ffn_up.weight');
    assert.equal(mapHfTensorName('model.layers.0.mlp.down_proj.weight', 'llama'), 'blk.0.ffn_down.weight');
  });

  it('maps top-level tensors', () => {
    assert.equal(mapHfTensorName('model.embed_tokens.weight', 'llama'), 'token_embd.weight');
    assert.equal(mapHfTensorName('model.norm.weight', 'llama'), 'output_norm.weight');
    assert.equal(mapHfTensorName('lm_head.weight', 'llama'), 'output.weight');
  });

  it('preserves bias suffixes', () => {
    assert.equal(mapHfTensorName('model.layers.0.self_attn.q_proj.bias', 'llama'), 'blk.0.attn_q.bias');
    assert.equal(mapHfTensorName('lm_head.bias', 'llama'), 'output.bias');
  });

  it('passes through unknown names unchanged', () => {
    assert.equal(mapHfTensorName('something.weird', 'llama'), 'something.weird');
  });

  it('maps qwen2 q/k norm tensors', () => {
    assert.equal(mapHfTensorName('model.layers.0.self_attn.q_norm.weight', 'qwen2'), 'blk.0.attn_q_norm.weight');
    assert.equal(mapHfTensorName('model.layers.0.self_attn.k_norm.weight', 'qwen2'), 'blk.0.attn_k_norm.weight');
  });
});

describe('adaptHfRepo', () => {
  const config = {
    model_type: 'llama',
    num_hidden_layers: 2,
    hidden_size: 64,
    intermediate_size: 256,
    num_attention_heads: 8,
    num_key_value_heads: 4,
    max_position_embeddings: 1024,
    vocab_size: 320,
    rms_norm_eps: 1e-6,
    rope_theta: 10000,
  };

  it('produces a model object compatible with analyzeModel output', () => {
    const tensors = llamaTensors(2, 64, 256, 320);
    const model = adaptHfRepo({ config, tensors, repoId: 'test/llama', analyzeModel });

    assert.equal(model.arch, 'llama');
    assert.equal(model.blockCount, 2);
    assert.equal(model.embeddingLength, 64);
    assert.equal(model.headCount, 8);
    assert.equal(model.headCountKV, 4);
    assert.equal(model.gqaRatio, 2);
    assert.equal(model.contextLength, 1024);
    assert.equal(model.vocabSize, 320);
    assert.equal(model.ffnHiddenDim, 256);
    assert.equal(model.normType, 'RMSNorm');
    assert.ok(/SiLU|SwiGLU/.test(model.activationFunction), `activation should be SiLU/SwiGLU, got ${model.activationFunction}`);
  });

  it('builds a layer tree with embedding, blocks, output', () => {
    const tensors = llamaTensors(2, 64, 256, 320);
    const model = adaptHfRepo({ config, tensors, repoId: 'test/llama', analyzeModel });

    const layerTypes = model.layers.map(l => l.type);
    assert.deepEqual(layerTypes, ['embedding', 'block', 'block', 'output']);

    const block0 = model.layers.find(l => l.type === 'block' && l.index === 0);
    assert.ok(block0, 'block 0 exists');
    const cats = new Set(block0.subgroups.map(s => s.label.toLowerCase()));
    assert.ok(cats.has('attention'), 'has attention subgroup');
    assert.ok(cats.has('mlp'),       'has mlp subgroup');
    assert.ok(cats.has('norm'),      'has norm subgroup');
  });

  it('computes total parameter count from tensor shapes', () => {
    const tensors = llamaTensors(2, 64, 256, 320);
    const model = adaptHfRepo({ config, tensors, repoId: 'test/llama', analyzeModel });
    const expected = tensors.reduce((s, t) => s + t.numElements, 0);
    assert.equal(model.totalParams, expected);
  });

  it('computes nonzero memory bytes for known dtypes', () => {
    const tensors = llamaTensors(1, 32, 64, 128);
    const model = adaptHfRepo({ config: { ...config, num_hidden_layers: 1 }, tensors, repoId: 'x/y', analyzeModel });
    assert.ok(model.totalMemory > 0, 'totalMemory should be > 0 for F16/F32 tensors');
  });

  it('maps mistral and qwen2 model_types correctly', () => {
    const tensors = llamaTensors(1, 32, 64, 128);
    const m1 = adaptHfRepo({ config: { ...config, model_type: 'mistral', num_hidden_layers: 1 }, tensors, repoId: 'x', analyzeModel });
    assert.equal(m1.arch, 'llama'); // mistral aliases to llama in our table

    const m2 = adaptHfRepo({ config: { ...config, model_type: 'qwen2', num_hidden_layers: 1 }, tensors, repoId: 'x', analyzeModel });
    assert.equal(m2.arch, 'qwen2');
  });

  it('throws on missing config or tensors', () => {
    assert.throws(() => adaptHfRepo({ tensors: [], analyzeModel }), /config/i);
    assert.throws(() => adaptHfRepo({ config: {}, analyzeModel }), /tensors/i);
    assert.throws(() => adaptHfRepo({ config: {}, tensors: [] }), /analyzeModel/i);
  });

  it('preserves source descriptor for diagnostics', () => {
    const tensors = llamaTensors(1, 32, 64, 128);
    const src = { kind: 'hf-repo', repoId: 'a/b', revision: 'abc' };
    const model = adaptHfRepo({ config: { ...config, num_hidden_layers: 1 }, tensors, repoId: 'a/b', source: src, analyzeModel });
    assert.equal(model.ggufSource?.kind, 'hf-repo');
    assert.equal(model.ggufSource?.revision, 'abc');
  });
});
