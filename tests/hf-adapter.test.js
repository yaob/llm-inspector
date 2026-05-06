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

  it('maps Phi3 fused projections', () => {
    assert.equal(mapHfTensorName('model.layers.0.self_attn.qkv_proj.weight', 'phi3'), 'blk.0.attn_qkv.weight');
    assert.equal(mapHfTensorName('model.layers.0.self_attn.o_proj.weight', 'phi3'), 'blk.0.attn_output.weight');
    assert.equal(mapHfTensorName('model.layers.5.mlp.gate_up_proj.weight', 'phi3'), 'blk.5.ffn_up.weight');
    assert.equal(mapHfTensorName('model.layers.5.mlp.down_proj.weight', 'phi3'), 'blk.5.ffn_down.weight');
    assert.equal(mapHfTensorName('model.layers.0.input_layernorm.weight', 'phi3'), 'blk.0.attn_norm.weight');
  });

  it('maps Mixtral block_sparse_moe tensors per expert', () => {
    assert.equal(mapHfTensorName('model.layers.0.block_sparse_moe.gate.weight', 'mixtral'), 'blk.0.ffn_gate_inp.weight');
    assert.equal(mapHfTensorName('model.layers.3.block_sparse_moe.experts.0.w1.weight', 'mixtral'), 'blk.3.ffn_gate_exp.0.weight');
    assert.equal(mapHfTensorName('model.layers.3.block_sparse_moe.experts.7.w2.weight', 'mixtral'), 'blk.3.ffn_down_exp.7.weight');
    assert.equal(mapHfTensorName('model.layers.3.block_sparse_moe.experts.7.w3.weight', 'mixtral'), 'blk.3.ffn_up_exp.7.weight');
  });

  it('still maps non-MoE tensors for Mixtral', () => {
    assert.equal(mapHfTensorName('model.layers.0.self_attn.q_proj.weight', 'mixtral'), 'blk.0.attn_q.weight');
    assert.equal(mapHfTensorName('model.layers.0.input_layernorm.weight', 'mixtral'), 'blk.0.attn_norm.weight');
  });

  it('maps Mamba backbone tensors', () => {
    assert.equal(mapHfTensorName('backbone.layers.0.norm.weight', 'mamba'), 'blk.0.attn_norm.weight');
    assert.equal(mapHfTensorName('backbone.layers.0.mixer.in_proj.weight', 'mamba'), 'blk.0.ssm_in.weight');
    assert.equal(mapHfTensorName('backbone.layers.0.mixer.conv1d.weight', 'mamba'), 'blk.0.ssm_conv1d.weight');
    assert.equal(mapHfTensorName('backbone.layers.0.mixer.conv1d.bias', 'mamba'), 'blk.0.ssm_conv1d.bias');
    assert.equal(mapHfTensorName('backbone.layers.0.mixer.x_proj.weight', 'mamba'), 'blk.0.ssm_x.weight');
    assert.equal(mapHfTensorName('backbone.layers.0.mixer.dt_proj.weight', 'mamba'), 'blk.0.ssm_dt.weight');
    assert.equal(mapHfTensorName('backbone.layers.0.mixer.out_proj.weight', 'mamba'), 'blk.0.ssm_out.weight');
    assert.equal(mapHfTensorName('backbone.layers.0.mixer.A_log', 'mamba'), 'blk.0.ssm_a.weight');
    assert.equal(mapHfTensorName('backbone.layers.0.mixer.D', 'mamba'), 'blk.0.ssm_d.weight');
    assert.equal(mapHfTensorName('backbone.embeddings.weight', 'mamba'), 'token_embd.weight');
    assert.equal(mapHfTensorName('backbone.norm_f.weight', 'mamba'), 'output_norm.weight');
  });

  it('preserves bias on Mamba conv1d', () => {
    assert.equal(mapHfTensorName('backbone.layers.3.mixer.conv1d.bias', 'mamba'), 'blk.3.ssm_conv1d.bias');
  });

  it('maps Qwen3 tensors using the LLaMA-family pattern', () => {
    assert.equal(mapHfTensorName('model.layers.0.self_attn.q_proj.weight', 'qwen3'), 'blk.0.attn_q.weight');
    assert.equal(mapHfTensorName('model.layers.0.self_attn.q_norm.weight', 'qwen3'), 'blk.0.attn_q_norm.weight');
    assert.equal(mapHfTensorName('model.layers.0.mlp.gate_proj.weight', 'qwen3'), 'blk.0.ffn_gate.weight');
  });

  it('maps Gemma2 sandwich norms', () => {
    assert.equal(mapHfTensorName('model.layers.0.pre_feedforward_layernorm.weight', 'gemma2'), 'blk.0.ffn_norm.weight');
    assert.equal(mapHfTensorName('model.layers.0.post_feedforward_layernorm.weight', 'gemma2'), 'blk.0.ffn_norm_2.weight');
  });

  it('falls back to underscore-joined names for an unknown arch on block tensors', () => {
    assert.equal(
      mapHfTensorName('model.layers.0.self_attn.q_proj.weight', 'unknown_arch'),
      'blk.0.self_attn_q_proj.weight'
    );
  });

  it('passes through top-level tensors that have no mapping rule', () => {
    assert.equal(mapHfTensorName('something.else.weight', 'llama'), 'something.else.weight');
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

  it('groups Mixtral per-expert tensors under the moe category', () => {
    const numExperts = 4;
    const hidden = 32, ffn = 64, vocab = 128;
    const tensors = [
      { name: 'model.embed_tokens.weight', dtype: 'F16', shape: [vocab, hidden], dataOffsets: [0, 0], numElements: vocab * hidden, byteLength: vocab * hidden * 2, ggmlType: 1 },
      { name: 'model.norm.weight',         dtype: 'F32', shape: [hidden],         dataOffsets: [0, 0], numElements: hidden, byteLength: hidden * 4, ggmlType: 0 },
      { name: 'lm_head.weight',            dtype: 'F16', shape: [vocab, hidden], dataOffsets: [0, 0], numElements: vocab * hidden, byteLength: vocab * hidden * 2, ggmlType: 1 },
      { name: 'model.layers.0.input_layernorm.weight',          dtype: 'F32', shape: [hidden], dataOffsets: [0, 0], numElements: hidden, byteLength: hidden * 4, ggmlType: 0 },
      { name: 'model.layers.0.post_attention_layernorm.weight', dtype: 'F32', shape: [hidden], dataOffsets: [0, 0], numElements: hidden, byteLength: hidden * 4, ggmlType: 0 },
      { name: 'model.layers.0.self_attn.q_proj.weight',         dtype: 'F16', shape: [hidden, hidden], dataOffsets: [0, 0], numElements: hidden * hidden, byteLength: hidden * hidden * 2, ggmlType: 1 },
      { name: 'model.layers.0.self_attn.k_proj.weight',         dtype: 'F16', shape: [hidden, hidden], dataOffsets: [0, 0], numElements: hidden * hidden, byteLength: hidden * hidden * 2, ggmlType: 1 },
      { name: 'model.layers.0.self_attn.v_proj.weight',         dtype: 'F16', shape: [hidden, hidden], dataOffsets: [0, 0], numElements: hidden * hidden, byteLength: hidden * hidden * 2, ggmlType: 1 },
      { name: 'model.layers.0.self_attn.o_proj.weight',         dtype: 'F16', shape: [hidden, hidden], dataOffsets: [0, 0], numElements: hidden * hidden, byteLength: hidden * hidden * 2, ggmlType: 1 },
      { name: 'model.layers.0.block_sparse_moe.gate.weight',    dtype: 'F32', shape: [numExperts, hidden], dataOffsets: [0, 0], numElements: numExperts * hidden, byteLength: numExperts * hidden * 4, ggmlType: 0 },
    ];
    for (let e = 0; e < numExperts; e++) {
      tensors.push({ name: `model.layers.0.block_sparse_moe.experts.${e}.w1.weight`, dtype: 'F16', shape: [ffn, hidden], dataOffsets: [0, 0], numElements: ffn * hidden, byteLength: ffn * hidden * 2, ggmlType: 1 });
      tensors.push({ name: `model.layers.0.block_sparse_moe.experts.${e}.w2.weight`, dtype: 'F16', shape: [hidden, ffn], dataOffsets: [0, 0], numElements: hidden * ffn, byteLength: hidden * ffn * 2, ggmlType: 1 });
      tensors.push({ name: `model.layers.0.block_sparse_moe.experts.${e}.w3.weight`, dtype: 'F16', shape: [ffn, hidden], dataOffsets: [0, 0], numElements: ffn * hidden, byteLength: ffn * hidden * 2, ggmlType: 1 });
    }
    const mixtralConfig = {
      model_type: 'mixtral', num_hidden_layers: 1,
      hidden_size: hidden, intermediate_size: ffn,
      num_attention_heads: 4, num_key_value_heads: 2,
      max_position_embeddings: 1024, vocab_size: vocab,
      num_local_experts: numExperts, num_experts_per_tok: 2,
    };
    const model = adaptHfRepo({ config: mixtralConfig, tensors, repoId: 'x/mixtral', analyzeModel });
    assert.equal(model.arch, 'mixtral');
    const block0 = model.layers.find(l => l.type === 'block' && l.index === 0);
    const moeGroup = block0.subgroups.find(s => s.label.toLowerCase() === 'moe');
    assert.ok(moeGroup, 'block has a moe subgroup');
    // gate (router) + 3 expert tensors per expert
    assert.equal(moeGroup.tensors.length, 1 + numExperts * 3);
    assert.equal(model.metadata['mixtral.expert_count'], numExperts);
    assert.equal(model.metadata['mixtral.expert_used_count'], 2);
  });

  it('produces a Mamba model with ssm subgroup', () => {
    const hidden = 16, vocab = 64, layers = 1;
    const tensors = [
      { name: 'backbone.embeddings.weight', dtype: 'F32', shape: [vocab, hidden], dataOffsets: [0, 0], numElements: vocab * hidden, byteLength: vocab * hidden * 4, ggmlType: 0 },
      { name: 'backbone.norm_f.weight',     dtype: 'F32', shape: [hidden],         dataOffsets: [0, 0], numElements: hidden, byteLength: hidden * 4, ggmlType: 0 },
      { name: 'lm_head.weight',             dtype: 'F32', shape: [vocab, hidden], dataOffsets: [0, 0], numElements: vocab * hidden, byteLength: vocab * hidden * 4, ggmlType: 0 },
    ];
    for (let i = 0; i < layers; i++) {
      tensors.push({ name: `backbone.layers.${i}.norm.weight`,           dtype: 'F32', shape: [hidden], dataOffsets: [0, 0], numElements: hidden, byteLength: hidden * 4, ggmlType: 0 });
      tensors.push({ name: `backbone.layers.${i}.mixer.in_proj.weight`,  dtype: 'F32', shape: [hidden * 2, hidden], dataOffsets: [0, 0], numElements: hidden * 2 * hidden, byteLength: hidden * 2 * hidden * 4, ggmlType: 0 });
      tensors.push({ name: `backbone.layers.${i}.mixer.out_proj.weight`, dtype: 'F32', shape: [hidden, hidden],     dataOffsets: [0, 0], numElements: hidden * hidden, byteLength: hidden * hidden * 4, ggmlType: 0 });
    }
    const mambaConfig = { model_type: 'mamba', num_hidden_layers: layers, hidden_size: hidden, vocab_size: vocab, state_size: 16, conv_kernel: 4 };
    const model = adaptHfRepo({ config: mambaConfig, tensors, repoId: 'x/mamba', analyzeModel });
    assert.equal(model.arch, 'mamba');
    assert.equal(model.metadata['mamba.ssm.state_size'], 16);
    assert.equal(model.metadata['mamba.ssm.conv_kernel'], 4);
    const block0 = model.layers.find(l => l.type === 'block' && l.index === 0);
    const ssmGroup = block0.subgroups.find(s => s.label.toLowerCase() === 'ssm');
    assert.ok(ssmGroup, 'block has an ssm subgroup');
  });

  it('produces a Phi3 model with fused qkv and gate_up grouped under attn/ffn', () => {
    const hidden = 32, ffn = 64, vocab = 128, layers = 1;
    const tensors = [
      { name: 'model.embed_tokens.weight', dtype: 'F16', shape: [vocab, hidden], dataOffsets: [0, 0], numElements: vocab * hidden, byteLength: vocab * hidden * 2, ggmlType: 1 },
      { name: 'model.norm.weight',         dtype: 'F32', shape: [hidden],         dataOffsets: [0, 0], numElements: hidden, byteLength: hidden * 4, ggmlType: 0 },
      { name: 'lm_head.weight',            dtype: 'F16', shape: [vocab, hidden], dataOffsets: [0, 0], numElements: vocab * hidden, byteLength: vocab * hidden * 2, ggmlType: 1 },
    ];
    for (let i = 0; i < layers; i++) {
      tensors.push({ name: `model.layers.${i}.input_layernorm.weight`,          dtype: 'F32', shape: [hidden], dataOffsets: [0, 0], numElements: hidden, byteLength: hidden * 4, ggmlType: 0 });
      tensors.push({ name: `model.layers.${i}.post_attention_layernorm.weight`, dtype: 'F32', shape: [hidden], dataOffsets: [0, 0], numElements: hidden, byteLength: hidden * 4, ggmlType: 0 });
      tensors.push({ name: `model.layers.${i}.self_attn.qkv_proj.weight`,       dtype: 'F16', shape: [3 * hidden, hidden], dataOffsets: [0, 0], numElements: 3 * hidden * hidden, byteLength: 3 * hidden * hidden * 2, ggmlType: 1 });
      tensors.push({ name: `model.layers.${i}.self_attn.o_proj.weight`,         dtype: 'F16', shape: [hidden, hidden], dataOffsets: [0, 0], numElements: hidden * hidden, byteLength: hidden * hidden * 2, ggmlType: 1 });
      tensors.push({ name: `model.layers.${i}.mlp.gate_up_proj.weight`,         dtype: 'F16', shape: [2 * ffn, hidden], dataOffsets: [0, 0], numElements: 2 * ffn * hidden, byteLength: 2 * ffn * hidden * 2, ggmlType: 1 });
      tensors.push({ name: `model.layers.${i}.mlp.down_proj.weight`,            dtype: 'F16', shape: [hidden, ffn], dataOffsets: [0, 0], numElements: hidden * ffn, byteLength: hidden * ffn * 2, ggmlType: 1 });
    }
    const phi3Config = {
      model_type: 'phi3', num_hidden_layers: layers,
      hidden_size: hidden, intermediate_size: ffn,
      num_attention_heads: 4, num_key_value_heads: 4,
      max_position_embeddings: 2048, vocab_size: vocab,
      rms_norm_eps: 1e-5,
    };
    const model = adaptHfRepo({ config: phi3Config, tensors, repoId: 'x/phi3', analyzeModel });
    assert.equal(model.arch, 'phi3');
    const block0 = model.layers.find(l => l.type === 'block' && l.index === 0);
    const labels = block0.subgroups.map(s => s.label.toLowerCase());
    assert.ok(labels.some(l => l.includes('attn') || l.includes('attention')), `expected an attention subgroup, got ${labels}`);
    assert.ok(labels.some(l => l.includes('ffn') || l.includes('mlp') || l.includes('feed')), `expected a feedforward subgroup, got ${labels}`);
    // The fused qkv tensor must end up inside the block, mapped to attn_qkv.
    const allBlockTensorNames = block0.subgroups.flatMap(s => s.tensors.map(t => t.name));
    assert.ok(allBlockTensorNames.includes('blk.0.attn_qkv.weight'), 'fused attn_qkv should be present');
    assert.ok(allBlockTensorNames.includes('blk.0.ffn_up.weight'), 'fused gate_up_proj should map to ffn_up');
  });

  it('falls back to the input model_type when no mapping exists', () => {
    const tensors = llamaTensors(1, 32, 64, 128);
    const model = adaptHfRepo({
      config: { ...config, model_type: 'totally_made_up', num_hidden_layers: 1 },
      tensors, repoId: 'x/y', analyzeModel,
    });
    assert.equal(model.arch, 'totally_made_up');
    assert.equal(model.metadata['general.architecture'], 'totally_made_up');
  });
});
