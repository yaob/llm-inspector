/**
 * Hugging Face repo adapter.
 *
 * Converts a `config.json` plus a flat list of safetensors tensor entries
 * (from one or more shards) into the same GGUF-shaped object that
 * src/parsers/gguf.js produces, so analyzeModel() can render it unchanged.
 *
 * Phase 1 covers LLaMA-family architectures (Llama, Mistral, Qwen2);
 * unknown architectures fall through with a passthrough name mapping so
 * the user still gets a structural view, just without component labels.
 */

import { GGML_TYPE_NAMES } from '../parsers/gguf.js';

/** Map HF model_type to GGUF arch. Falls back to the input string. */
const MODEL_TYPE_TO_ARCH = {
  llama: 'llama',
  mistral: 'llama',          // identical tensor layout
  qwen2: 'qwen2',
  qwen2_moe: 'qwen2moe',
  gemma: 'gemma',
  gemma2: 'gemma2',
  phi3: 'phi3',
  internlm2: 'internlm2',
  deepseek: 'deepseek',
  deepseek_v2: 'deepseek2',
  starcoder2: 'starcoder2',
};

/** Architectures whose tensor naming is the standard LLaMA pattern. */
const LLAMA_FAMILY = new Set(['llama', 'qwen2', 'gemma', 'gemma2', 'internlm2', 'deepseek', 'starcoder2', 'mistral']);

/**
 * Map an HF tensor name to its GGUF-standard equivalent.
 * Returns the input unchanged if no rule matches.
 */
export function mapHfTensorName(name, arch) {
  // Block-scoped tensors: model.layers.<N>.<rest>
  const m = name.match(/^model\.layers\.(\d+)\.(.+?)(?:\.(weight|bias))?$/);
  if (m) {
    const [, idx, sub, suffix] = m;
    const dotSuffix = suffix ? `.${suffix}` : '.weight';
    const mapped = mapBlockSubname(sub, arch);
    return `blk.${idx}.${mapped}${dotSuffix}`;
  }
  // Top-level tensors
  const tail = name.replace(/\.(weight|bias)$/, '');
  const suffix = name.endsWith('.bias') ? '.bias' : '.weight';
  switch (tail) {
    case 'model.embed_tokens':       return `token_embd${suffix}`;
    case 'model.norm':                return `output_norm${suffix}`;
    case 'lm_head':                   return `output${suffix}`;
    case 'model.embed_positions':    return `pos_embd${suffix}`;
    default: return name;
  }
}

function mapBlockSubname(sub, arch) {
  if (LLAMA_FAMILY.has(arch)) {
    switch (sub) {
      case 'self_attn.q_proj':        return 'attn_q';
      case 'self_attn.k_proj':        return 'attn_k';
      case 'self_attn.v_proj':        return 'attn_v';
      case 'self_attn.o_proj':        return 'attn_output';
      case 'self_attn.q_norm':        return 'attn_q_norm';
      case 'self_attn.k_norm':        return 'attn_k_norm';
      case 'input_layernorm':         return 'attn_norm';
      case 'post_attention_layernorm':return 'ffn_norm';
      case 'pre_feedforward_layernorm':  return 'ffn_norm';      // gemma2
      case 'post_feedforward_layernorm': return 'ffn_norm_2';    // gemma2
      case 'mlp.gate_proj':           return 'ffn_gate';
      case 'mlp.up_proj':             return 'ffn_up';
      case 'mlp.down_proj':           return 'ffn_down';
    }
  }
  return sub.replace(/\./g, '_');
}

/**
 * Build a GGUF-shaped { metadata, tensors, version, source, alignment, tensorDataStart }
 * from an HF config + safetensors tensor list, then return the analyzed model.
 *
 * @param {object} args
 * @param {object} args.config            parsed config.json
 * @param {Array}  args.tensors           [{ name, dtype, shape, byteLength, numElements, ggmlType, dataOffsets, shard }, ...]
 * @param {string} [args.repoId]          "owner/repo" — used as model name fallback
 * @param {object} [args.source]          opaque source descriptor stored on the model
 * @param {Function} args.analyzeModel    injected analyzer (avoids a circular import in tests)
 */
export function adaptHfRepo({ config, tensors, repoId, source, analyzeModel }) {
  if (!config || typeof config !== 'object') throw new Error('config.json is missing or invalid');
  if (!Array.isArray(tensors)) throw new Error('tensors must be an array');
  if (typeof analyzeModel !== 'function') throw new Error('analyzeModel must be provided');

  const modelType = String(config.model_type || '').toLowerCase();
  const arch = MODEL_TYPE_TO_ARCH[modelType] || modelType || 'unknown';

  const blockCount = Number(config.num_hidden_layers ?? config.n_layer ?? 0);
  const embeddingLength = Number(config.hidden_size ?? config.n_embd ?? 0);
  const headCount = Number(config.num_attention_heads ?? config.n_head ?? 0);
  const headCountKV = Number(config.num_key_value_heads ?? headCount);
  const contextLength = Number(config.max_position_embeddings ?? config.n_positions ?? 0);
  const ffnHiddenDim = Number(config.intermediate_size ?? config.n_inner ?? 0);
  const vocabSize = Number(config.vocab_size ?? 0);

  const metadata = {
    'general.architecture': arch,
    'general.name': config._name_or_path || repoId || arch,
  };
  if (blockCount)        metadata[`${arch}.block_count`] = blockCount;
  if (embeddingLength)   metadata[`${arch}.embedding_length`] = embeddingLength;
  if (headCount)         metadata[`${arch}.attention.head_count`] = headCount;
  if (headCountKV)       metadata[`${arch}.attention.head_count_kv`] = headCountKV;
  if (contextLength)     metadata[`${arch}.context_length`] = contextLength;
  if (ffnHiddenDim)      metadata[`${arch}.feed_forward_length`] = ffnHiddenDim;
  if (config.rms_norm_eps != null)        metadata[`${arch}.attention.layer_norm_rms_epsilon`] = config.rms_norm_eps;
  else if (config.layer_norm_eps != null) metadata[`${arch}.attention.layer_norm_epsilon`] = config.layer_norm_eps;
  if (config.rope_theta != null)          metadata[`${arch}.rope.freq_base`] = config.rope_theta;
  if (config.sliding_window != null)      metadata[`${arch}.attention.sliding_window`] = config.sliding_window;
  // analyzeModel uses tokens.length for vocabSize; a sparse {length} avoids allocation.
  if (vocabSize > 0) metadata['tokenizer.ggml.tokens'] = { length: vocabSize };

  const ggufTensors = tensors.map(t => {
    const ggmlType = (t.ggmlType == null) ? -1 : t.ggmlType;
    const typeName = (t.ggmlType != null && GGML_TYPE_NAMES[t.ggmlType]) ? t.dtype : (t.dtype || 'unknown');
    return {
      name: mapHfTensorName(t.name, arch),
      dimensions: Array.isArray(t.shape) ? t.shape.slice() : [],
      type: ggmlType,
      typeName,
      numElements: Number(t.numElements) || 0,
      byteLength: Number(t.byteLength) || 0,
      offset: 0,
      hfName: t.name,
      hfDtype: t.dtype,
      shard: t.shard || null,
    };
  });

  const ggufLike = {
    version: 'safetensors',
    tensorCount: ggufTensors.length,
    metadata,
    tensors: ggufTensors,
    alignment: 1,
    tensorDataStart: 0,
    source: source || { kind: 'hf-repo', repoId },
  };

  return analyzeModel(ggufLike);
}
