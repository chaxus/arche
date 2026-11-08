import torch

from arche.config import Llama32Config
from arche.load_weights import hf_to_arche_key, load_hf_state_dict_into_llama
from arche.model import Llama


def _tiny() -> Llama:
    cfg = Llama32Config(
        name="tiny",
        vocab_size=32,
        num_layers=1,
        embed_dim=16,
        intermediate_dim=32,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=16,
        rope_theta=10_000.0,
        rope_scale_factor=1.0,
        norm_eps=1e-5,
        tie_word_embeddings=True,
    )
    return Llama(cfg)


def test_hf_key_mapping():
    assert hf_to_arche_key("model.embed_tokens.weight") == "embed_tokens.weight"
    assert hf_to_arche_key("model.norm.weight") == "norm.weight"
    assert hf_to_arche_key("model.layers.0.self_attn.q_proj.weight") == (
        "layers.0.self_attn.q_proj.weight"
    )
    assert hf_to_arche_key("lm_head.weight") == "lm_head.weight"
    assert hf_to_arche_key("something.else") is None


def test_load_fake_hf_state_dict():
    model = _tiny()
    # Build a HF-shaped dict from the randomly initialized arche weights.
    src = model.state_dict()
    hf = {
        "model.embed_tokens.weight": src["embed_tokens.weight"].clone(),
        "model.norm.weight": src["norm.weight"].clone(),
        "model.layers.0.input_layernorm.weight": src["layers.0.input_layernorm.weight"].clone(),
        "model.layers.0.post_attention_layernorm.weight": src[
            "layers.0.post_attention_layernorm.weight"
        ].clone(),
        "model.layers.0.self_attn.q_proj.weight": src["layers.0.self_attn.q_proj.weight"].clone(),
        "model.layers.0.self_attn.k_proj.weight": src["layers.0.self_attn.k_proj.weight"].clone(),
        "model.layers.0.self_attn.v_proj.weight": src["layers.0.self_attn.v_proj.weight"].clone(),
        "model.layers.0.self_attn.o_proj.weight": src["layers.0.self_attn.o_proj.weight"].clone(),
        "model.layers.0.mlp.gate_proj.weight": src["layers.0.mlp.gate_proj.weight"].clone(),
        "model.layers.0.mlp.up_proj.weight": src["layers.0.mlp.up_proj.weight"].clone(),
        "model.layers.0.mlp.down_proj.weight": src["layers.0.mlp.down_proj.weight"].clone(),
        "lm_head.weight": src["embed_tokens.weight"].clone(),
    }
    other = _tiny()
    with torch.no_grad():
        for p in other.parameters():
            p.zero_()
    missing, unexpected = load_hf_state_dict_into_llama(other, hf, strict=True)
    assert missing == []
    assert unexpected == []
    assert torch.equal(other.embed_tokens.weight, model.embed_tokens.weight)
