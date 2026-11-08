from arche import llama32_1b, llama32_3b


def test_llama32_1b_matches_public_card():
    cfg = llama32_1b()
    assert cfg.vocab_size == 128_256
    assert cfg.num_layers == 16
    assert cfg.embed_dim == 2048
    assert cfg.intermediate_dim == 8192
    assert cfg.num_heads == 32
    assert cfg.num_kv_heads == 8
    assert cfg.head_dim == 64
    assert cfg.num_query_groups == 4
    assert cfg.rope_theta == 500_000.0
    assert cfg.tie_word_embeddings is True


def test_llama32_3b_head_dim():
    cfg = llama32_3b()
    assert cfg.embed_dim == cfg.num_heads * cfg.head_dim
    assert cfg.num_heads % cfg.num_kv_heads == 0
