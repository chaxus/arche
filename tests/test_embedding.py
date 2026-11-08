import torch

from arche.embedding import TokenEmbedding


def test_embedding_shape():
    emb = TokenEmbedding(100, 64)
    ids = torch.tensor([[1, 2, 3, 4]])
    out = emb(ids)
    assert out.shape == (1, 4, 64)
