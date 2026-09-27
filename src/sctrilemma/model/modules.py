from typing import Literal, Optional

import torch
import torch.nn as nn

GeneEmbeddingMode = Literal["learnable"]


class AttentionPooling(nn.Module):
    """Attention pooling over latent tokens."""

    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        out, _ = self.attn(query, x, x)
        return out.squeeze(1)


class GeneEmbeddingNetwork(nn.Module):
    """Learnable gene identity embedding used by the active VAE backbone."""

    def __init__(
        self,
        d_model: int = 512,
        vocab_size: int = 60000,
        embedding_mode: GeneEmbeddingMode = "learnable",
        **_: object,
    ):
        super().__init__()
        if embedding_mode != "learnable":
            raise ValueError(
                "Clean scTrilemma keeps only learnable gene embeddings; "
                f"got embedding_mode={embedding_mode!r}."
            )
        self.d_model = d_model
        self.embedding_mode = embedding_mode
        self.gene_id_embedding = nn.Embedding(vocab_size, d_model)

    def forward(
        self,
        gene_indices: Optional[torch.Tensor] = None,
        **_: object,
    ) -> torch.Tensor:
        if gene_indices is None:
            raise ValueError("gene_indices is required for learnable gene embeddings.")
        return self.gene_id_embedding(gene_indices)
