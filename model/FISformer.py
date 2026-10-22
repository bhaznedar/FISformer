import torch
import torch.nn as nn

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import AttentionLayer
from layers.FIS_Interaction import FISInteraction
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """
    FISformer: Transformer encoder in which self-attention is replaced by the
    FIS Interaction layer (first-order Sugeno fuzzy inference over query/key pairs).

    Variate tokenization (each variate is one token) and the embedding /
    projection modules follow iTransformer (Liu et al., ICLR 2024).

    Paper: B. Haznedar and L. Karacan, IEEE Trans. Fuzzy Systems, 34(8), 2437-2450, 2026.
           doi: 10.1109/TFUZZ.2026.3690012
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # Embedding: B L N -> B N E
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        head_dim = configs.d_model // configs.n_heads
        num_rules = getattr(configs, 'num_rules', 3)
        membership_type = getattr(configs, 'membershipType', 'gaussian')
        record_maps = bool(getattr(configs, 'save_fis_maps', False))

        # Encoder with FIS Interaction blocks
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FISInteraction(False, configs.factor, attention_dropout=configs.dropout,
                                       output_attention=configs.output_attention,
                                       membershipType=membership_type, input_dim=head_dim,
                                       num_rules=num_rules, num_head=configs.n_heads,
                                       record_maps=record_maps),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Instance normalization (Non-stationary Transformer)
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N

        enc_out = self.enc_embedding(x_enc, x_mark_enc)          # B L N -> B N E
        enc_out, attns = self.encoder(enc_out, attn_mask=None)   # B N E -> B N E
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # B N E -> B S N

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, S, N]

    def fis_layers(self):
        """Return the FISInteraction modules (one per encoder layer)."""
        return [m for m in self.modules() if isinstance(m, FISInteraction)]
