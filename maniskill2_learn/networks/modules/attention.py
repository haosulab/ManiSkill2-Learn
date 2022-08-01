import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from maniskill2_learn.utils.meta import Registry, build_from_cfg

ATTENTION_LAYERS = Registry("attention layer")


def compute_attention(score, v, dropout=None, mask=None):
    """
    :param score: [B, NH, NQ, NK]
    :param v: Value [B, NH, NK, E]
    :param mask: [B, NQ, NK]
    :param dropout:
    :return: [B, NH, NQ, E]
    """
    if mask is not None:
        mask = mask[:, None]
        score = score * mask + (-1e8) * (1 - mask)
    score = F.softmax(score, dim=-1)  # [B, NH, NQ, NK]
    if dropout is not None:
        score = dropout(score)
    return torch.einsum("bnij,bnjk->bnik", score, v)  # [B, NH, NQ, E]


class MultiHeadedAttentionBase(nn.Module):
    def __init__(self, embed_dim, num_heads, latent_dim, dropout=None):
        """
        :param embed_dim: The dimension of feature in each entity.
        :param num_heads: The number of attention heads.
        :param latent_dim:
        :param dropout:
        """
        super().__init__()
        self.sqrt_latent_dim = np.sqrt(latent_dim)
        self.w_k = nn.Parameter(torch.empty(num_heads, embed_dim, latent_dim))
        self.w_v = nn.Parameter(torch.empty(num_heads, embed_dim, latent_dim))
        self.w_o = nn.Parameter(torch.empty(num_heads, latent_dim, embed_dim))
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.w_k)
        nn.init.xavier_normal_(self.w_v)
        nn.init.xavier_normal_(self.w_o)
        if hasattr(self, "q"):  # For pooling
            nn.init.xavier_normal_(self.q)
        if hasattr(self, "w_q"):  # For self-attention
            nn.init.xavier_normal_(self.w_q)
        if hasattr(self, "w_kr"):  # For self-attention xl
            nn.init.xavier_normal_(self.w_kr)

    def get_atten_score(self, x, *args, **kwargs):
        raise NotImplementedError


@ATTENTION_LAYERS.register_module()
class AttentionPooling(MultiHeadedAttentionBase):
    def __init__(self, embed_dim, num_heads, latent_dim, dropout=None):
        super().__init__(embed_dim, num_heads, latent_dim, dropout)
        self.q = nn.Parameter(torch.empty(num_heads, 1, latent_dim))
        self._reset_parameters()

    def get_atten_score(self, x):
        k = torch.einsum("blj,njd->bnld", x, self.w_k)  # [B, NH, N, EL]
        score = torch.einsum("nij,bnkj->bnik", self.q, k) / self.sqrt_latent_dim  # [B, NH, 1, NK]
        return score

    def forward(self, x, mask=None, *args, **kwargs):
        """
        :param x: [B, N, E] [batch size, length, embed_dim] the input to the layer, a tensor of shape
        :param mask: [B, 1, N] [batch size, 1, length]
        :return: [B, E] [batch_size, embed_dim] one feature with size
        """
        # print(x.shape, self.w_v.shape)
        # exit(0)
        v = torch.einsum("blj,njd->bnld", x, self.w_v)  # [B, NH, N, EL]
        score = self.get_atten_score(x)
        out = compute_attention(score, v, self.dropout, mask)
        out = torch.einsum("bnlj,njk->blk", out, self.w_o)  # [B, 1, E]
        out = out[:, 0]
        return out


@ATTENTION_LAYERS.register_module()
class MultiHeadAttention(MultiHeadedAttentionBase):
    """
    Attention is all you need:
        https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, embed_dim, num_heads, latent_dim, dropout=None):
        super().__init__(embed_dim, num_heads, latent_dim, dropout)
        self.w_q = nn.Parameter(torch.empty(num_heads, embed_dim, latent_dim))
        self._reset_parameters()

    def get_atten_score(self, key, query):
        q = torch.einsum("blj,njd->bnld", query, self.w_q)  # [B, NH, NQ, EL]
        k = torch.einsum("blj,njd->bnld", key, self.w_k)  # [B, NH, NK, EL]
        score = torch.einsum("bnij,bnkj->bnik", q, k) / self.sqrt_latent_dim  # [B, NH, NQ, NK]
        return score

    def forward(self, key, query, mask=None, *args, **kwargs):
        """
        :param key: [B, NK, E] [batch size, length of keys, embed_dim] the input to the layer, a tensor of shape
        :param query: [B, NQ, E] [batch size, length of queries, embed_dim] the input to the layer, a tensor of shape
        :param mask: [B, NQ, NK] [batch size, length of keys, length of queries]
        :return: [B, NQ, E] [batch_size, length, embed_dim] Features after self attention
        """
        score = self.get_atten_score(key, query)
        v = torch.einsum("blj,njd->bnld", key, self.w_v)  # [B, NH, NK, EL]
        out = compute_attention(score, v, self.dropout, mask)  # [B, NH, NQ, E]
        out = torch.einsum("bnlj,njk->blk", out, self.w_o)  # [B, NQ, E]
        out = self.dropout(out)
        return out


@ATTENTION_LAYERS.register_module()
class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, embed_dim, num_heads, latent_dim, dropout=None):
        super(MultiHeadSelfAttention, self).__init__(embed_dim, num_heads, latent_dim, dropout)

    def forward(self, x, mask=None, *args, **kwargs):
        """
        :param x: [B, N, E] [batch size, length, embed_dim] the input to the layer, a tensor of shape
        :param mask: [B, N, N] [batch size, length, length]
        :return: [B, N, E] [batch_size, length, embed_dim] Features after self attention
        """
        return super(MultiHeadSelfAttention, self).forward(x, x, mask, *args, **kwargs)


@ATTENTION_LAYERS.register_module()
class MultiHeadSelfAttentionXL(MultiHeadedAttentionBase):
    """
    Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
        https://arxiv.org/pdf/1901.02860.pdf
    """

    def __init__(self, embed_dim, num_heads, latent_dim, u, v, dropout=None):
        super().__init__(embed_dim, num_heads, latent_dim, dropout)
        self.num_heads = num_heads
        self.w_q = nn.Parameter(torch.empty(num_heads, embed_dim, latent_dim))  # [NH, E, EL]
        self.w_kr = nn.Parameter(torch.empty(num_heads, embed_dim, latent_dim))  # [NH, E, EL]
        self.u, self.v = u, v  # [NH, EL]
        assert self.u.shape == (num_heads, latent_dim) == self.v.shape

        # pe or positional encoding: pe(pos, 2i) = sin(pos / (10000^(2i / E)))
        assert embed_dim % 2 == 0
        self.inv_freq = nn.Parameter(1 / ((1e4) ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim)), requires_grad=False)  # [E / 2]
        self._reset_parameters()

    def get_atten_score(self, x, history=None, all_x=None):
        if all_x is None:
            all_x = x if history is None else torch.cat([history, x], dim=1)  # [B, H + N, EL]
        q = torch.einsum("blj,njd->bnld", x, self.w_q)  # [B, NH, N, EL]

        BD_q = q + self.v[:, None]
        AC_q = q + self.u[:, None]

        AC_k = torch.einsum("blj,njd->bnld", all_x, self.w_k)  # [B, NH, N + H, EL]
        AC = torch.einsum("bnij,bnkj->bnik", AC_q, AC_k) / self.sqrt_latent_dim  # [B, NH, NQ, NK]

        B = x.shape[0]  # B
        N = x.shape[1]  # N
        HpN = all_x.shape[1]  # H + N

        # print('BNH+N', B, N, HpN)
        freq = (torch.arange(HpN + N, device=x.device)[:, None] - HpN + 1) * self.inv_freq  # [H + 2N, E / 2]   from -N - H + 1 to N
        rel_embed = torch.cat([freq.sin(), freq.cos()], dim=-1)  # [H + 2N, E]

        BD_k = torch.einsum("lj,njd->nld", rel_embed, self.w_kr)  # [NH, H + 2N, EL]
        BD = torch.einsum("bnij,nkj->bnik", BD_q, BD_k) / self.sqrt_latent_dim  # [B, NH, N, H + 2N]
        BD = BD.view(B, self.num_heads, -1)  # [B, NH, N * (H + 2N)]
        # target_size = N * (HpN + N - 1)

        # BD = torch.cat([BD, torch.zeros_like(BD[..., :-N])], dim=-1)
        BD = BD[..., :-N].view(B, self.num_heads, N, HpN + N - 1)
        BD = BD[..., N - 1 :]  # [B, NH, N, H + N]

        score = AC + BD  # [B, NH, N, N+H]
        return score

    def forward(self, x, mask=None, history=None, *args, **kwargs):
        """
        :param x: [B, N, E] [batch size, length, embed_dim] the input to the layer, a tensor of shape
        :param mask: [B, N, N] [batch size, length, length]
        :param history: [B, H, E] [batch size, length, length]
        :return: [B, N, E] [batch_size, length, embed_dim] Features after self attention
        """
        all_x = x if history is None else torch.cat([history, x], dim=1)  # [B, H + N, EL]
        score = self.get_atten_score(x, history, all_x)
        v = torch.einsum("blj,njd->bnld", all_x, self.w_v)  # [B, NH, H + N, EL]
        out = compute_attention(score, v, self.dropout, mask)  # [B, NH, N, E]
        out = torch.einsum("bnlj,njk->blk", out, self.w_o)  # [B, N, E]
        out = self.dropout(out)
        return out


def build_attention_layer(cfg, default_args=None):
    return build_from_cfg(cfg, ATTENTION_LAYERS, default_args)
