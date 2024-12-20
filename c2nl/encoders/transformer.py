"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn
import torch

from c2nl.modules.util_class import LayerNorm, AverageSelfAttention
from c2nl.modules.multi_head_attn import MultiHeadedAttention
from c2nl.modules.position_ffn import PositionwiseFeedForward, CLSPositionwiseFeedForward
from c2nl.encoders.encoder import EncoderBase
from c2nl.utils.misc import sequence_mask
import torch.backends.cuda
import torch.backends.cudnn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.
    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self,
                 d_model,
                 heads,
                 d_ff,
                 d_k,
                 d_v,
                 dropout,
                 max_relative_positions=0,
                 use_neg_dist=True):
        super(TransformerEncoderLayer, self).__init__()

        self.attention = MultiHeadedAttention(heads,
                                              d_model,
                                              d_k,
                                              d_v,
                                              dropout=dropout,
                                              max_relative_positions=max_relative_positions,
                                              use_neg_dist=use_neg_dist)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, inputs, mask):
        """
        Transformer Encoder Layer definition.
        Args:
            inputs (`FloatTensor`): `[batch_size x src_len x model_dim]`
            mask (`LongTensor`): `[batch_size x src_len x src_len]`
        Returns:
            (`FloatTensor`):
            * outputs `[batch_size x src_len x model_dim]`
        """
        context, attn_per_head, _ = self.attention(inputs, inputs, inputs,
                                                   mask=mask, attn_type="self")
        out = self.layer_norm(self.dropout(context) + inputs)
        return self.feed_forward(out), attn_per_head


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
          :class:`MultiHeadedAttention`, also the input size of
          the first-layer of the :class:`PositionwiseFeedForward`.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the :class:`PositionwiseFeedForward`.
      dropout (float): dropout probability.
    """

    def __init__(self,
                 d_model,
                 heads,
                 d_k,
                 d_v,
                 d_ff,
                 dropout,
                 max_relative_positions=0,
                 coverage_attn=False):
        super(TransformerDecoderLayer, self).__init__()

        self.attention = MultiHeadedAttention(
            heads, d_model, d_k, d_v, dropout=dropout,
            max_relative_positions=max_relative_positions)
        self.layer_norm = LayerNorm(d_model)

        self.context_attn = MultiHeadedAttention(
            heads, d_model, d_k, d_v, dropout=dropout,
            coverage=coverage_attn)
        self.layer_norm_2 = LayerNorm(d_model)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self,
                inputs,
                memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=None,
                step=None,
                coverage=None):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, 1, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (LongTensor): ``(batch_size, 1, src_len)``
            tgt_pad_mask (LongTensor): ``(batch_size, 1, 1)``
        Returns:
            (FloatTensor, FloatTensor):
            * output ``(batch_size, 1, model_dim)``
            * attn ``(batch_size, 1, src_len)``
        """
        dec_mask = None
        if step is None:
            tgt_len = tgt_pad_mask.size(-1)
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8)
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)

        query, _, _ = self.attention(inputs,
                                     inputs,
                                     inputs,
                                     mask=dec_mask,
                                     layer_cache=layer_cache,
                                     attn_type="self")
        query_norm = self.layer_norm(self.drop(query) + inputs)

        mid, attn, coverage = self.context_attn(memory_bank,
                                                memory_bank,
                                                query_norm,
                                                mask=src_pad_mask,
                                                layer_cache=layer_cache,
                                                attn_type="context",
                                                step=step,
                                                coverage=coverage)
        mid_norm = self.layer_norm_2(self.drop(mid) + query_norm)

        output = self.feed_forward(mid_norm)
        return output, attn, coverage


class TransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".
    .. mermaid::
       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O
    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the models
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
    Returns:
        (`FloatTensor`, `FloatTensor`):
        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self,
                 num_layers,
                 d_model=512,
                 heads=8,
                 d_k=64,
                 d_v=64,
                 d_ff=2048,
                 dropout=0.2,
                 max_relative_positions=0,
                 use_neg_dist=True,
                 coverage_attn=False,
                 gaussian=False,
                 auto_regressive=False,
                 mean_latent=False,
                 num_prop_layers=1,):
        super(TransformerEncoder, self).__init__()
        self.gaussian = gaussian
        self.auto_regressive = auto_regressive
        self.mean_latent = mean_latent
        self.num_layers = num_layers
        self.num_prop_layers = num_prop_layers
        if isinstance(max_relative_positions, int):
            max_relative_positions = [max_relative_positions] * self.num_layers
        assert len(max_relative_positions) == self.num_layers

        self.layer = nn.ModuleList(
            [TransformerEncoderLayer(d_model,
                                     heads,
                                     d_ff,
                                     d_k,
                                     d_v,
                                     dropout,
                                     max_relative_positions=max_relative_positions[i],
                                     use_neg_dist=use_neg_dist)
             for i in range(num_layers-num_prop_layers)])
        # self.propagate_layer = TransformerEncoderLayer(d_model,
        #                                             heads,
        #                                             d_ff,
        #                                             d_k,
        #                                             d_v,
        #                                             dropout,
        #                                             max_relative_positions=max_relative_positions[-1],
        #                                             use_neg_dist=use_neg_dist)
        self.propagate_layer = nn.ModuleList(
            [TransformerEncoderLayer(d_model,
                                        heads,
                                        d_ff,
                                        d_k,
                                        d_v,
                                        dropout,
                                        max_relative_positions=max_relative_positions[i+num_layers-num_prop_layers],
                                        use_neg_dist=use_neg_dist)
                for i in range(num_prop_layers)])

        self.layer_norm = LayerNorm(d_model)
        self.summ_encode_layer = TransformerEncoderLayer(d_model,
                                                    heads,
                                                    d_ff,
                                                    d_k,
                                                    d_v,
                                                    dropout,
                                                    max_relative_positions=max_relative_positions[-1],
                                                    use_neg_dist=use_neg_dist)
        self.dropout_1 = nn.Dropout(dropout)
        self.context_attn = MultiHeadedAttention(heads,
                                                 d_model,
                                                 d_k,
                                                 d_v,
                                                 dropout=dropout,
                                                 coverage=coverage_attn)
        self.dropout_2 = nn.Dropout(dropout)
        self.mean = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.logvar = PositionwiseFeedForward(d_model, d_ff, dropout)
        # self.output = nn.Linear(768, d_model)
        self.dropout_3 = nn.Dropout(dropout)
        self.averageSelfAttention = AverageSelfAttention(d_model)

    def count_parameters(self):
        params = list(self.layer.parameters())
        return sum(p.numel() for p in params if p.requires_grad)

    def forward(self, src, lengths=None, code_cls=None, summ_emb=None, summ_pad_mask=None, std_scale=1.):
        """
        Args:
            src (`FloatTensor`): `[batch_size x src_len x model_dim]`
            lengths (`LongTensor`): length of each sequence `[batch]`
            std_scale (`float`): scale of the standard deviation
        Returns:
            (`FloatTensor`):
            * outputs `[batch_size x src_len x model_dim]`
        """
        self._check_args(src, lengths)
        # print("len encoder layer, ", len(self.layer))
        # print("len prop layer, ", len(self.propagate_layer))
        out = src
        mask = None if lengths is None else \
            ~sequence_mask(lengths, out.shape[1]).unsqueeze(1)
        # Run the forward pass of every layer of the tranformer.
        representations = []
        attention_scores = []
        for i in range(self.num_layers-1):
            out, attn_per_head = self.layer[i](out, mask)
            representations.append(out)
            attention_scores.append(attn_per_head)

        memory_bank = representations[-1]

        if self.training:
            tgt_pad_mask = summ_pad_mask.unsqueeze(1)
            post_mask = tgt_pad_mask
            # print("using auto-regressive", self.auto_regressive)
            # print("using gaussian prior", self.gaussian)
            if self.auto_regressive:
                # print("using auto-regressive")
                tgt_len = tgt_pad_mask.size(-1)
                future_mask = torch.ones(
                    [tgt_len, tgt_len],
                    device=tgt_pad_mask.device,
                    dtype=torch.uint8)
                future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
                post_mask = torch.gt(tgt_pad_mask + future_mask, 0)
            query_norm = self.summ_encode_layer(summ_emb, post_mask)[0]
            mid = self.context_attn(query_norm,
                                    query_norm,
                                    memory_bank,
                                    mask=summ_pad_mask.unsqueeze(1),
                                    attn_type="context")[0]
            attended_memory_bank = self.layer_norm(self.dropout_2(mid) + memory_bank)
            attended_cls_tokens = attended_memory_bank[:, 0, :].unsqueeze(1)
            if self.mean_latent:
                # attended_tokens_mean = torch.mean(attended_memory_bank, dim=1).unsqueeze(1)
                attended_tokens_mean = self.averageSelfAttention(attended_memory_bank, mask.squeeze(1))[0].unsqueeze(1)
                mu = self.mean(attended_tokens_mean)
                logvar = self.logvar(attended_tokens_mean)
            else:
                # print("using cls latent")
                mu = self.mean(attended_cls_tokens)
                logvar = self.logvar(attended_cls_tokens)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std).to(std.device)
            z = eps.mul(std).add_(mu)
        else:

            z = torch.empty_like(code_cls).normal_(mean=0,
                                                   std=1 * std_scale).to(memory_bank.device)
            mu = torch.zeros_like(code_cls).to(code_cls.device)
            if not self.gaussian:
                z += code_cls
                mu += code_cls
            logvar = torch.zeros_like(code_cls).to(code_cls.device)

        # out, attn_per_head = self.propagate_layer(torch.cat((z, memory_bank[:, 1:, :]), dim=1), mask)
        #
        # representations.append(out)
        # attention_scores.append(attn_per_head)
        if self.mean_latent:
            out = memory_bank + z
            for i in range(self.num_prop_layers):
                out, attn_per_head = self.propagate_layer[i](out, mask)
                representations.append(out)
                attention_scores.append(attn_per_head)
        else:
            out = torch.cat((z, memory_bank[:, 1:, :]), dim=1)
            for i in range(self.num_prop_layers):
                out, attn_per_head = self.propagate_layer[i](out, mask)
                representations.append(out)
                attention_scores.append(attn_per_head)

        return representations, attention_scores, mu, logvar

