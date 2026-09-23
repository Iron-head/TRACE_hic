import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from trace_hic.model.succeed_hic import AttentionPool1D

class ConvBlock(nn.Module):
    def __init__(self, size, stride = 2, hidden_in = 64, hidden = 64):
        super(ConvBlock, self).__init__()
        pad_len = int(size / 2)
        self.scale = nn.Sequential(
                        nn.Conv1d(hidden_in, hidden, size, stride, pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        )
        self.res = nn.Sequential(
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        nn.ReLU(),
                        nn.Conv1d(hidden, hidden, size, padding = pad_len),
                        nn.BatchNorm1d(hidden),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        scaled = self.scale(x)
        identity = scaled
        res_out = self.res(scaled)
        out = self.relu(res_out + identity)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channel, output_size = 256, filter_size = 5, num_blocks = 12):
        super(Encoder, self).__init__()
        self.filter_size = filter_size
        self.conv_start = nn.Sequential(
                                    nn.Conv1d(in_channel, 32, 3, 2, 1),
                                    nn.BatchNorm1d(32),
                                    nn.ReLU(),
                                    )
        hiddens =        [32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]
        hidden_ins = [32, 32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256]
        self.res_blocks = self.get_res_blocks(num_blocks, hidden_ins, hiddens)
        self.conv_end = nn.Conv1d(256, output_size, 1)

    def forward(self, x):
        x = self.conv_start(x)
        x = self.res_blocks(x)
        out = self.conv_end(x)
        return out

    def get_res_blocks(self, n, his, hs):
        blocks = []
        for i, h, hi in zip(range(n), hs, his):
            blocks.append(ConvBlock(self.filter_size, hidden_in = hi, hidden = h))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks


model_epi = None


def set_succeed_model(model: nn.Module) -> None:
    """
    Inject a pretrained SUCCEED encoder model to be used by EncoderSplit.

    This is expected to be called by training scripts (e.g. `hic/training/main_alter.py`)
    before any model that relies on `EncoderSplit(..., succeed=True)` is instantiated.
    """
    global model_epi
    model_epi = model

class EncoderSplit(Encoder):
    def __init__(self, num_epi, seq_channels = 4, output_size = 256, filter_size = 5, num_blocks = 12,
                 succeed=True, preencoded_seq = False, use_genomic_features = True,
                 succeed_attention_pool = False, succeed_pool_size = 64,
                 succeed_pool_heads = 8):
        super(Encoder, self).__init__()
        self.filter_size = filter_size
        self.seq_channels = seq_channels
        self.preencoded_seq = preencoded_seq
        self.use_genomic_features = use_genomic_features
        self.succeed_attention_pool_enabled = bool(succeed_attention_pool)
        self.conv_start_seq = None
        self.res_blocks_seq = None
        self.seq_embed_proj = None
        self.succeed_attention_pool = None
        hiddens =        [32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256, 256]
        hidden_ins = [32, 32, 32, 32, 32, 64, 64, 128, 128, 128, 128, 256]
        hiddens_half = (np.array(hiddens) / 2).astype(int)
        hidden_ins_half = (np.array(hidden_ins) / 2).astype(int)

        if self.use_genomic_features:
            if num_epi <= 0:
                raise ValueError('num_epi must be positive when genomic features are enabled.')
            self.conv_start_epi = nn.Sequential(
                                        nn.Conv1d(num_epi, 16, 3, 2, 1),
                                        nn.BatchNorm1d(16),
                                        nn.ReLU(),
                                        )
            self.res_blocks_epi = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)
        else:
            self.conv_start_epi = None
            self.res_blocks_epi = None

        self.succeed = succeed
        self.use_succeed_model = self.succeed and (self.seq_channels == 4) and (not self.preencoded_seq)
        if self.succeed_attention_pool_enabled and not (
            self.use_succeed_model or self.preencoded_seq
        ):
            raise ValueError(
                'succeed_attention_pool requires raw SUCCEED DNA or native '
                '128-bp precomputed embeddings.'
            )
        seq_out = hiddens_half[-1]
        if self.preencoded_seq:
            seq_out = 256
            if self.seq_channels == seq_out:
                self.seq_embed_proj = nn.Identity()
            else:
                self.seq_embed_proj = nn.Conv1d(self.seq_channels, seq_out, 1)
        elif self.use_succeed_model:
            if model_epi is None:
                raise RuntimeError(
                    "`EncoderSplit(..., succeed=True)` requires a pretrained SUCCEED model. "
                    "Call `trace_hic.model.blocks.set_succeed_model(model)` before constructing the Hi-C model, "
                    "or disable SUCCEED via `succeed=False`."
                )
            self.succeed_model = model_epi
            # Freeze all parameters in `model_epi`
            for param in self.succeed_model.parameters():
                param.requires_grad = False
            seq_out = 256
            self.seq_embed_proj = nn.Conv1d(seq_out, seq_out, 1)
        else:
            self.conv_start_seq = nn.Sequential(
                                        nn.Conv1d(seq_channels, 16, 3, 2, 1),
                                        nn.BatchNorm1d(16),
                                        nn.ReLU(),
                                        )
            self.res_blocks_seq = self.get_res_blocks(num_blocks, hidden_ins_half, hiddens_half)

        # Both the live native SUCCEED path and the disk-cache path expose
        # [B, 256, 16384] native 128-bp embeddings.  Pool them here so the
        # downstream Hi-C model sees 256 tokens in either case.
        if self.succeed_attention_pool_enabled:
            self.succeed_attention_pool = AttentionPool1D(
                dim=seq_out,
                pool_size=succeed_pool_size,
                num_heads=succeed_pool_heads,
            )
        epi_out = hiddens_half[-1] if self.use_genomic_features else 0
        self.conv_end = nn.Conv1d(seq_out + epi_out, output_size, 1)

    def train(self, mode: bool = True):
        """Keep a frozen SUCCEED encoder in inference mode.

        Lightning/PyTorch recursively calls ``train()`` on all registered
        child modules.  The SUCCEED parameters are frozen, but its BatchNorm
        running statistics would still change if the encoder were left in
        training mode.  Re-apply ``eval()`` after the recursive call so that
        the pretrained encoder remains deterministic during Hi-C training.
        """
        super().train(mode)
        if self.use_succeed_model and hasattr(self, "succeed_model"):
            self.succeed_model.eval()
        return self


    def forward(
        self,
        x,
        sequence_conditioner=None,
        regulator_expression=None,
        native_global_module=None,
    ):

        if isinstance(x, tuple):
            seq = x[0].transpose(1, 2).contiguous()
            epi = x[1].transpose(1, 2).contiguous() if self.use_genomic_features else None
        else:
            seq = x if not self.use_genomic_features else x[:, :self.seq_channels, :]
            epi = x[:, self.seq_channels:, :] if self.use_genomic_features else None
        if self.preencoded_seq:
            seq = self.seq_embed_proj(seq)
        elif self.use_succeed_model:
            seq = self.succeed_model(seq)
            seq = self.seq_embed_proj(seq)
        else:
            seq = self.res_blocks_seq(self.conv_start_seq(seq))

        if sequence_conditioner is not None or native_global_module is not None:
            if self.succeed_attention_pool is None:
                raise ValueError(
                    'Native-token conditioning/global context requires '
                    'succeed_attention_pool so the sequence is known to be '
                    'at 128-bp resolution.'
                )
            if sequence_conditioner is not None and regulator_expression is None:
                raise ValueError(
                    'regulator_expression is required by sequence_conditioner.'
                )
            # The sequence encoder is channel-first [B,256,16384], whereas
            # token attention is batch-first [B,16384,256].  Conditioning is
            # deliberately applied before the 64-to-1 attention pool.
            seq_tokens = seq.transpose(1, 2).contiguous()
            if sequence_conditioner is not None:
                seq_tokens = sequence_conditioner(
                    seq_tokens, regulator_expression
                )
            if native_global_module is not None:
                seq_tokens = native_global_module(seq_tokens)
            seq = seq_tokens.transpose(1, 2).contiguous()

        if self.succeed_attention_pool is not None:
            seq = self.succeed_attention_pool(seq)

        if self.use_genomic_features:
            epi = self.res_blocks_epi(self.conv_start_epi(epi))
            x = torch.cat([seq, epi], dim = 1)
        else:
            x = seq
        out = self.conv_end(x)
        return out

class ResBlockDilated(nn.Module):
    def __init__(self, size, hidden = 64, stride = 1, dil = 2):
        super(ResBlockDilated, self).__init__()
        pad_len = dil
        self.res = nn.Sequential(
                        nn.Conv2d(hidden, hidden, size, padding = pad_len,
                            dilation = dil),
                        nn.BatchNorm2d(hidden),
                        nn.ReLU(),
                        nn.Conv2d(hidden, hidden, size, padding = pad_len,
                            dilation = dil),
                        nn.BatchNorm2d(hidden),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        res_out = self.res(x)
        out = self.relu(res_out + identity)
        return out

class Decoder(nn.Module):
    def __init__(self, in_channel, hidden = 256, filter_size = 3, num_blocks = 5):
        super(Decoder, self).__init__()
        self.filter_size = filter_size

        self.conv_start = nn.Sequential(
                                    nn.Conv2d(in_channel, hidden, 3, 1, 1),
                                    nn.BatchNorm2d(hidden),
                                    nn.ReLU(),
                                    )
        self.res_blocks = self.get_res_blocks(num_blocks, hidden)
        self.conv_end = nn.Conv2d(hidden, 1, 1)

    def forward(self, x):
        x = self.conv_start(x)
        x = self.res_blocks(x)
        out = self.conv_end(x)
        return out

    def get_res_blocks(self, n, hidden):
        blocks = []
        for i in range(n):
            dilation = 2 ** (i + 1)
            blocks.append(ResBlockDilated(self.filter_size, hidden = hidden, dil = dilation))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks

class TransformerLayer(torch.nn.TransformerEncoderLayer):
    # Pre-LN structure

    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        # MHA section
        src_norm = self.norm1(src)
        src_side, attn_weights = self.self_attn(src_norm, src_norm, src_norm,
                                    attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src_side)

        # MLP section
        src_norm = self.norm2(src)
        src_side = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(src_side)
        return src, attn_weights

class TransformerEncoder(torch.nn.TransformerEncoder):

    def __init__(self, encoder_layer, num_layers, norm=None, record_attn = False):
        super(TransformerEncoder, self).__init__(encoder_layer, num_layers)
        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.record_attn = record_attn

    def forward(self, src, mask = None, src_key_padding_mask = None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        attn_weight_list = []

        for mod in self.layers:
            output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attn_weight_list.append(attn_weights.unsqueeze(0).detach())
        if self.norm is not None:
            output = self.norm(output)

        if self.record_attn:
            return output, torch.cat(attn_weight_list)
        else:
            return output

    def _get_clones(self, module, N):
        return torch.nn.modules.ModuleList([copy.deepcopy(module) for i in range(N)])

class PositionalEncoding(nn.Module):

    def __init__(self, hidden, dropout = 0.1, max_len = 256):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden, 2) * (-np.log(10000.0) / hidden))
        # AttnModule uses batch_first=True, so keep the positional table in
        # [1, token, hidden] order.  The previous [token, 1, hidden] table
        # broadcast over the batch dimension and therefore assigned a
        # position to each sample instead of to each of its 256 tokens.
        pe = torch.zeros(1, max_len, hidden)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if x.ndim != 3:
            raise ValueError(f'Expected [batch, tokens, hidden], got {tuple(x.shape)}')
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f'Got {x.size(1)} tokens, but positional encoding supports '
                f'at most {self.pe.size(1)}.'
            )
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class AttnModule(nn.Module):
    def __init__(self, hidden = 128, layers = 8, record_attn = False, inpu_dim = 256):
        super(AttnModule, self).__init__()

        self.record_attn = record_attn
        self.pos_encoder = PositionalEncoding(hidden, dropout = 0.1)
        encoder_layers = TransformerLayer(hidden,
                                          nhead = 8,
                                          dropout = 0.1,
                                          dim_feedforward = 512,
                                          batch_first = True)
        self.module = TransformerEncoder(encoder_layers,
                                         layers,
                                         record_attn = record_attn)

    def forward(self, x):
        x = self.pos_encoder(x)
        output = self.module(x)
        return output

    def inference(self, x):
        return self.module(x)

if __name__ == '__main__':
    main()
