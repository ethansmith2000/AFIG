import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from utils import get_2d_freqs_from_1d, inverse_fft
from tqdm import tqdm

def quick_get_device_and_dtype(model):
    device = next(model.parameters()).device
    weight_dtype = next(model.parameters()).dtype
    return device, weight_dtype


class Attention(nn.Module):
    def __init__(self, query_dim=768, context_dim=1024,
                 heads=8, dropout=0.0,
                 ):
        super().__init__()
        self.to_qkv = nn.Linear(query_dim, query_dim * 3, bias=False)
        self.heads = heads
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.norm = nn.LayerNorm(query_dim)

    def batch_to_head_dim(self, tensor):
        batch_size, heads, seq_len, dim = tensor.shape
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, dim * self.heads)
        return tensor

    def head_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, self.heads, dim // self.heads)
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    def forward(self, x):
        b, n, _ = x.shape

        resid_x = x

        norm_x = self.norm(x)
        q, k, v = self.to_qkv(norm_x).chunk(3, dim=-1)
        q = self.head_to_batch_dim(q)
        k = self.head_to_batch_dim(k)
        v = self.head_to_batch_dim(v)

        attn_output = F.scaled_dot_product_attention(q, k, v, #attn_mask=
                                    is_causal=True,
                                    #dropout_p=self.dropout
                                    )

        attn_output = self.batch_to_head_dim(attn_output)

        attn_output = self.out_proj(attn_output)

        x = resid_x + attn_output

        return x


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.net(self.norm(x))


class TransformerLayer(nn.Module):

    def __init__(self, query_dim=768, context_dim=1024,
                 heads=8, dropout=0.0, ff_mult=4, use_cross_attn=False):

        super().__init__()
        self.self_attn = Attention(query_dim=query_dim,
                 context_dim=context_dim,
                 heads=heads,
                 dropout=dropout,)
        if use_cross_attn:
            self.cross_attn = Attention(query_dim=query_dim,
                    context_dim=context_dim,
                    heads=heads,
                    dropout=dropout,)
        else:
            self.cross_attn = None

        self.ff = FeedForward(query_dim, mult=ff_mult, dropout=dropout)
        self.gradient_checkpointing = False

    def forward(self, x, context=None):
        x = self.self_attn(x)
        x = self.ff(x)
        return x

class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs, temperature):
        super().__init__()
        self.num_freqs = num_freqs
        self.temperature = temperature
        freq_bands = temperature ** (torch.arange(num_freqs, dtype=torch.float32) / num_freqs)
        self.register_buffer("freq_bands", freq_bands)

    @torch.no_grad()
    def forward(self, x, cat_dim=-1):
        """
        :param x: arbitrary shape of tensor
        :param cat_dim: cat dim
        """
        out = []
        freq_bands = self.freq_bands.to(dtype=x.dtype)
        for freq in freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, cat_dim)




############################################################################################################
class FFTDecoderBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True
        for layer in self.layers:
            layer.gradient_checkpointing = True


    def modulo_phase(self, x, phase_mask=None):
        if phase_mask is not None:
            # assume the version without covariance method
            x[:, phase_mask] = torch.where(x[:, phase_mask] > 3.1415, -3.1415 + (x[:, phase_mask] - 3.1415),
                                           x[:, phase_mask])
            x[:, phase_mask] = torch.where(x[:, phase_mask] < -3.1415, 3.1415 + (x[:, phase_mask] + 3.1415),
                                           x[:, phase_mask])
        else:
            if len(x.shape) == 4:
                x[:, :, :, 3:] = torch.where(x[:, :, :, 3:] > 3.1415,
                                                        -3.1415 + (x[:, :, :, 3:] - 3.1415), x[:, :, :, 3:])
                x[:, :, :, 3:] = torch.where(x[:, :, :, 3:] < -3.1415,
                                                        3.1415 + (x[:, :, :, 3:] + 3.1415), x[:, :, :, 3:])
            else:
                x[:, :, 3:] = torch.where(x[:, :, 3:] > 3.1415, -3.1415 + (x[:, :, 3:] - 3.1415), x[:, :, 3:])
                x[:, :, 3:] = torch.where(x[:, :, 3:] < -3.1415, 3.1415 + (x[:, :, 3:] + 3.1415), x[:, :, 3:])

        return x

    def topk_sample(self, logits, k):
        b, s = logits.shape[0], logits.shape[1]
        num_dims = len(logits.shape)

        if num_dims == 3:
            logits = logits.reshape(b * s, -1)

        # Find the top k logits and their indices for each sequence in the batch
        top_k_probs, top_k_indices = torch.topk(logits, k, dim=-1)

        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(top_k_probs, dim=-1)

        # Sample from the top k probabilities for each sequence in the batch
        next_word_indices = torch.multinomial(probabilities, 1)

        # Gather the selected indices from the top k indices
        batch_size = logits.size(0)
        selected_indices = torch.gather(top_k_indices, 1, next_word_indices)

        if num_dims == 3:
            selected_indices = selected_indices.reshape(b, s, 1)

        return selected_indices

    # Back-compat alias.
    def top_k_sampling(self, logits, k):
        return self.topk_sample(logits, k)


    def convert_to_image(self, whole_sequence):
        fft = get_2d_freqs_from_1d(whole_sequence, 32, 32, False).float()

        mag, angle = fft.chunk(2, dim=1)

        image = inverse_fft(mag, angle)

        image = image.cpu().numpy() * 255
        image = np.clip(image, 0, 255)
        image = image.transpose(0, 2, 3, 1).astype(np.uint8)
        images = [image[i] for i in range(image.shape[0])]
        images = [Image.fromarray(image) for image in images]

        return images


    @torch.no_grad()
    def gen_sample(self, batch_size, sample_topk=3):
        whole_sequence = None
        images = None

        return whole_sequence, images


    def forward(self, x):
        x = None

        return x


class FFTDecoderQuantized(FFTDecoderBase):

    def __init__(self,
                 query_dim=768,
                 in_channels=6,
                 heads=8,
                 dropout=0.0,
                 ff_mult=2,
                 num_layers=12,
                 ctx_len=4000,
                 vocab_size=8192,
                 ):
        super().__init__()

        self.proj_in = nn.Linear(query_dim, query_dim)
        self.in_norm = nn.LayerNorm(query_dim)

        self.vocab = torch.linspace(-7.5, 7.5, vocab_size)
        self.vocab[0] = -20
        self.vocab[1] = 20
        self.vocab = torch.nn.Parameter(self.vocab)
        # set bos and eos tokens, because we will be using distance for quantization, set to very large numbers

        self.layers = nn.ModuleList([TransformerLayer(query_dim=query_dim,
                                                      context_dim=query_dim,
                                                      heads=heads,
                                                      dropout=dropout,
                                                      ff_mult=ff_mult,
                                                      ) for _ in range(num_layers)])
        self.gradient_checkpointing = False
        self.embeddings = nn.Embedding(vocab_size, query_dim)
        self.positional_embeddings = nn.Embedding(ctx_len, query_dim)

        self.final_norm = nn.LayerNorm(query_dim)
        self.proj_out = nn.Linear(query_dim, query_dim)

        self.head = nn.Linear(query_dim, vocab_size)


    @torch.no_grad()
    def run_step(self, whole_sequence, topk=3):
        preds = self(whole_sequence)
        preds = preds[:, -1, :]
        preds = self.topk_sample(preds, topk)
        whole_sequence = torch.cat([whole_sequence, preds], dim=1)

        return whole_sequence


    @torch.no_grad()
    def gen_sample(self, batch_size, sample_topk=3):
        device, weight_dtype = quick_get_device_and_dtype(self)
        start = torch.zeros(batch_size, 1)
        whole_sequence = start.clone().to(device).long()

        i = 0
        progress_bar = tqdm(total=3264)
        while whole_sequence.shape[1] <= 3264:
            whole_sequence = self.run_step(whole_sequence, sample_topk)
            i += 1
            progress_bar.update(1)

        whole_sequence = whole_sequence[:, 1:]
        whole_sequence = self.vocab[whole_sequence]

        whole_sequence = whole_sequence.reshape(whole_sequence.shape[0], whole_sequence.shape[1] // 6, 6)

        images = self.convert_to_image(whole_sequence)

        return whole_sequence, images


    def forward(self, x):
        b, n = x.shape

        x = self.embeddings(x)

        pos_idx = torch.arange(n, device=x.device).unsqueeze(0)
        pos_emb = self.positional_embeddings(pos_idx)
        x = x + pos_emb

        x = self.proj_in(x)
        x = self.in_norm(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        x = self.proj_out(x)

        x = self.head(x)

        return x

