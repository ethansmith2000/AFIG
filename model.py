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


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(32, outchannel),
            nn.SiLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(32, outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.silu(out)

        return out


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
                                    scale=self.scale,
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

    def forward(self, x, context):
        x = self.self_attn(x, x)
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
        preds = self.top_k_sampling(preds, topk)
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
            x = layer(x, x)

        x = self.final_norm(x)
        x = self.proj_out(x)

        x = self.head(x)

        return x


class FFTDecoderMixtureWithCovariance(FFTDecoderBase):

    def __init__(self,
                 query_dim=768,
                 in_channels=6,
                 heads=8,
                 dropout=0.0,
                 ff_mult=2,
                 num_layers=12,
                 ctx_len=1000,
                 num_gaussians=20,
                 num_freqs=20,
                 ):
        super().__init__()

        # these will be done outside model forward
        self.fourier_embedder = FourierEmbedder(num_freqs, 100)

        self.in_proj = nn.Linear(num_freqs * in_channels * 2, query_dim)
        self.in_norm = nn.LayerNorm(query_dim)

        self.in_mlp = nn.Sequential(
            nn.Linear(query_dim, query_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 2, query_dim),
        )

        self.mean_dim = query_dim // 2
        self.cov_dim = query_dim // 4
        self.mix_dim = query_dim // 4

        self.layers = nn.ModuleList([TransformerLayer(query_dim=query_dim,
                                                      context_dim=query_dim,
                                                      heads=heads,
                                                      dropout=dropout,
                                                      ff_mult=ff_mult,
                                                      ) for _ in range(num_layers)])
        self.gradient_checkpointing = False

        self.positional_embeddings = nn.Embedding(ctx_len, query_dim)
        self.bos_embedding = nn.Parameter(torch.randn(1, 1, query_dim))

        self.mean_proj_out = nn.Linear(self.mean_dim, num_gaussians * in_channels)
        self.mean_final_norm = nn.LayerNorm(self.mean_dim)

        self.mix_proj_out = nn.Linear(self.mix_dim, num_gaussians)
        self.mix_final_norm = nn.LayerNorm(self.mix_dim)

        self.cov_proj = nn.Linear(self.cov_dim, 576)
        self.cov_norm = nn.LayerNorm(self.cov_dim)

        self.cov_norm_2 = nn.LayerNorm(36)

        self.register_buffer("positive_definite_constant", torch.tensor(2e-4))

        class MLP(nn.Module):
            def __init__(self, in_dim, out_dim, dropout=0.0):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, in_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(in_dim * 2, out_dim),
                    nn.Dropout(dropout),
                )
                self.norm = nn.LayerNorm(in_dim)

            def forward(self, x):
                return x + self.net(self.norm(x))

        self.cov_network = nn.ModuleList(
            [MLP(576, 576, dropout=dropout) for _ in range(2)]
        )

        self.cov_proj_out = nn.Linear(16, num_gaussians)


    def sample_gmm(self, means, covs, mix_probs, z=None, topk=3, scale=1.0):
        # b, s, num_gaussians, 6
        # b, s, num_gaussians, 6, 6
        # b, s, num_gaussians

        b, s, num_gaussians = mix_probs.shape[:3]

        with torch.autocast("cuda", enabled=True, dtype=torch.float32):
            # modulo phase values
            means = self.modulo_phase(means)

            # sample one of the gaussians
            topk = topk if topk is not None else num_gaussians
            idx = self.topk_sample(mix_probs, topk)

            # gather the means, covs, and mix_probs
            means = torch.gather(means, dim=-2, index=idx[:, :, :, None].expand(-1, -1, -1, 6)).squeeze(-2)
            covs = torch.gather(covs, dim=-3, index=idx[:, :, :, None, None].expand(-1, -1, -1, 6, 6)).squeeze(-3)


            L = torch.linalg.cholesky(covs.float())
            if z is None:
                z = torch.randn_like(means) * scale

            # do the sample
            samples = means[:, :, None, :] + z[:, :, None, :] @ L.permute(0, 1, 3, 2)

            samples = samples.squeeze(-2)

            samples = self.modulo_phase(samples)

            return samples

    def gen_sample(self, batch_size, sample_scale=1.0, sample_topk=3):
        device, weight_dtype = quick_get_device_and_dtype(self)

        start = torch.zeros(batch_size, 1, 6, device=device).to(weight_dtype)
        whole_sequence = start.clone().to(device).to(weight_dtype)

        i = 0
        while whole_sequence.shape[1] <= 544:
            mean_x, std_x, mix_x = self(whole_sequence)
            mean_x = mean_x[:, -1:]
            std_x = std_x[:, -1:]
            mix_x = mix_x[:, -1:]
            preds = self.sample_gmm(mean_x, std_x, mix_x, scale=sample_scale, topk=sample_topk)
            whole_sequence = torch.cat([whole_sequence, preds], dim=1)
            i += 1

        whole_sequence = whole_sequence[:, 1:]

        images = self.convert_to_image(whole_sequence)

        return whole_sequence, images


    def gmm_log_prob(self, means, covs, mix_probs, ground_truth):
        # b, s, num_gaussians, 6
        # b, s, num_gaussians, 6, 6
        # b, s, num_gaussians
        n_dim = 6

        with torch.autocast("cuda", enabled=True, dtype=torch.float32):
            means = means.float()
            covs = covs.float()
            mix_probs = mix_probs.float()
            ground_truth = ground_truth.float()

            log_mix_probs = F.log_softmax(mix_probs, dim=-1)

            # modulo phase values
            modulo_means = self.modulo_phase(means.clone())

            cov_inv = torch.inverse(covs)
            log_cov_det = torch.logdet(covs)

            # get difference
            modulo_diff = ground_truth[:, :, None, :] - modulo_means
            diff = ground_truth[:, :, None, :] - means

            # take lesser of the two
            diff = torch.where(torch.abs(diff) < torch.abs(modulo_diff), diff, modulo_diff)

            # normalized = False

            # if normalized:
            #     eps = 1e-6
            #     diff_norm = torch.norm(diff, dim=-1, keepdim=True)
            #     cov_inv_norm = torch.norm(cov_inv, dim=(-2, -1), keepdim=True)
            #     diff = diff / (diff_norm + eps)
            #     cov_inv = cov_inv / (cov_inv_norm + eps)

            multiplied = torch.matmul(diff.unsqueeze(-2), cov_inv)

            pre_summed = torch.matmul(multiplied, diff.unsqueeze(-1))

            mahalanobis_dist = torch.sum(pre_summed, dim=(-2, -1))

            # if normalized:
            #     mahalanobis_dist = mahalanobis_dist * (diff_norm.squeeze(-1) * cov_inv_norm.squeeze((-2, -1)))

        with torch.autocast("cuda", enabled=True, dtype=torch.float32):
            mahalanobis_dist = mahalanobis_dist.float()

            #  log probabilities
            log_probs = -0.5 * (n_dim * math.log(2 * math.pi) + log_cov_det + mahalanobis_dist)

            # weighted sum of log probs
            mixed_log_probs = torch.logsumexp(log_probs + log_mix_probs, dim=-1)

        return mixed_log_probs

    def forward(self, x):
        x = self.fourier_embedder(x)

        x = self.in_proj(x)
        x = x + self.in_mlp(self.in_norm(x))

        pos_idx = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        pos_emb = self.positional_embeddings(pos_idx)
        x = x + pos_emb

        for layer in self.layers:
            x = layer(x, x)

        # 448, 448, 128
        mean_x = x[:, :, :self.mean_dim]
        cov_x = x[:, :, self.mean_dim:self.mean_dim + self.cov_dim]
        mix_x = x[:, :, self.mean_dim + self.cov_dim:]

        mean_x = self.mean_final_norm(mean_x)
        mean_x = self.mean_proj_out(mean_x)

        mix_x = self.mix_final_norm(mix_x)
        mix_x = self.mix_proj_out(mix_x)

        #
        b, s = mean_x.shape[:2]
        cov_x = self.cov_norm(cov_x)
        cov_x = self.cov_proj(cov_x)
        # b, s, 576

        for layer in self.cov_network:
            cov_x = layer(cov_x)

        # b, s, 6, 6, dim -> b, s, 6, 6, num_gaussians
        cov_x = cov_x.reshape(b, s, 6, 6, -1)
        cov_x = self.cov_proj_out(cov_x)
        # b, s, 6, 6, num_gaussians -> b, s, num_gaussians, 6, 6
        cov_x = cov_x.permute(0, 1, 4, 2, 3)

        # norm
        cov_x = cov_x.reshape(b, s, -1, 36)
        cov_x = self.cov_norm_2(cov_x)
        cov_x = cov_x.reshape(b, s, -1, 6, 6)

        # numerical precision
        orig_dtype = cov_x.dtype

        with torch.autocast("cuda", enabled=True, dtype=torch.float32):
            cov_x = cov_x.float()

            # lower triangle
            cov_x = torch.tril(cov_x)

            mask = (torch.eye(6, device=cov_x.device).float()[None, None, None, :, :] > 0).expand_as(cov_x)
            cov_x[mask] = torch.nn.functional.softplus(cov_x[mask].float())

            cov_x = torch.matmul(cov_x.float(), cov_x.permute(0, 1, 2, 4, 3).float())

            positive_definite_constant = torch.eye(6, device=cov_x.device)[None, None, None, :,
                                         :] * self.positive_definite_constant

            cov_x = cov_x + positive_definite_constant

            # numerical stability for calculating logprobs
            # cov_x = cov_x.reshape(b, s, -1, 36)
            # cov_x = self.cov_norm_2(cov_x)
            # cov_x = cov_x.reshape(b, s, -1, 6, 6)

            # b, s, 6^2, 16
            # cov_x = cov_x.reshape(b, s, 36, -1)
            # cov_x = cov_x.reshape(b * s, 36, -1)

            # cov_x = self.cov_network_proj_in(cov_x)
            # for layer in self.cov_network:
            #     cov_x = layer(cov_x, cov_x)
            # cov_x = self.cov_network_proj_out(cov_x)

            # cov_x = cov_x.reshape(b, s, 6, 6, -1)

            # # b, s, 16, 6, 6
            # cov_x = cov_x.reshape(b, s, -1, 6, 6)
            # cov_x = cov_x.reshape(b * s, -1, 6, 6)

            mean_x = mean_x.reshape(b, s, -1, 6)

            return mean_x, cov_x, mix_x



class FFTDecoderMixtureUnrolled(FFTDecoderBase):

    def __init__(self,
                 query_dim=768,
                 in_channels=1,
                 heads=8,
                 dropout=0.0,
                 ff_mult=2,
                 num_layers=12,
                 ctx_len=1000,
                 num_gaussians=20,
                 num_freqs=20,
                 ):
        super().__init__()

        # these will be done outside model forward
        self.fourier_embedder = FourierEmbedder(num_freqs, 100)

        self.in_proj = nn.Linear(num_freqs * in_channels * 2, query_dim)
        self.in_norm = nn.LayerNorm(query_dim)

        self.in_mlp = nn.Sequential(
            nn.Linear(query_dim, query_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 2, query_dim),
        )

        self.mean_dim = query_dim // 2
        self.std_dim = query_dim // 4
        self.mix_dim = query_dim // 4

        self.layers = nn.ModuleList([TransformerLayer(query_dim=query_dim,
                                                      context_dim=query_dim,
                                                      heads=heads,
                                                      dropout=dropout,
                                                      ff_mult=ff_mult,
                                                      ) for _ in range(num_layers)])
        self.gradient_checkpointing = False

        self.positional_embeddings = nn.Embedding(ctx_len, query_dim)
        self.bos_embedding = nn.Parameter(torch.randn(1, 1, query_dim))

        self.mean_proj_out = nn.Linear(self.mean_dim, num_gaussians * in_channels)
        self.mean_final_norm = nn.LayerNorm(self.mean_dim)

        self.mix_proj_out = nn.Linear(self.mix_dim, num_gaussians)
        self.mix_final_norm = nn.LayerNorm(self.mix_dim)

        self.std_proj_out = nn.Linear(self.std_dim, num_gaussians * in_channels)
        self.std_norm = nn.LayerNorm(self.std_dim)


    def sample_gmm(self, means, stds, mix_probs, phase_mask, z=None, topk=3, scale=1.0):
        # b, s, num_gaussians
        b, s, num_gaussians = mix_probs.shape[:3]

        # modulo phase values
        means = self.modulo_phase(means, phase_mask)

        # sample one of the gaussians
        topk = topk if topk is not None else num_gaussians
        idx = self.topk_sample(mix_probs, topk)

        # gather the means, covs, and mix_probs
        means = torch.gather(means, dim=-1, index=idx)
        stds = torch.gather(stds, dim=-1, index=idx)

        # z is the random component
        if z is None:
            z = torch.randn_like(means) * scale

        samples = means + z * stds

        samples = self.modulo_phase(samples, phase_mask)

        return samples.squeeze(-1)

    def gmm_log_prob(self, means, stds, mix_probs, ground_truth, phase_mask):
        # b, s, num_gaussians
        mix_probs = F.softmax(mix_probs, dim=-1)

        # modulo phase values
        modulo_means = self.modulo_phase(means.clone(), phase_mask)

        # get difference
        modulo_diff = ground_truth[:, :, None] - modulo_means
        diff = ground_truth[:, :, None] - means

        # take lesser of the two
        diff = torch.where(torch.abs(diff) < torch.abs(modulo_diff), diff, modulo_diff)

        # log probs
        log_prob = -0.5 * (diff / stds) ** 2 - torch.log(stds) - 0.5 * torch.log(2 * torch.tensor(3.1415))
        log_probs = (log_prob + torch.log(mix_probs + 1e-8)).logsumexp(-1)

        return log_probs

    def gen_sample(self, batch_size, sample_scale=1.0, sample_topk=3):
        device, weight_dtype = quick_get_device_and_dtype(self)

        start = torch.zeros(batch_size, 1, device=device).to(weight_dtype)
        whole_sequence = start.clone().to(device).to(weight_dtype)

        phase_mask = [0, 0, 0, 1, 1, 1]
        phase_mask = torch.tensor(phase_mask, device=device).bool().repeat(544)
        i = 0
        while whole_sequence.shape[1] <= 3264:
            mean_x, std_x, mix_x = self(whole_sequence)
            mean_x = mean_x[:, -1:]
            std_x = std_x[:, -1:]
            mix_x = mix_x[:, -1:]
            preds = self.sample_gmm(mean_x, std_x, mix_x, phase_mask[i:i + 1], scale=sample_scale, topk=sample_topk)
            whole_sequence = torch.cat([whole_sequence, preds], dim=1)
            i += 1

        whole_sequence = whole_sequence[:, 1:]

        whole_sequence = whole_sequence.reshape(batch_size, whole_sequence.shape[1] // 6, 6)

        images = self.convert_to_image(whole_sequence)

        return whole_sequence, images

    def forward(self, x):

        x = self.fourier_embedder(x.unsqueeze(-1))

        x = self.in_proj(x)

        pos_idx = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        pos_emb = self.positional_embeddings(pos_idx)
        x = x + pos_emb

        x = x + self.in_mlp(self.in_norm(x))

        for layer in self.layers:
            x = layer(x, x)

        # 448, 448, 128
        mean_x = x[:, :, :self.mean_dim]
        std_x = x[:, :, self.mean_dim:self.mean_dim + self.std_dim]
        mix_x = x[:, :, self.mean_dim + self.std_dim:]

        mean_x = self.mean_final_norm(mean_x)
        mean_x = self.mean_proj_out(mean_x)

        mix_x = self.mix_final_norm(mix_x)
        mix_x = self.mix_proj_out(mix_x)

        b, s = mean_x.shape[:2]
        std_x = self.std_norm(std_x)
        std_x = self.std_proj_out(std_x)

        std_x = F.relu(std_x) + 1e-4

        return mean_x, std_x, mix_x