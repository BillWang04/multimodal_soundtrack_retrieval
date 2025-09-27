import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
import torchaudio
import numpy as np
class TFRep(nn.Module):
    """
    time-frequency represntation
    """
    def __init__(self, 
                sample_rate= 16000,
                f_min=0,
                f_max=8000,
                n_fft=1024,
                win_length=1024,
                hop_length = int(0.01 * 16000),
                n_mels = 128,
                power = None,
                pad= 0,
                normalized= False,
                center= True,
                pad_mode= "reflect"
                ):
        super(TFRep, self).__init__()
        self.window = torch.hann_window(win_length)
        self.spec_fn = torchaudio.transforms.Spectrogram(
            n_fft = n_fft,
            win_length = win_length,
            hop_length = hop_length,
            power = power
        )
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels, 
            sample_rate,
            f_min,
            f_max,
            n_fft // 2 + 1)
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def melspec(self, wav):
        spec = self.spec_fn(wav)
        power_spec = spec.real.abs().pow(2)
        mel_spec = self.mel_scale(power_spec)
        mel_spec = self.amplitude_to_db(mel_spec)
        return mel_spec

    def spec(self, wav):
        spec = self.spec_fn(wav)
        real = spec.real
        imag = spec.imag
        power_spec = real.abs().pow(2)
        eps = 1e-10
        mag = torch.clamp(mag ** 2 + phase ** 2, eps, np.inf) ** 0.5
        cos = real / mag
        sin = imag / mag
        return power_spec, imag, mag, cos, sin


class Res2DMaxPoolModule(nn.Module):
    def __init__(self, input_channels, output_channels, pooling=2):
        super(Res2DMaxPoolModule, self).__init__()
        self.conv_1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)

        # residual
        self.diff = False
        if input_channels != output_channels:
            self.conv_3 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
            self.bn_3 = nn.BatchNorm2d(output_channels)
            self.diff = True

    def forward(self, x):
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.mp(self.relu(out))
        return out
    
class ResFrontEnd(nn.Module):
    """
    After the convolution layers, we flatten the time-frequency representation to be a vector.
    mix_type : cf -> mix channel and frequency dim
    mix_type : ft -> mix frequency and time dim
    """
    def __init__(self, input_size ,conv_ndim, attention_ndim, mix_type="cf",nharmonics=1):
        super(ResFrontEnd, self).__init__()
        self.mix_type = mix_type
        self.input_bn = nn.BatchNorm2d(nharmonics)
        self.layer1 = Res2DMaxPoolModule(nharmonics, conv_ndim, pooling=(2, 2))
        self.layer2 = Res2DMaxPoolModule(conv_ndim, conv_ndim, pooling=(2, 2))
        self.layer3 = Res2DMaxPoolModule(conv_ndim, conv_ndim, pooling=(2, 2))
        self.layer4 = Res2DMaxPoolModule(conv_ndim, conv_ndim, pooling=(2, 2))
        F,T = input_size
        self.ntime = T // 2 // 2 // 2 // 2
        self.nfreq = F // 2 // 2 // 2 // 2
        if self.mix_type == "ft":
            self.fc_ndim = conv_ndim
        else:
            self.fc_ndim = self.nfreq * conv_ndim
        self.fc = nn.Linear(self.fc_ndim, attention_ndim)

    def forward(self, hcqt):
        # batch normalization
        out = self.input_bn(hcqt)
        # CNN
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # permute and channel control
        b, c, f, t = out.shape
        if self.mix_type == "ft":
            out = out.contiguous().view(b, c, -1)  # batch, channel, tf_dim
            out = out.permute(0,2,1) # batch x length x dim
        else:
            out = out.permute(0, 3, 1, 2)  # batch, time, conv_ndim, freq
            out = out.contiguous().view(b, t, -1)  # batch, length, hidden
        out = self.fc(out)  # batch, time, attention_ndim
        return out
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                            )
                        ),
                        Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class MusicTransformer(nn.Module):
    def __init__(self,
                audio_representation,
                frontend, 
                audio_rep,
                is_vq=False, 
                dropout=0.1, 
                attention_ndim=256,
                attention_nheads=8,
                attention_nlayers=4,
                attention_max_len=512
        ):
        super(MusicTransformer, self).__init__()
        # Input preprocessing
        self.audio_representation = audio_representation
        self.audio_rep = audio_rep
        # Input embedding
        self.frontend = frontend
        self.is_vq = is_vq
        self.vq_modules = None
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, attention_max_len + 1, attention_ndim))
        self.cls_token = nn.Parameter(torch.randn(attention_ndim))
        # transformer
        self.transformer = Transformer(
            attention_ndim,
            attention_nlayers,
            attention_nheads,
            attention_ndim // attention_nheads,
            attention_ndim * 4,
            dropout,
        )
        self.to_latent = nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.audio_rep == "mel":
            spec = self.audio_representation.melspec(x)
            spec = spec.unsqueeze(1)
        elif self.audio_rep == "stft":
            spec = None
        h_audio = self.frontend(spec) # B x L x D
        if self.is_vq:
            h_audio = self.vq_modules(h_audio)
        cls_token = self.cls_token.repeat(h_audio.shape[0], 1, 1)
        h_audio = torch.cat((cls_token, h_audio), dim=1)
        h_audio += self.pos_embedding[:, : h_audio.size(1)]
        h_audio = self.dropout(h_audio)
        z_audio = self.transformer(h_audio)
        return z_audio
