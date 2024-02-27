import torch
import torch.nn as nn
import ptwt
from einops import rearrange
from einops.layers.torch import Rearrange


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=None, bias=False, bn=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=bias)
        self.bn = nn.BatchNorm2d(c2) if bn is True else nn.Identity()
        self.act = act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        return self.act(self.bn(out))



class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Linear(inner_dim, dim) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dim_head)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim))

    def forward(self, x):
        x = self.attn(x) + x
        x = self.net(x) + x

        return self.norm(x)


class SWA(nn.Module):
    def __init__(self, size, dim):
        super().__init__()
        self.conv = Conv(dim, 2, 1, act=False, bn=False)
        
        self.ww = nn.Sequential(
            Conv(2, 1, (size, 5), p=(0,2), act=False, bn=False),
            Rearrange('b 1 1 s -> b s'),
            nn.Linear(size, size),
            nn.Sigmoid()
        )
        
        self.hw = nn.Sequential(
            Conv(2, 1, (5, size), p=(2,0), act=False, bn=False),
            Rearrange('b 1 s 1 -> b s'),
            nn.Linear(size, size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv(x)
        
        ww = self.ww(x).unsqueeze(1)
        hw = self.hw(x).unsqueeze(-1)

        return torch.cat([torch.mm(hw[i],ww[i]).unsqueeze(0) for i in range(x.shape[0])])


class WTA(nn.Module):
    def __init__(self, c, image_size, patch_size, h, hd, swa=True):
        super(WTA, self).__init__()
        
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.enswa=swa
        self.c = c
        dim = 4 * self.c
        self.lenth = image_size // patch_size
        num_patches = self.lenth ** 2
        patch_dim = dim * patch_size * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = nn.Sequential(
            Transformer(dim, h, hd, 4*dim),
            Transformer(dim, 1, dim, 4*dim),
        )

        self.swa = nn.Sequential(
            SWA(self.lenth, dim),
            Rearrange('b h w -> b (h w) 1', h=self.lenth)
        ) if self.enswa else nn.Identity()
        
        self.attention = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        dtype = x.dtype
        a, (b, c, d) = ptwt.wavedec2(x.float(), 'haar', 'zero', 1)
        y = torch.cat((a, b, c, d), 1).to(dtype)
        
        ti = self.to_patch_embedding(y)

        ti = self.transformer(ti)

        sw = self.swa(rearrange(ti, 'b (h w) c -> b c h w', h = self.lenth).contiguous()) if self.enswa else torch.ones((ti.shape), device=ti.device, dtype=ti.dtype)
    
        w = self.attention((ti * sw).mean(dim=1))
        y *= (w + 1).view(y.shape[0], y.shape[1], 1, 1)
        y = y.float()
        
        return ptwt.waverec2([y[:,:self.c], (y[:,self.c:self.c*2], y[:,self.c*2:self.c*3], y[:,self.c*3:])], 'haar').to(dtype)


class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x
