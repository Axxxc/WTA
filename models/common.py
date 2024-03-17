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
        self.conv = Conv(dim, 2, 1, bn=False)
        
        self.ww = nn.Sequential(
            Conv(2, 1, (size, 5), p=(0,2), bn=False),
            Rearrange('b 1 1 s -> b s'),
            nn.Linear(size, size),
            nn.Sigmoid()
        )
        
        self.hw = nn.Sequential(
            Conv(2, 1, (5, size), p=(2,0), bn=False),
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
    def __init__(self, c, image_size, patch_size, h, hd):
        super(WTA, self).__init__()
        
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.c = c
        dim = 4 * self.c
        num_patches = (image_size // patch_size) ** 2
        patch_dim = dim * patch_size * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        self.transformer = nn.Sequential(
            Transformer(dim, h, hd, 4*dim),
            Transformer(dim, 1, dim, 4*dim),
        )
        
        self.attention = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        dtype = x.dtype
        a, (b, c, d) = ptwt.wavedec2(x.float(), 'haar', 'zero', 1)
        y = torch.cat((a, b, c, d), 1).to(dtype)
        
        ti = self.to_patch_embedding(y)

        ti += self.pos_embedding

        ti = self.transformer(ti)
    
        w = self.attention(ti.mean(dim = 1))
        y *= w.view(y.shape[0], y.shape[1], 1, 1)
        y = y.float()
        
        return ptwt.waverec2([y[:,:self.c], (y[:,self.c:self.c*2], y[:,self.c*2:self.c*3], y[:,self.c*3:])], 'haar').to(dtype)


class EDFA(nn.Module):
    def __init__(self, c, l):
        super(EDFA, self).__init__()
        assert l % 4 == 0, 'the input size of EDFA model should be the multiple of 4'
        dim = c // 2
        
        self.conv = Conv(c, dim, act=nn.ReLU())
        
        self.h = nn.Sequential(
            Rearrange('b c h w -> b w h c'),
            Conv(l, l//2, act=nn.ReLU()),
            Rearrange('b w h c -> b h w c'),
            Conv(l, l//2, act=nn.ReLU()),
            Rearrange('b h w c -> b c h w'),
            nn.Upsample(None, 2)
        )
        
        self.q = nn.Sequential(
            Rearrange('b c h w -> b w h c'),
            Conv(l, l//4, act=nn.ReLU()),
            Rearrange('b w h c -> b h w c'),
            Conv(l, l//4, act=nn.ReLU()),
            Rearrange('b h w c -> b c h w'),
            nn.Upsample(None, 4)
        )
        
        self.o = Conv(dim*3, dim, act=nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        out = torch.cat([x, self.h(x), self.q(x)], 1)
        
        return self.o(out)


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size =  max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(in_size, expand_size, bn=False, act=nn.ReLU()),
            Conv(expand_size, in_size, bn=False, act=nn.Hardsigmoid())
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = Conv(in_size, expand_size, act = act)
        self.conv2 = Conv(expand_size, expand_size, kernel_size, stride, g=expand_size, act = act)
        self.se = SeModule(expand_size) if se else nn.Identity()
        self.conv3 = Conv(expand_size, out_size)
        self.act = act

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = Conv(in_size, out_size)

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                Conv(in_size, in_size, 3, 2, g=in_size),
                Conv(in_size, out_size),
            )

        if stride == 2 and in_size == out_size:
            self.skip = Conv(in_size, out_size, 3, 2, g=in_size)

    def forward(self, x):
        skip = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = self.conv3(out)
        
        if self.skip is not None:
            skip = self.skip(skip)
        return self.act(out + skip)


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


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=nn.ReLU())
        self.cv2 = Conv(c1, c_, 1, 1, act=nn.ReLU())
        self.cv3 = Conv(2 * c_, c2, 1, act=nn.ReLU())  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=nn.ReLU())
        self.cv2 = Conv(c_, c2, 3, 1, g=g, act=nn.ReLU())
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):
        """
        Initializes YOLOv5 SPPF layer with given channels and kernel size for YOLOv5 model, combining convolution and
        max pooling.

        Equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=nn.ReLU())
        self.cv2 = Conv(c_ * 4, c2, 1, 1, act=nn.ReLU())
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Processes input through a series of convolutions and max pooling operations for feature extraction."""
        x = self.cv1(x)

        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


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
