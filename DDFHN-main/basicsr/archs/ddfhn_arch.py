import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY


class Spatial_domain_Local(nn.Module):
    def __init__(self, dim, groups=8):
        super(Spatial_domain_Local, self).__init__()
        self.dim = dim
        self.dim_sp = dim//2

        self.C_1 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=1, groups=self.dim_sp)
        self.C_2 = nn.Conv2d(self.dim_sp, self.dim_sp, 5, stride=1, padding=2, groups=self.dim_sp)
        self.c = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        _, _, h, w = x.shape
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.C_1(x1)
        x2 = self.C_2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.c(x)
        return x

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=4):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.bn = nn.BatchNorm2d(out_channels * 2)

        self.fdc = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2 * self.groups,
                                                        kernel_size=1, stride=1, padding=0, groups=self.groups, bias=True)
        self.weight = nn.Sequential(
             nn.Conv2d(in_channels=in_channels * 2, out_channels=self.groups, kernel_size=1, stride=1, padding=0),
             nn.Softmax(dim=1)
        )

        self.fpe = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3,
                                        padding=1, stride=1, groups=in_channels * 2,bias=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        ffted = rearrange(ffted, 'b c h w d -> b (c d) h w').contiguous()
        ffted = self.bn(ffted)
        ffted = self.fpe(ffted) + ffted
        dy_weight = self.weight(ffted)
        ffted = self.fdc(ffted).view(batch, self.groups, 2*c, h, -1)  # (batch, c*2, h, w/2+1)
        ffted = torch.einsum('ijkml,ijml->ikml', ffted, dy_weight)
        ffted = F.gelu(ffted)
        ffted = rearrange(ffted, 'b (c d) h w -> b c h w d', d=2).contiguous()
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')
        return output

class Frequency_domain_Gloal(nn.Module):
    def __init__(
            self,
            dim
    ):
        super(Frequency_domain_Gloal, self).__init__()
        self.dim = dim
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU()
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU()
        )
        self.FFC = FourierUnit(self.dim*2, self.dim*2)

    def forward(self, x):
        x = self.conv_init(x)
        x0 = x
        x = self.FFC(x)
        x = self.conv_fina(x+x0)
        return x

# Spatio-Frequency Feature Mixer Unit
class SFFMU(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer_for_local=Spatial_domain_Local,
            token_mixer_for_gloal=Frequency_domain_Gloal,
    ):
        super(SFFMU, self).__init__()
        self.dim = dim
        self.mixer_local = token_mixer_for_local(dim=self.dim,)
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim,)

        self.ca_conv = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2*dim, 2*dim//2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*dim//2, 2*dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.gelu = nn.GELU()
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, 2*dim, 1),
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        x_local = self.mixer_local(x[0])
        x_gloal = self.mixer_gloal(x[1])
        x = torch.cat([x_local, x_gloal], dim=1)
        x = self.gelu(x)
        x = self.ca(x) * x
        x = self.ca_conv(x)

        return x

class FreBlockCha(nn.Module):
    def __init__(self, dim):
        super(FreBlockCha, self).__init__()
        self.processreal = nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=1,padding=0,stride=1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(dim,dim,kernel_size=1,padding=0,stride=1))
        self.processimag = nn.Sequential(
            nn.Conv2d(dim,dim, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim,dim, kernel_size=1, padding=0, stride=1))

    def forward(self,x):
        real = self.processreal(x.real)
        imag = self.processimag(x.imag)
        x_out = torch.complex(real, imag)

        return x_out

# Hybird Feature Feed-forward Unit (HFFU)
class HFFU(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(HFFU, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2
        # PW first or DW first?
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim*2, 1),
        )

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1, dilation=1, groups=self.dim_sp),)
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=2, dilation=2, groups=self.dim_sp),)
        self.c = nn.Conv2d(self.dim, self.dim, 1, 1, 0)

        self.frequency_process = FreBlockCha(self.dim)

        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
        )

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.conv_init(x)
        x1, x2 = x.chunk(2, dim=1)
        x11, x12 = x1.chunk(2, dim=1)
        x11 = self.conv1_1(x11)
        x12 = self.conv1_2(x12)
        x1 = self.c(torch.cat([x11, x12], dim=1))

        x2 = torch.fft.rfft2(x2, norm='backward')
        x2 = self.frequency_process(x2)
        x2 = torch.fft.irfft2(x2, s=(H, W), norm='backward')

        x = torch.cat([x1, x2], 1)
        x = self.conv_fina(x)
        return x

# Dual-domain Feature Extraction Block (DDFEB)
class DDFEB(nn.Module):
    def __init__(
            self,
            dim,
            norm_layer=nn.BatchNorm2d,
            token_mixer=SFFMU,
    ):
        super(DDFEB, self).__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mixer = token_mixer(dim=self.dim)
        self.ffn = HFFU(dim=self.dim)

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        copy = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = x * self.beta + copy

        copy = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x * self.gamma + copy

        return x

# @ARCH_REGISTRY.register()
class DDFHN(nn.Module):
    def __init__(self, dim, n_blocks=8, upscaling_factor=4):
        super().__init__()
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)

        self.feats = nn.Sequential(*[DDFEB(dim) for _ in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor)
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x


if __name__== '__main__':
    #############Test Model Complexity #############
    import time
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    # 指定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # x = torch.randn(1, 3, 640, 480)
    x = torch.randn(1, 3, 320, 240)

    model = DDFHN(dim=36, n_blocks=8, upscaling_factor=4)
    print(model)
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)


