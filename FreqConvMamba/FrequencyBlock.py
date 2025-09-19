import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange



class FFN(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(FFN, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2
        # PW first or DW first?
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim * 2, 1),
        )

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1,
                      groups=self.dim_sp),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=5, padding=2,
                      groups=self.dim_sp),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=7, padding=3,
                      groups=self.dim_sp),
        )

        self.gelu = nn.GELU()
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim_sp, dim=1))
        x[1] = self.conv1_1(x[1])
        x[2] = self.conv1_2(x[2])
        x[3] = self.conv1_3(x[3])
        x = torch.cat(x, dim=1)
        x = self.gelu(x)
        x = self.conv_fina(x)

        return x


class TokenMixer_For_Local(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(TokenMixer_For_Local, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2

        self.CDilated_1 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=1, dilation=1, groups=self.dim_sp)
        self.CDilated_2 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=2, dilation=2, groups=self.dim_sp)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        cd1 = self.CDilated_1(x1)
        cd2 = self.CDilated_2(x2)
        x = torch.cat([cd1, cd2], dim=1)

        return x


class FrequencyBlock(nn.Module):
    # simple tasks, e.g. dehazing\deraining can set groups=1 for better latency; complex tasks, e.g. motion blur can set groups=4 for better performance.
    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FrequencyBlock, self).__init__()
        self.groups = groups
        self.bn = nn.BatchNorm2d(out_channels * 2)

        self.fdc = nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2 * self.groups,
                             kernel_size=1, stride=1, padding=0, groups=self.groups, bias=True)
        self.weight = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=self.groups, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )

        self.fpe = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3,
                             padding=1, stride=1, groups=in_channels * 2, bias=True)

    def forward(self, x):
        batch, c, h, w = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft2(x, norm='ortho')  # (4,64,256,129)由于实信号对称性，只需保留一半频率
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)  # (4,64,256,129,1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)  # (4,64,256,129,1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)  # (4,64,256,129,2)
        ffted = rearrange(ffted, 'b c h w d -> b (c d) h w').contiguous()  # (4,128,256,129)
        # 重组逻辑:将通道维(c)和实虚部维(d)合并新通道数 = c * 2
        ffted = self.bn(ffted)
        ffted = self.fpe(ffted) + ffted
        # fpe: 分组数 = 总通道数的深度可分离卷积 作用: 在频域进行空间特征增强
        dy_weight = self.weight(ffted)  # (4,1,256,129)  生成通道注意力权重
        ffted = self.fdc(ffted).view(batch, self.groups, 2 * c, h, -1)  # (4,1,128,256,129)
        # fdc: 分组卷积(groups=self.groups) 目的: 为每个分组创建独立的特征表示
        ffted = torch.einsum('ijkml,ijml->ikml', ffted, dy_weight)  # (4,128,256,129) 作用: 动态融合各分组特征，根据权重加和
        ffted = F.gelu(ffted)
        ffted = rearrange(ffted, 'b (c d) h w -> b c h w d', d=2).contiguous()  # (4,64,256,129,2)
        # 逆向重组:将合并的[2 * c]拆分为[c, 2]实部虚部分离
        ffted = torch.view_as_complex(ffted)  # (4,64,256,129)
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')  # (4,64,256,256)

        return output


class TokenMixer_For_Gloal(nn.Module):
    def __init__(
            self,
            dim
    ):
        super(TokenMixer_For_Gloal, self).__init__()
        self.dim = dim
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU()
        )
        self.FFC = FourierUnit(self.dim * 2, self.dim * 2)

    def forward(self, x):
        x = self.conv_init(x)
        x0 = x
        x = self.FFC(x)
        x = self.conv_fina(x + x0)

        return x


class Mixer(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer_for_local=TokenMixer_For_Local,
            token_mixer_for_gloal=TokenMixer_For_Gloal,
    ):
        super(Mixer, self).__init__()
        self.dim = dim
        self.mixer_local = token_mixer_for_local(dim=self.dim, )
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim, )

        self.ca_conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * dim, 2 * dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * dim // 2, 2 * dim, kernel_size=1),
            nn.Sigmoid()
        )  # 计算通道注意力权重

        self.gelu = nn.GELU()
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, 1),
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


class Block(nn.Module):
    def __init__(
            self,
            dim,
            norm_layer=nn.BatchNorm2d,
            token_mixer=Mixer,
    ):
        super(Block, self).__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        # self.norm2 = norm_layer(dim)
        self.mixer = token_mixer(dim=self.dim)
        # self.ffn = FFN(dim=self.dim)

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        copy = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = x * self.beta + copy

        return x


class Stage(nn.Module):
    def __init__(
            self,
            depth=int,
            in_channels=int,
    ) -> None:
        super(Stage, self).__init__()
        self.blocks = nn.Sequential(*[
            Block(
                dim=in_channels,
                norm_layer=nn.BatchNorm2d,
                token_mixer=Mixer,
            )
            for index in range(depth)
        ])

    def forward(self, input=torch.Tensor) -> torch.Tensor:
        _, dim, _, _ = input.size()
        output = self.blocks(input)
        return output
