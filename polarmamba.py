import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import datetime

# ==========================================
# 0. 配置与常量
# ==========================================
CELL_AREA = (25000.0 * 25000.0) / 1e12


# ==========================================
# 1. 注意力模块
# ==========================================

class h_sigmoid(nn.Module):
    """Hard Sigmoid 激活函数"""

    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    """Hard Swish 激活函数"""

    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """
    坐标注意力模块

    原理：
    1. 分别在水平和垂直方向做全局池化
    2. 拼接后通过共享的1x1卷积
    3. 分开后各自生成注意力权重
    4. 对输入特征进行加权

    效果：让模型关注重要的空间位置（如冰边缘）
    """

    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # 水平方向池化：(B,C,H,W) -> (B,C,H,1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # 垂直方向池化：(B,C,H,W) -> (B,C,1,W)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 中间通道数
        mip = max(8, inp // reduction)

        # 共享的1x1卷积
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        # 分别生成H和W方向的注意力
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 水平池化: (B,C,H,W) -> (B,C,H,1)
        x_h = self.pool_h(x)
        # 垂直池化: (B,C,H,W) -> (B,C,1,W) -> (B,C,W,1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 拼接: (B,C,H+W,1)
        y = torch.cat([x_h, x_w], dim=2)

        # 共享卷积
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 分开: (B,C,H,1) 和 (B,C,W,1)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        # 恢复形状: (B,C,W,1) -> (B,C,1,W)
        x_w = x_w.permute(0, 1, 3, 2)

        # 生成注意力权重 (0-1之间)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 加权输出
        out = identity * a_w * a_h

        return out


# ==========================================
# 1.5 极感知多方向 Mamba 模块（核心创新）
# ==========================================

class SelectiveScanDirection(nn.Module):
    """
    单方向选择性状态空间扫描 (Selective SSM)
    基于 Mamba 核心思想的原生 PyTorch 实现，适用于中等长度序列 (L~729)
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2.0, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state

        # 输入投影到两个分支: x (用于SSM) 和 z (用于门控)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 因果卷积 (局部上下文 + 因果性)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, groups=self.d_inner,
            padding=d_conv - 1, bias=True
        )

        # 投影生成 SSM 参数: delta, B, C
        self.x_proj = nn.Linear(self.d_inner, d_state * 3, bias=False)
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)

        # 状态矩阵 A (对数参数化保证稳定)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # 跳跃连接参数 D
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, L, d_model)
        return: (B, L, d_model)
        """
        residual = x

        # 投影并拆分
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # 各 (B, L, d_inner)

        # 因果卷积: 先转置为 (B, d_inner, L)，padding 后截断保证因果性
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :x_conv.size(2)]
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # 生成选择性参数
        x_proj_out = self.x_proj(x_conv)  # (B, L, 3*d_state)
        delta, B, C = x_proj_out.split([self.d_state, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)

        # 执行选择性扫描
        y = self.selective_scan(x_conv, delta, B, C)

        # 门控融合 (与 SwiGLU 类似)
        y = y * F.silu(z)

        y = self.out_proj(y)
        y = self.dropout(y)
        return self.norm(y + residual)

    def selective_scan(self, u, delta, B, C):
        """
        简化的选择性扫描 (序列长度 L 较小，如 27*27=729，循环开销可接受)
        u:     (B, L, d_inner)
        delta: (B, L, d_inner)
        B,C:   (B, L, d_state)
        """
        B_batch, L, d_inner = u.shape
        d_state = B.size(-1)

        # 离散化状态矩阵 A
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        delta = delta.unsqueeze(-1)  # (B, L, d_inner, 1)
        A = A.unsqueeze(0).unsqueeze(0)  # (1, 1, d_inner, d_state)

        deltaA = torch.exp(delta * A)  # (B, L, d_inner, d_state)
        deltaB = delta * B.unsqueeze(2)  # (B, L, d_inner, d_state)
        deltaB_u = deltaB * u.unsqueeze(-1)  # (B, L, d_inner, d_state)

        # 递归扫描 (模拟状态更新)
        x = torch.zeros(B_batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []
        for i in range(L):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = (C[:, i].unsqueeze(1) * x).sum(dim=-1)  # (B, d_inner)
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * u
        return y


class PolarMambaScanner(nn.Module):
    """
    极感知多方向 Mamba 扫描器 (Polar-aware Multi-Directional Mamba)

    创新扫描方式：
    1. horizontal (h)      : 行优先，左→右
    2. vertical (v)        : 列优先，上→下
    3. horizontal_flip (hf): 行优先，右→左
    4. vertical_flip (vf)  : 列优先，下→上
    5. polar (p)           : 以图像中心为北极点，按半径-角度排序向外扫描

    物理动机：
    - 四方向笛卡尔扫描捕获经向/纬向各向异性运动
    - 极坐标扫描天然契合北极海冰从极点向外辐射分布的拓扑结构
    """

    def __init__(self, d_model, d_state=16, expand=2.0, d_conv=4,
                 directions=None, dropout=0.0):
        super().__init__()
        if directions is None:
            directions = ['h', 'v', 'hf', 'vf', 'polar']
        self.directions = directions
        self.n_dir = len(directions)
        self.d_model = d_model

        # 每个方向独立的 SSM 扫描器
        self.scanners = nn.ModuleList([
            SelectiveScanDirection(d_model, d_state=d_state, expand=expand,
                                   d_conv=d_conv, dropout=dropout)
            for _ in range(self.n_dir)
        ])

        # 输入自适应方向门控 (根据全局特征动态加权各方向)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(d_model, self.n_dir),
            nn.Sigmoid()
        )

        # 极坐标索引缓存 (延迟初始化)
        self.register_buffer('_idx_polar', None, persistent=False)
        self.register_buffer('_idx_polar_inv', None, persistent=False)

    def _get_polar_index(self, H, W, device):
        cy, cx = H // 2, W // 2
        y = torch.arange(H, dtype=torch.float32, device=device).view(-1, 1) - cy
        x = torch.arange(W, dtype=torch.float32, device=device).view(1, -1) - cx
        r = torch.sqrt(y ** 2 + x ** 2)
        theta = torch.atan2(y, x)
        # 先按半径排序，同半径按角度排序，形成从中心向外的螺旋序列
        idx = torch.argsort(r.view(-1) * (2 * 3.14159265) + theta.view(-1))
        return idx.long()

    def _scan(self, x, direction):
        """将 2D 特征图重排为指定方向的 1D 序列"""
        B, C, H, W = x.shape
        if direction == 'h':
            return x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        elif direction == 'v':
            return x.permute(0, 3, 2, 1).reshape(B, H * W, C)
        elif direction == 'hf':
            return x.flip(dims=[3]).permute(0, 2, 3, 1).reshape(B, H * W, C)
        elif direction == 'vf':
            return x.flip(dims=[2]).permute(0, 3, 2, 1).reshape(B, H * W, C)
        elif direction == 'polar':
            if self._idx_polar is None or self._idx_polar.shape[0] != H * W:
                idx = self._get_polar_index(H, W, x.device)
                inv = torch.empty_like(idx)
                inv[idx] = torch.arange(H * W, device=x.device)
                self.register_buffer('_idx_polar', idx, persistent=False)
                self.register_buffer('_idx_polar_inv', inv, persistent=False)
            x_flat = x.view(B, C, H * W)
            x_sorted = x_flat[:, :, self._idx_polar]
            return x_sorted.permute(0, 2, 1)  # (B, H*W, C)
        else:
            raise ValueError(f"Unknown scan direction: {direction}")

    def _unscan(self, x_seq, direction, B, C, H, W):
        """将 1D 序列恢复为 2D 特征图"""
        if direction == 'h':
            return x_seq.view(B, H, W, C).permute(0, 3, 1, 2)
        elif direction == 'v':
            return x_seq.view(B, W, H, C).permute(0, 3, 2, 1)
        elif direction == 'hf':
            return x_seq.view(B, H, W, C).permute(0, 3, 1, 2).flip(dims=[3])
        elif direction == 'vf':
            return x_seq.view(B, W, H, C).permute(0, 3, 2, 1).flip(dims=[2])
        elif direction == 'polar':
            x_perm = x_seq.permute(0, 2, 1)  # (B, C, H*W)
            x_restored = x_perm[:, :, self._idx_polar_inv]
            return x_restored.view(B, C, H, W)
        else:
            raise ValueError(f"Unknown scan direction: {direction}")

    def forward(self, x):
        B, C, H, W = x.shape

        # 各方向扫描 + SSM
        outputs = []
        for i, direction in enumerate(self.directions):
            x_seq = self._scan(x, direction)
            y_seq = self.scanners[i](x_seq)
            y = self._unscan(y_seq, direction, B, C, H, W)
            outputs.append(y)

        # 堆叠: (B, n_dir, C, H, W)
        stacked = torch.stack(outputs, dim=1)

        # 自适应方向权重: 根据输入全局上下文决定各方向重要性
        g = self.gate(x)  # (B, n_dir)
        g = g.view(B, self.n_dir, 1, 1, 1)

        # 加权融合
        out = (stacked * g).sum(dim=1)  # (B, C, H, W)

        return out + x  # 残差连接


# ==========================================
# 2. 残差块（集成CoordAtt）
# ==========================================

class ResidualBlock(nn.Module):
    """
    残差块 + 坐标注意力 + Dropout

    结构：
    输入 → Conv → BN → ReLU → Dropout → Conv → BN → (+) → ReLU → CoordAtt → 输出
     │                                              ↑
     └──────────── Shortcut ────────────────────────┘
    """

    def __init__(self, in_channels, out_channels, stride=1,
                 dropout_rate=0.1, use_coord_att=True):
        super(ResidualBlock, self).__init__()

        self.use_coord_att = use_coord_att

        # 第一个卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Dropout
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # 第二个卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ★★★ 坐标注意力模块 ★★★
        if self.use_coord_att:
            self.coord_att = CoordAtt(out_channels, out_channels, reduction=32)

        # 捷径连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 主路径
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        # 残差连接
        out += self.shortcut(x)
        out = self.relu(out)

        # ★★★ 应用坐标注意力 ★★★
        if self.use_coord_att:
            out = self.coord_att(out)

        return out


# ==========================================
# 3. 主网络（CoordAtt + PolarMamba 双增强）
# ==========================================

class IceNetResUnetWithAttention(nn.Module):
    """
    带坐标注意力与极感知 Mamba 的 ResUNet 模型

    改进点：
    1. 每个残差块集成 CoordAtt
    2. Bridge 层添加 PolarMambaScanner，实现多方向长程依赖建模
    3. 使用残差预测（预测变化量）
    """

    def __init__(self, t1_steps, input_vars, n_forecast_days=14,
                 base_filter=32, dropout_rate=0.15, use_coord_att=True):
        super(IceNetResUnetWithAttention, self).__init__()

        self.n_forecast_days = n_forecast_days
        self.use_coord_att = use_coord_att

        # 输入通道 = 时间步 × 变量数 + 1(mask)
        in_channels = t1_steps * input_vars + 1

        # --- 初始卷积 ---
        self.init_conv = nn.Conv2d(in_channels, base_filter, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(base_filter)

        # --- Encoder（编码器）---
        self.enc1 = ResidualBlock(base_filter, base_filter,
                                  dropout_rate=dropout_rate,
                                  use_coord_att=use_coord_att)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualBlock(base_filter, base_filter * 2,
                                  dropout_rate=dropout_rate,
                                  use_coord_att=use_coord_att)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualBlock(base_filter * 2, base_filter * 4,
                                  dropout_rate=dropout_rate,
                                  use_coord_att=use_coord_att)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ResidualBlock(base_filter * 4, base_filter * 8,
                                  dropout_rate=dropout_rate,
                                  use_coord_att=use_coord_att)
        self.pool4 = nn.MaxPool2d(2)

        # --- Bridge（桥接层）---
        self.bridge_conv1 = nn.Conv2d(base_filter * 8, base_filter * 16, 3,
                                      padding=2, dilation=2)
        self.bridge_bn1 = nn.BatchNorm2d(base_filter * 16)
        self.bridge_dropout1 = nn.Dropout2d(p=dropout_rate * 2)

        self.bridge_conv2 = nn.Conv2d(base_filter * 16, base_filter * 8, 3,
                                      padding=2, dilation=2)
        self.bridge_bn2 = nn.BatchNorm2d(base_filter * 8)
        self.bridge_dropout2 = nn.Dropout2d(p=dropout_rate * 2)

        # Bridge 坐标注意力
        if self.use_coord_att:
            self.bridge_att = CoordAtt(base_filter * 8, base_filter * 8, reduction=32)

        # ★★★ 核心创新：极感知多方向 Mamba 模块 ★★★
        # 嵌入在 Bridge 末端，处理最小分辨率特征图 (如 27x27)
        self.bridge_mamba = PolarMambaScanner(
            d_model=base_filter * 8,
            d_state=16,
            expand=2.0,
            d_conv=4,
            directions=['h', 'v', 'hf', 'vf', 'polar'],
            dropout=dropout_rate
        )

        # --- Decoder（解码器）---
        self.up4 = nn.ConvTranspose2d(base_filter * 8, base_filter * 8, 2, stride=2)
        self.dec4 = ResidualBlock(base_filter * 16, base_filter * 4,
                                  dropout_rate=dropout_rate,
                                  use_coord_att=use_coord_att)

        self.up3 = nn.ConvTranspose2d(base_filter * 4, base_filter * 4, 2, stride=2)
        self.dec3 = ResidualBlock(base_filter * 8, base_filter * 2,
                                  dropout_rate=dropout_rate,
                                  use_coord_att=use_coord_att)

        self.up2 = nn.ConvTranspose2d(base_filter * 2, base_filter * 2, 2, stride=2)
        self.dec2 = ResidualBlock(base_filter * 4, base_filter,
                                  dropout_rate=dropout_rate,
                                  use_coord_att=use_coord_att)

        self.up1 = nn.ConvTranspose2d(base_filter, base_filter, 2, stride=2)
        self.dec1 = ResidualBlock(base_filter * 2, base_filter,
                                  dropout_rate=dropout_rate,
                                  use_coord_att=use_coord_att)

        # --- 输出头 ---
        self.final_conv = nn.Conv2d(base_filter, n_forecast_days, kernel_size=1)

    def forward(self, x, land_mask):
        B, T, C, H, W = x.shape

        # 展平时间维度
        x_flat = x.view(B, T * C, H, W)

        # 处理 mask 维度
        if land_mask.dim() == 2:
            mask_in = land_mask.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)
        elif land_mask.dim() == 3:
            mask_in = land_mask.unsqueeze(0).repeat(B, 1, 1, 1)
        else:
            mask_in = land_mask

        # 拼接输入和 mask
        x_in = torch.cat([x_flat, mask_in], dim=1)

        # --- Encoder ---
        e0 = F.relu(self.bn_init(self.init_conv(x_in)))

        e1 = self.enc1(e0)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # --- Bridge ---
        b = F.relu(self.bridge_bn1(self.bridge_conv1(p4)))
        b = self.bridge_dropout1(b)
        b = F.relu(self.bridge_bn2(self.bridge_conv2(b)))
        b = self.bridge_dropout2(b)

        if self.use_coord_att:
            b = self.bridge_att(b)

        # ★★★ 应用 PolarMamba 长程建模 ★★★
        b = self.bridge_mamba(b)

        # --- Decoder ---
        d4 = self.up4(b)
        if d4.shape != e4.shape:
            d4 = F.interpolate(d4, size=e4.shape[2:])
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:])
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:])
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:])
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)

        # --- 残差预测 ---
        delta = self.final_conv(d1)
        last_input = x[:, -1, 0, :, :].unsqueeze(1)
        final_prediction = last_input + delta

        return torch.clamp(final_prediction, 0.0, 1.0)


# ==========================================
# 4. 数据集与工具函数
# ==========================================

class OsiIceDataset(Dataset):
    def __init__(self, ice_conc_data, input_days=14, output_days=14):
        self.data = ice_conc_data
        self.input_days = input_days
        self.output_days = output_days
        self.num_samples = len(self.data) - (self.input_days + self.output_days)

    def __len__(self):
        return max(0, self.num_samples)

    def __getitem__(self, idx):
        input_slice = self.data[idx: idx + self.input_days]
        target_slice = self.data[idx + self.input_days: idx + self.input_days + self.output_days]

        input_tensor = torch.from_numpy(input_slice).unsqueeze(1).float()
        target_tensor = torch.from_numpy(target_slice).unsqueeze(1).float()

        return input_tensor, target_tensor


def hybrid_gradient_loss(pred, target, mask, alpha=0.5):
    """混合损失：L1 + 梯度损失"""
    diff = torch.abs(pred - target)

    if mask.dim() == 2:
        mask_expanded = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask_expanded = mask.unsqueeze(0)
    else:
        mask_expanded = mask

    l1_loss = (diff * mask_expanded).sum() / (mask_expanded.expand_as(diff).sum() + 1e-6)

    def gradient(x):
        dw = torch.abs(x[..., :, 1:] - x[..., :, :-1])
        dh = torch.abs(x[..., 1:, :] - x[..., :-1, :])
        return dw, dh

    pred_dw, pred_dh = gradient(pred)
    target_dw, target_dh = gradient(target)

    if mask.dim() == 2:
        mask_dw = mask[:, 1:].unsqueeze(0).unsqueeze(0)
        mask_dh = mask[1:, :].unsqueeze(0).unsqueeze(0)
    else:
        mask_dw = mask_expanded[..., :, 1:]
        mask_dh = mask_expanded[..., 1:, :]

    grad_loss_w = (torch.abs(pred_dw - target_dw) * mask_dw).mean()
    grad_loss_h = (torch.abs(pred_dh - target_dh) * mask_dh).mean()

    grad_loss = grad_loss_w + grad_loss_h

    return l1_loss + alpha * grad_loss


def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def get_days_in_year(year):
    return 366 if is_leap_year(year) else 365


def compute_climatology(train_data, start_year=2015):
    """计算日平均气候态"""
    print("Computing Climatology from Training Data...")
    H, W = train_data.shape[1], train_data.shape[2]
    clim_sum = np.zeros((365, H, W), dtype=np.float32)
    clim_count = np.zeros((365, H, W), dtype=np.float32)

    current_idx = 0
    year = start_year
    max_idx = len(train_data)

    while current_idx < max_idx:
        days_in_year = get_days_in_year(year)
        for d in range(days_in_year):
            if current_idx >= max_idx:
                break
            doy = d
            if days_in_year == 366:
                if d == 59:
                    current_idx += 1
                    continue
                elif d > 59:
                    doy = d - 1
            if doy < 365:
                clim_sum[doy] += train_data[current_idx]
                clim_count[doy] += 1.0
            current_idx += 1
        year += 1

    clim_mean = np.divide(clim_sum, clim_count, out=np.zeros_like(clim_sum),
                          where=clim_count > 0)
    return clim_mean


def get_batch_doys(batch_start_indices, input_days, forecast_days, start_date_offset):
    """获取批次中每个样本的 Day of Year"""
    batch_doys = []
    base_date = start_date_offset

    for idx in batch_start_indices:
        sample_doys = []
        target_start_idx = idx + input_days
        curr_date = base_date + datetime.timedelta(days=int(target_start_idx))

        for d in range(forecast_days):
            forecast_date = curr_date + datetime.timedelta(days=d)
            tt = forecast_date.timetuple()
            doy = tt.tm_yday - 1
            is_leap = is_leap_year(forecast_date.year)
            if is_leap:
                if doy == 59:
                    doy = 58
                elif doy > 59:
                    doy = doy - 1
            if doy >= 365:
                doy = 364
            sample_doys.append(doy)
        batch_doys.append(sample_doys)

    return torch.tensor(batch_doys)


# ==========================================
# 5. Early Stopping
# ==========================================

class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=5, min_delta=1e-5, save_path="./pth/best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self._save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            self._save_checkpoint(model)
        else:
            self.counter += 1
            print(f"   ⚠ EarlyStopping: {self.counter}/{self.patience} "
                  f"(best={self.best_loss:.6f} @ epoch {self.best_epoch + 1})")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, model):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(model.state_dict(), self.save_path)
        print(f"   ✓ Best model saved (loss={self.best_loss:.6f})")


def quick_validate(model, val_loader, valid_mask_tensor, device):
    """快速验证"""
    model.eval()
    total_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if targets.dim() == 5:
                targets = targets.squeeze(2)

            preds = model(inputs, valid_mask_tensor)

            if valid_mask_tensor.dim() == 2:
                mask_expanded = valid_mask_tensor.unsqueeze(0).unsqueeze(0)
            else:
                mask_expanded = valid_mask_tensor

            diff = torch.abs(preds - targets) * mask_expanded
            valid_count = mask_expanded.expand_as(preds).sum() + 1e-6
            mae = diff.sum() / valid_count

            total_mae += mae.item()
            num_batches += 1

    model.train()
    return total_mae / max(num_batches, 1)


# ==========================================
# 6. 主程序
# ==========================================

def main():
    # --- 配置参数 ---
    data_dir = r"./data"
    nc_file = os.path.join(data_dir, "arctic_sea_ice_merged_filled.nc")
    mask_file = os.path.join(data_dir, "valid_mask.npy")

    batch_size = 8
    lr = 3e-4
    epochs = 20
    input_days = 14
    output_days = 14

    # 防过拟合参数
    weight_decay = 1e-4
    dropout_rate = 0.15
    early_stopping_patience = 7
    grad_clip_max_norm = 1.0
    val_split_ratio = 0.1

    # ★★★ 注意力与 Mamba 开关 ★★★
    use_coord_att = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CoordAtt enabled: {use_coord_att}")
    print(f"PolarMamba enabled: True (5-direction scanning)")

    # --- 1. 加载数据 ---
    print("\nLoading OSI NetCDF data...")
    try:
        ds = xr.open_dataset(nc_file)
        print("Slicing data from 2015-01-01 to 2024-12-31...")
        ds = ds.sel(time=slice("2015-01-01", "2024-12-31"))
        ice_conc_all = ds['ice_conc'].fillna(0).values
        ds.close()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Data range before clip: Min={ice_conc_all.min():.4f}, Max={ice_conc_all.max():.4f}")
    ice_conc_all = np.clip(ice_conc_all, 0, 1)

    total_timesteps = ice_conc_all.shape[0]
    print(f"Total time steps (2015-2024): {total_timesteps}")

    # --- 2. 划分数据集 ---
    start_year = 2015
    train_years_end = 2021

    train_days_count = 0
    for year in range(start_year, train_years_end + 1):
        train_days_count += get_days_in_year(year)

    print(f"\n--- Splitting Data ---")
    print(f"Train End Day Index: {train_days_count} (End of 2021)")

    train_data_full = ice_conc_all[:train_days_count]
    test_data = ice_conc_all[train_days_count:]

    # 划分验证集
    val_days = int(len(train_data_full) * val_split_ratio)
    train_data = train_data_full[:-val_days]
    val_data = train_data_full[-val_days:]

    print(f"Train Set Shape: {train_data.shape}")
    print(f"Val   Set Shape: {val_data.shape}")
    print(f"Test  Set Shape: {test_data.shape}")

    # 计算气候态
    clim_mean_np = compute_climatology(train_data, start_year=2015)
    clim_mean_tensor = torch.from_numpy(clim_mean_np).unsqueeze(1).to(device)

    # 创建 DataLoader
    train_dataset = OsiIceDataset(train_data, input_days, output_days)
    val_dataset = OsiIceDataset(val_data, input_days, output_days)
    test_dataset = OsiIceDataset(test_data, input_days, output_days)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # 加载 Mask
    print("\nLoading Valid Mask...")
    try:
        valid_mask_np = np.load(mask_file)
        valid_mask_tensor = torch.from_numpy(valid_mask_np).float().to(device)
        valid_pixels_count = valid_mask_tensor.sum().item()
        print(f"Valid pixels: {valid_pixels_count:.0f}")
    except FileNotFoundError:
        print(f"Error: Mask file not found at {mask_file}")
        return

    # --- 3. 初始化模型 ---
    # ★★★ 使用 CoordAtt + PolarMamba 双增强模型 ★★★
    model = IceNetResUnetWithAttention(
        t1_steps=input_days,
        input_vars=1,
        n_forecast_days=output_days,
        base_filter=32,
        dropout_rate=dropout_rate,
        use_coord_att=use_coord_att
    ).to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters: {total_params:,} total, {trainable_params:,} trainable")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    early_stopper = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=1e-5,
        save_path="./pth/best_icenet_polarmamba.pth"
    )

    # --- 4. 训练循环 ---
    print(f"\n{'=' * 60}")
    print(f"Starting Training for max {epochs} epochs")
    print(f"  CoordAtt:     {use_coord_att}")
    print(f"  PolarMamba:   5-direction (h/v/hf/vf/polar)")
    print(f"  Input days:   {input_days} | Forecast days: {output_days}")
    print(f"  Dropout:      {dropout_rate}")
    print(f"  Weight Decay: {weight_decay}")
    print(f"  LR Schedule:  CosineAnnealingWarmRestarts")
    print(f"  EarlyStopping Patience: {early_stopping_patience}")
    print(f"{'=' * 60}\n")

    for epoch in range(epochs):
        model.train()
        train_loss_accum = 0

        current_lr = optimizer.param_groups[0]['lr']

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [lr={current_lr:.2e}]")
        for inputs, targets in loop:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if targets.dim() == 5:
                targets = targets.squeeze(2)

            optimizer.zero_grad()

            preds = model(inputs, valid_mask_tensor)
            loss = hybrid_gradient_loss(preds, targets, valid_mask_tensor)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()

            train_loss_accum += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        avg_train_loss = train_loss_accum / len(train_loader)
        val_mae = quick_validate(model, val_loader, valid_mask_tensor, device)

        print(f"Epoch {epoch + 1:>2d} | Train Loss: {avg_train_loss:.6f} | "
              f"Val MAE: {val_mae:.6f} | LR: {current_lr:.2e}")

        early_stopper(val_mae, model, epoch)
        if early_stopper.early_stop:
            print(f"\n★ Early Stopping triggered at epoch {epoch + 1}!")
            print(f"  Best epoch was {early_stopper.best_epoch + 1} "
                  f"with Val MAE = {early_stopper.best_loss:.6f}")
            break

    # --- 5. 加载最佳模型进行测试 ---
    print(f"\n{'=' * 60}")
    print("Loading BEST model for evaluation...")
    print(f"{'=' * 60}")

    best_path = "./pth/best_icenet_polarmamba.pth"
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded best model from epoch {early_stopper.best_epoch + 1}")
    else:
        print("Warning: Best model file not found, using current model")

    # --- 6. 测试 ---
    print("\nStarting Evaluation on Test Set (2022-2024) using BEST Model...")

    model.eval()

    total_mae = 0
    total_rmse = 0
    total_bias = 0
    total_iiee = 0
    total_acc = 0
    num_batches = 0

    daily_mae = np.zeros(output_days)
    daily_rmse = np.zeros(output_days)
    daily_bias = np.zeros(output_days)
    daily_iiee = np.zeros(output_days)
    daily_acc = np.zeros(output_days)

    dataset_start_date = datetime.date(2015, 1, 1)
    test_set_start_date = dataset_start_date + datetime.timedelta(days=int(train_days_count))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="Testing")):
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs, valid_mask_tensor)
            preds = preds.unsqueeze(2)

            current_batch_size = inputs.shape[0]
            start_idx = batch_idx * batch_size
            indices = np.arange(start_idx, start_idx + current_batch_size)

            batch_doys = get_batch_doys(indices, input_days, output_days,
                                        test_set_start_date).to(device)

            clim_batch_flat = torch.index_select(clim_mean_tensor, 0, batch_doys.view(-1))
            clim_batch = clim_batch_flat.view(current_batch_size, output_days, 1, 432, 432)

            mask_expanded = valid_mask_tensor.view(1, 1, 1, 432, 432)

            diff = preds - targets
            diff_valid = diff * mask_expanded

            abs_diff_sum = torch.abs(diff_valid).sum()
            sq_diff_sum = (diff_valid ** 2).sum()
            bias_sum = diff_valid.sum()

            total_valid_points = valid_pixels_count * current_batch_size * output_days

            total_mae += (abs_diff_sum / total_valid_points).item()
            total_rmse += torch.sqrt(sq_diff_sum / total_valid_points).item()
            total_bias += (bias_sum / total_valid_points).item()

            pred_bin = (preds >= 0.15).float()
            target_bin = (targets >= 0.15).float()
            iiee_diff = torch.abs(pred_bin - target_bin) * mask_expanded
            batch_iiee = iiee_diff.sum() * CELL_AREA / (current_batch_size * output_days)
            total_iiee += batch_iiee.item()

            pred_anom = (preds - clim_batch) * mask_expanded
            target_anom = (targets - clim_batch) * mask_expanded

            p_flat = pred_anom.view(current_batch_size, output_days, -1)
            t_flat = target_anom.view(current_batch_size, output_days, -1)
            m_flat = mask_expanded.view(1, 1, -1)

            p_mean = p_flat.sum(dim=2, keepdim=True) / valid_pixels_count
            t_mean = t_flat.sum(dim=2, keepdim=True) / valid_pixels_count

            p_centered = (p_flat - p_mean) * m_flat
            t_centered = (t_flat - t_mean) * m_flat

            num = (p_centered * t_centered).sum(dim=2)
            den = (torch.sqrt((p_centered ** 2).sum(dim=2)) *
                   torch.sqrt((t_centered ** 2).sum(dim=2)))

            acc_val = num / (den + 1e-6)
            total_acc += acc_val.mean().item()

            num_batches += 1

            for day in range(output_days):
                d_diff_valid = diff_valid[:, day, :, :, :]
                d_pred_bin = pred_bin[:, day, :, :, :]
                d_target_bin = target_bin[:, day, :, :, :]
                mask_2d = valid_mask_tensor.view(1, 1, 432, 432)

                d_points = valid_pixels_count * current_batch_size
                daily_mae[day] += (torch.abs(d_diff_valid).sum() / d_points).item()
                daily_rmse[day] += torch.sqrt((d_diff_valid ** 2).sum() / d_points).item()
                daily_bias[day] += (d_diff_valid.sum() / d_points).item()

                d_iiee_diff = torch.abs(d_pred_bin - d_target_bin) * mask_2d
                daily_iiee[day] += (d_iiee_diff.sum() * CELL_AREA / current_batch_size).item()
                daily_acc[day] += acc_val[:, day].mean().item()

    # --- 结果输出 ---
    if num_batches > 0:
        final_mae = total_mae / num_batches
        final_rmse = total_rmse / num_batches
        final_bias = total_bias / num_batches
        final_iiee = total_iiee / num_batches
        final_acc = total_acc / num_batches
    else:
        final_mae = final_rmse = final_bias = final_iiee = final_acc = 0.0

    print(f"\n{'=' * 60}")
    print(f"Test Results (2022-2024) — Using Best Model with CoordAtt + PolarMamba")
    print(f"{'=' * 60}")
    print(f"Overall MAE:  {final_mae:.5f}")
    print(f"Overall RMSE: {final_rmse:.5f}")
    print(f"Overall Bias: {final_bias:.5f}")
    print(f"Overall IIEE: {final_iiee:.5f} (10^6 km^2)")
    print(f"Overall ACC:  {final_acc:.5f}")

    print(f"\nDay-wise Performance:")
    print(f"{'Day':<4} | {'MAE':<8} | {'RMSE':<8} | {'Bias':<8} | {'IIEE':<8} | {'ACC':<8}")
    print("-" * 60)
    for d in range(output_days):
        d_mae = daily_mae[d] / num_batches
        d_rmse = daily_rmse[d] / num_batches
        d_bias = daily_bias[d] / num_batches
        d_iiee = daily_iiee[d] / num_batches
        d_acc = daily_acc[d] / num_batches
        print(f"{d + 1:<4d} | {d_mae:.5f}   | {d_rmse:.5f}   | "
              f"{d_bias:.5f}   | {d_iiee:.5f}   | {d_acc:.5f}")


if __name__ == '__main__':
    main()