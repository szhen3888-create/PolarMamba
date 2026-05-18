# PolarMamba

用过去 14 天北极海冰浓度预报未来 14 天。骨干是 ResUNet，残差块里加了 CoordAtt，Bridge 里放了极感知多方向 Mamba（PolarMamba）。输入输出都是 0–1 浓度，格网 432×432。

## 思路简述

编码器–解码器结构和常见海冰 U-Net 类似。CoordAtt 用在各层 ResBlock 和 Bridge 上，主要想让模型多看冰缘一带。Bridge 分辨率最低（大约 27×27），在这里做序列建模：把特征图拉成序列，沿五个方向做 selective scan，再按门控权重合回来。

五个扫描方向：

- `h` / `hf`：按行，正向或反向
- `v` / `vf`：按列，正向或反向  
- `polar`：以格网中心为参考点，按半径和角度从里往外排

输出头预测的是相对**输入最后一天**的变化量，加回去以后再 `clamp` 到 [0, 1]。

训练损失是有效海洋格点上的 L1，再加一项梯度损失（`hybrid_gradient_loss`）。测试阶段会算 MAE、RMSE、Bias、IIEE（15% 阈值）和相对气候态的 ACC。默认训练到 2021 年底，测试 2022–2024。

## 依赖

Python 3.9+，PyTorch 2.0+，以及 `numpy`、`xarray`、`netcdf4`、`tqdm`。有 GPU 会快很多，`batch_size=8` 时大概需要 8G 左右显存。

```bash
pip install -r requirements.txt
```

`torch` 请按自己机器的 CUDA 版本从 [pytorch.org](https://pytorch.org/) 装，requirements 里没锁具体构建。

## 数据

把文件放到 `data/` 下：

```
data/
  arctic_sea_ice_merged_filled.nc
  valid_mask.npy
```

NC 里要有变量 `ice_conc`，时间维覆盖 2015-01-01 到 2024-12-31（和 `main()` 里切片一致）。浓度已是 0–1；读入时缺测填 0。数据可从 [OSI SAF](https://osi-saf.eumetsat.int/) 等产品自行下载、裁剪、合并，仓库不带数据。

`valid_mask.npy` 形状 `(432, 432)`，海洋有效区为 1，陆地/无效为 0，损失和评估都只算 mask=1 的格点。路径不对就改 `polarmamba.py` 里 `main()` 开头的 `data_dir`、`nc_file`、`mask_file`。

## 训练

```bash
python polarmamba.py
```

2015–2021 做训练，其中最后 10% 天数做验证；2022–2024 做测试。验证集 MAE 连续 7 个 epoch 不降就停，最优权重写到 `./pth/best_icenet_polarmamba.pth`。

常用设置在 `main()` 里改：`input_days`/`output_days`（默认 14）、`batch_size`（8）、`lr`（3e-4）、`epochs`（20）、`dropout_rate`（0.15）。Windows 上 `DataLoader` 报错可以把 `num_workers` 改成 0。

## 推理

训练脚本跑完直接用保存的权重。单独推理可以这样写：

```python
import numpy as np
import torch
from polarmamba import IceNetResUnetWithAttention

device = "cuda" if torch.cuda.is_available() else "cpu"
model = IceNetResUnetWithAttention(
    t1_steps=14, input_vars=1, n_forecast_days=14,
    base_filter=32, dropout_rate=0.15, use_coord_att=True,
).to(device)
model.load_state_dict(torch.load("./pth/best_icenet_polarmamba.pth", map_location=device))
model.eval()

history = np.load("last_14_days.npy")      # (14, 432, 432)
mask = np.load("./data/valid_mask.npy")    # (432, 432)

x = torch.from_numpy(history).float().unsqueeze(0).unsqueeze(2).to(device)
m = torch.from_numpy(mask).float().to(device)

with torch.no_grad():
    out = model(x, m)   # (1, 14, 432, 432)
```

`mask` 会拼进输入通道，和训练时保持一致。改分辨率或预报天数需要重新训练，不能只换权重。

## 文件说明

- `polarmamba.py`：模型、数据集、训练与测试，入口是 `main()`
- `data/`：放 NC 和 mask（别提交大文件，已在 `.gitignore` 里）
- `pth/`：checkpoint

## 引用

用到了 OSI SAF 数据的话请按其产品说明引用。代码引用待补。
