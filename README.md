# project1-numpy

用 NumPy 实现 OpenCLIP ViT-B/32 图像编码器的前向计算。输入是预处理后的图像
`[B,3,224,224]`，输出是每张图一个 512 维单位向量。不训练，不需要 GPU、PyTorch 或联网。

## 运行

```bash
python -m pip install -r requirements.txt
python check.py --stage all
python run.py --output output_tokens.npz
```

`check.py` 按阶段对照课程提供的参考数组；`run.py` 把 10 张图的 ID 和特征写进
`output_tokens.npz`。

## 文件

- `image_encoder.py`：TODO-01 至 TODO-06 的实现，实验只改这个文件。
- `helpers.py`：LayerNorm、GELU、注意力系数、单位化，课程提供。
- `preprocess_image.py`、`runtime.py`：图像预处理与数据加载。
- `check.py`、`run.py`：分阶段检查和结果导出。
- `transformer_steps.py`：讲义里的微型 block 演算。

`weights/`（约 312MB）和 `assets/` 是课程发的材料，没有提交到仓库。复现时把它们放回
本目录即可。

## 结果

```
assets       PASS: 152 matching visual tensors and ten image/reference pairs
preprocess   PASS max_abs=0.000e+00
patches      PASS max_abs=0.000e+00
tokens       PASS max_abs=4.578e-05
block0       PASS max_abs=6.109e-06
last_tokens  PASS max_abs=4.101e-05
output       PASS max_abs=5.960e-07
batch        PASS max_abs=4.657e-07
```

环境：Python 3.13.7、NumPy 2.5.5、Pillow 12.3.0。

## 报告

`report/` 下是实验报告的 LaTeX 源码和编译好的 PDF。
