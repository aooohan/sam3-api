# SAM3 Image Recognition Service

一个基于 `FastAPI` 的图片识别服务封装，底层推理使用 Meta 的 `SAM 3`。当前版本走的是“上传图片 + 文本提示词”的模式，返回检测框、分数，以及可选的二值掩码 RLE。

当前默认行为已经改成：

- 服务启动时自动检查本地 `models/sam3/`
- 如果本地没有 `sam3.pt`，就从 Hugging Face 下载到这个目录
- 后续模型加载统一只走本地 checkpoint，不再把 `build_sam3_image_model` 直接指向 Hugging Face
- 当前默认下载端点已经切到 `https://hf-mirror.com`

## 已完成

- `uv` 项目初始化
- `FastAPI` 服务骨架
- `sam3` 推理层独立封装
- 健康检查接口
- 图片识别接口
- 不依赖真实模型权重的接口测试
- `uv` 默认镜像改为腾讯云，`torch` 单独走官方 `cu126` 源

## 目录结构

```text
src/sam3_image_service/
  app.py         # FastAPI 应用和路由
  backend.py     # SAM3 推理封装
  config.py      # 环境变量配置
  schemas.py     # 接口输入/输出结构
```

## 先决条件

根据 SAM 3 官方说明，建议环境为：

- Python 3.12+
- PyTorch 2.7+
- CUDA 12.6+
- 如果你要自动下载权重，需要已申请并登录 Hugging Face 的 `facebook/sam3` 权限

官方参考：

- SAM 3 GitHub: https://github.com/facebookresearch/sam3
- SAM 3 Hugging Face: https://huggingface.co/facebook/sam3

## 安装

先装基础服务依赖：

```bash
uv sync --group dev
```

如果你要真正跑 `sam3` 推理，再安装运行时依赖：

```bash
uv sync --group dev --extra runtime
```

如果你要让服务自动下载权重到本地目录，记得先登录：

```bash
hf auth login
```

如果你已经有本地权重，也可以直接设置：

```bash
export SAM3_CHECKPOINT_PATH=/path/to/sam3.pt
```

如果你不想依赖 `hf auth login`，也可以直接提供 token：

```bash
export HF_TOKEN=hf_xxx
```

项目默认会使用下面这个镜像端点：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 启动

```bash
uv run sam3-image-service
```

默认监听 `0.0.0.0:8000`。

第一次启动时，如果本地还没有模型文件，服务会把 `sam3.pt` 下载到 `models/sam3/`，所以启动会比平时慢一些。下载完成后，后续重启会直接复用本地文件。

## 环境变量

- `APP_HOST`: 服务监听地址，默认 `0.0.0.0`
- `APP_PORT`: 服务端口，默认 `8000`
- `APP_LOG_LEVEL`: 日志级别，默认 `info`
- `SAM3_DEVICE`: `auto` / `cuda` / `cpu`，默认 `auto`
- `SAM3_CHECKPOINT_PATH`: 本地权重路径，可选
- `SAM3_MODEL_DIR`: 自动下载到本地后的模型目录，默认 `models/sam3`
- `SAM3_CHECKPOINT_FILENAME`: checkpoint 文件名，默认 `sam3.pt`
- `SAM3_HF_MODEL_ID`: Hugging Face 模型仓库，默认 `AnantP78/sam3_pt`
- `HF_ENDPOINT` / `SAM3_HF_ENDPOINT`: Hugging Face 下载端点，默认 `https://hf-mirror.com`
- `HF_TOKEN` / `SAM3_HF_TOKEN`: Hugging Face token，可选
- `SAM3_LOAD_FROM_HF`: 本地没有权重时，是否允许自动下载，默认 `true`
- `SAM3_DOWNLOAD_ON_STARTUP`: 启动时是否预下载模型到本地目录，默认 `true`
- `SAM3_FORCE_DOWNLOAD`: 是否强制重新下载本地模型，默认 `false`
- `SAM3_LOCAL_FILES_ONLY`: 只读本地缓存，不访问 Hugging Face，默认 `false`
- `SAM3_ENABLE_COMPILE`: 是否启用模型 compile，默认 `false`
- `SAM3_SCORE_THRESHOLD`: 默认分数阈值，默认 `0.25`
- `SAM3_MASK_THRESHOLD`: 默认掩码阈值，默认 `0.5`
- `SAM3_INCLUDE_MASKS_BY_DEFAULT`: 默认是否返回掩码，默认 `false`
- `SAM3_WARMUP_ON_STARTUP`: 启动时是否预热模型，默认 `false`

## 推荐用法

### 方案 1：启动时自动下载到本地

```bash
export HF_TOKEN=hf_xxx
uv run sam3-image-service
```

第一次启动会下载：

```text
models/sam3/
  config.json
  sam3.pt
```

后续服务始终从本地 `models/sam3/sam3.pt` 加载。

当前默认仓库已经切到：

```text
AnantP78/sam3_pt
```

也支持直接传完整仓库 URL，例如：

```bash
export SAM3_HF_MODEL_ID=https://huggingface.co/AnantP78/sam3_pt/
```

如果你想显式写出来，也可以这样启动：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=hf_xxx
uv run sam3-image-service
```

### 方案 2：完全手动管理本地权重

```bash
export SAM3_CHECKPOINT_PATH=/data/models/sam3.pt
export SAM3_LOAD_FROM_HF=false
uv run sam3-image-service
```

这时服务不会访问 Hugging Face。

## API

### 健康检查

```bash
curl http://127.0.0.1:8000/healthz
```

### 图片识别

```bash
curl -X POST http://127.0.0.1:8000/v1/recognize \
  -F "file=@./demo.jpg" \
  -F "prompt=white dog" \
  -F "include_masks=true"
```

返回示例：

```json
{
  "prompt": "white dog",
  "device": "cuda",
  "image_size": {
    "width": 1280,
    "height": 720
  },
  "detections": [
    {
      "index": 0,
      "label": "white dog",
      "score": 0.97,
      "area": 102345,
      "box": {
        "x1": 103.1,
        "y1": 88.4,
        "x2": 922.6,
        "y2": 640.9
      },
      "mask": {
        "size": [720, 1280],
        "counts": [0, 12, 8, 41]
      }
    }
  ],
  "took_ms": 321.4
}
```

## 说明

- 这个服务封装的是 `SAM 3` 的开放词汇检测/分割能力，不是传统闭集分类器。
- 如果没装 `runtime` 额外依赖，接口会返回明确的 `503` 提示，不会静默失败。
- 当前服务默认会先把 checkpoint 缓存到本地，再用本地路径加载模型，这样比每次直接走远端更稳。
- 目前接口先聚焦图片场景，后续要扩视频版可以继续接 `build_sam3_video_predictor`。
