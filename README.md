# SAM3 Image Recognition Service

一个基于 `FastAPI` 的图片识别服务封装，底层推理使用 Meta 的 `SAM 3`。当前版本走的是“上传图片 + 文本提示词”的模式，返回检测框、分数，以及可选的二值掩码 RLE。

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
- 已申请并登录 Hugging Face 的 `facebook/sam3` 权限

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

如果你走 Hugging Face 自动下载 checkpoint，记得先登录：

```bash
hf auth login
```

如果你已经有本地权重，也可以直接设置：

```bash
export SAM3_CHECKPOINT_PATH=/path/to/sam3.pt
```

## 启动

```bash
uv run sam3-image-service
```

默认监听 `0.0.0.0:8000`。

## 环境变量

- `APP_HOST`: 服务监听地址，默认 `0.0.0.0`
- `APP_PORT`: 服务端口，默认 `8000`
- `APP_LOG_LEVEL`: 日志级别，默认 `info`
- `SAM3_DEVICE`: `auto` / `cuda` / `cpu`，默认 `auto`
- `SAM3_CHECKPOINT_PATH`: 本地权重路径，可选
- `SAM3_LOAD_FROM_HF`: 未提供本地权重时，是否从 Hugging Face 拉取，默认 `true`
- `SAM3_ENABLE_COMPILE`: 是否启用模型 compile，默认 `false`
- `SAM3_SCORE_THRESHOLD`: 默认分数阈值，默认 `0.25`
- `SAM3_MASK_THRESHOLD`: 默认掩码阈值，默认 `0.5`
- `SAM3_INCLUDE_MASKS_BY_DEFAULT`: 默认是否返回掩码，默认 `false`
- `SAM3_WARMUP_ON_STARTUP`: 启动时是否预热模型，默认 `false`

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
- 目前接口先聚焦图片场景，后续要扩视频版可以继续接 `build_sam3_video_predictor`。
# sam3-api
