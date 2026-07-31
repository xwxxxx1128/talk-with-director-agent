# FilmLens Agent Console（李安口吻影视分析 Agent）
基于 [Agno]的影视镜头 / 文字方案分析 Agent，以**李安的口吻**回答。后端用 FastAPI，前端用 React + Vite + MUI 的企业风格控制台。对话模型为**本地微调后的 `liang` 模型**（Qwen2.5-7B-Instruct + LoRA 合并，经 vLLM 以 OpenAI 兼容接口暴露）。

![效果展示](image.png)
![效果展示](image-1.png)

> **微调模型**：对话模型基于 **Qwen2.5-7B-Instruct** 使用李安相关语料进行 **LoRA 微调并合并** 得到的真实模型权重。
> 微调合并后的完整模型已发布至 ModelScope：
> **https://www.modelscope.cn/models/xwxxxx1/qwen2.5-7b-liang-merged**

基于 [Agno]的影视镜头 / 文字方案分析 Agent，以**李安的口吻**回答。后端用 FastAPI，前端用 React + Vite + MUI 的企业风格控制台。对话模型为**本地微调后的 `liang` 模型**（Qwen2.5-7B-Instruct + LoRA 合并，经 vLLM 以 OpenAI 兼容接口暴露）。

## 架构总览

```
用户(浏览器) ──> 前端 React(MUI) ──HTTP/JSON──> 后端 FastAPI(app.py)
                                                    │
                       ┌────────────────────────────┼───────────────────────────┐
                       │                            │                           │
                 ① 对话 / 文字方案          ② 图片分析(两步式)             ③ RAG 检索
                 直接发给 liang             视觉模型先出画面描述          本地句向量(无外部embedding服务)
                                                       │                  sentence-transformers
                                                       └─> 描述文本 ─> liang
```

- **对话模型 `liang`**：纯文本模型（Qwen2.5-7B-Instruct 基底），通过 vLLM 的 `/v1/chat/completions` 调用。
- **图片分析（两步式）**：`liang` 是文本模型，不能直接看图。流程为：
  1. 用**视觉模型**（可配 `vision_model` / `vision_base_url`，如 qwen2.5-vl、gpt-4o-mini 等）把图片转成中性、具体的画面描述；
  2. 把「描述文本 + 用户问题 + RAG 语料」作为纯文本交给 `liang`，由其用李安口吻输出。
  - 这样 GPU 上只需常驻 `liang` 一个模型，视觉描述按需走你配置的中转站/视觉 API。
- **RAG 检索**：默认使用**本地句向量**（`sentence-transformers` + `BAAI/bge-small-zh-v1.5`），语料来自 `output/paragraphs.jsonl`，向量缓存到 `output/embed_cache/`。无需任何外部 embedding 服务；若模型不可下载则自动降级为关键词检索。
- **对话历史**：SQLite（`film_analysis.db`）持久化。

## 环境要求

- Python 3.10+
- Node.js 18+（构建前端）
- 一块能跑 7B 模型的 GPU（用于 vLLM 部署 `liang`；若无 GPU 可改用外部中转站/API）

## 安装依赖

```bash
# 后端（含 vllm、agno、sentence-transformers、lancedb 等）
pip install -r requirements.txt

# 前端
cd frontend
npm install
npm run build      # 产物输出到 frontend/dist，由后端直接托管
```

> 首次启动后端时，RAG 会自动从 HuggingFace 下载 `BAAI/bge-small-zh-v1.5`（约 130MB）。
> 若下载慢可设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`

## 运行步骤

### 1. 启动对话模型（vLLM，端口 8000）

```bash
vllm serve /path/to/qwen2.5-7b-liang-merged \
  --served-model-name liang \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 8192 --gpu-memory-utilization 0.9
```

等待日志出现 `Application startup complete` 即就绪。`--served-model-name liang` 必须与前端的「模型 ID」一致。

### 2. 启动后端（端口 7860，避免与 vLLM 冲突）

```bash
cd /home/xwx/ang-lee
python app.py
# 默认 host=0.0.0.0, port=7860, reload=True
```

后端会：
- 若 `frontend/dist` 存在，则在 `7860` 端口托管前端页面；
- 初始化 RAG（首次运行会下载句向量模型并计算缓存）。

### 3. 打开前端

- 已构建：`http://localhost:7860`（后端直接托管）
- 或开发模式：`cd frontend && npm run dev`（Vite 默认 5173，需在前端里把 API 指到 `http://localhost:7860`）

### 4. 在配置面板填写

| 字段 | 说明 | 示例 |
| --- | --- | --- |
| OpenAI API Key | vLLM 不校验，填任意值即可 | `sk-local` |
| 模型 ID | 必须 = vLLM 的 `--served-model-name` 
| OMDb API Key | 可选，用于电影 IMDB/导演信息 | — |
| API 中转站 | vLLM 的 OpenAI 兼容地址 | `http://localhost:8000/v1` |
| 视觉模型（图片描述用） | 图片分析第一步所需，留空则图片功能不可用 | `qwen2.5-vl-7b` / `gpt-4o-mini` |
| 视觉模型 API 地址 | 留空则复用上方中转站 | — |

保存后 Agent 立即初始化。

## 功能特性

### 📷 图片分析（两步式）
上传电影截图/剧照，系统先由视觉模型生成画面描述（构图、光线、色彩、人物/空间、情绪氛围），再交给 `liang` 以李安口吻分析。维度包括：
- 景别与构图（远景、全景、中景、近景、特写）
- 光线方向与质感（顺光、逆光、侧光、软/硬光）
- 主色调与色彩情绪
- 人物/主体的空间关系
- 视觉元素如何服务叙事

### ✍️ 文字方案生成
输入场景描述，Agent 生成拍摄方案：机位角度、镜头焦段、布光、人物调度、情绪表达强化。

### 🔧 技术特性
- **RAG 语义检索**：本地句向量（无外部 embedding 服务），降级关键词检索
- **对话历史**：SQLite 持久化
- **外部 API**：可选的 OMDb（电影信息）
- **自定义配置**：模型名 / API 地址 / 视觉模型均可配置

## 接口说明（后端 FastAPI）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/config` | 读取当前配置（API Key 已脱敏） |
| POST | `/api/config` | 保存并初始化 Agent（含 `vision_model`/`vision_base_url`） |
| POST | `/api/message` | 文本对话，body: `{message, use_rag}` |
| POST | `/api/upload-image` | 图片分析，form: `file`, `question` |
| GET | `/api/last-analysis` | 获取最近一次分析 |
| GET | `/api/history` | 对话历史 |

## 项目结构

```
ang-lee/
├── app.py                  # 后端主程序（FastAPI + Agno），端口 7860
├── rag_retriever.py        # LiAngRAG：本地句向量检索 + 关键词降级
├── image_utils.py          # 图片处理辅助
├── process_mobi_book.py    # 语料处理与 embedding 工具
├── requirements.txt        # Python 依赖（含 vllm、agno、sentence-transformers）
├── output/
│   ├── paragraphs.jsonl    # RAG 语料（李安相关文本）
│   ├── embed_cache/        # 本地向量缓存（首次运行自动生成）
│   └── ...
├── frontend/               # React + Vite + MUI 前端
│   └── dist/               # 构建产物，由后端托管
├── film_analysis.db        # 对话历史（SQLite，自动生成）
└── README.md
```

## 语料说明（版权）

RAG 检索所用的语料 `output/paragraphs.jsonl` **因版权原因不随仓库发布**，请自行准备。

语料主要节选自李安相关传记 / 访谈（如《十年一觉电影梦：李安传》等），以第一人称口吻记录其创作理念、拍摄手法与人生感悟，用于让 `liang` 习得“李安口吻”。示例如下（仅作展示，非全文）：

> - “创作欲好像不是求生，而是求死，是自我解构的一个演化过程。”
> - “当你冒险追求绝对值时，经常处于临界点上，如履薄冰，兴奋感与危机感共生，求生与求死并存。”
> - “我觉得台湾外省人在中国历史上是个比较特殊的文化现象，对于中原文化，他有一种延续，大陆是个新发展，香港又是另一回事。”
> - “长久以来，在我的电影里，结尾都以悲剧收场、以死亡终结，似乎要追求到某种美感才能结束。”

仓库不再包含本地 `checkpoint-40/`（LoRA 微调产物，已通过 `.gitignore` 整体排除）；训练权重合并后的全量模型见上方 ModelScope 链接，直接拉取即可使用。

## 部署说明（一句话）

GPU 上只常驻 `liang`（vLLM，8000 端口）；后端（7860）提供 API 与前端；视觉模型按需走你配置的中转站/视觉 API。RAG 向量在进程内本地计算，不依赖任何外部 embedding 服务。

## 故障排除# 

# 4. 推送
git push


- **Agent 未初始化**：检查「API 中转站」是否指向 vLLM 且 `served-model-name` 与「模型 ID」一致。
- **图片分析 400「未配置视觉模型」**：在配置面板填写 `vision_model`（及其 `vision_base_url`）。
- **RAG 无语义结果 / 日志出现“降级为关键词检索”**：通常是 `BAAI/bge-small-zh-v1.5` 下载失败；检查网络 / `HF_ENDPOINT` 镜像。
- **端口冲突**：vLLM 占 8000，后端固定在 7860（`app.py` 末尾 `uvicorn.run(..., port=7860)`），如需修改请同步修改前端 API 地址。

## 许可证

仅用于学习与研究目的。
