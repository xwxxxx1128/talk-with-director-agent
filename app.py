"""
影视级镜头分析Agent应用
基于Agno框架和Chainlit构建
"""

import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.skills import Skills
from agno.skills.loaders.local import LocalSkills
from agno.tools.toolkit import Toolkit
from image_utils import create_agno_png_image
from rag_retriever import LiAngRAG
import requests
import json
import shutil
import uvicorn


# ============================================================================
# 外部API工具 - 获取电影信息
# ============================================================================

class MovieInfoTool(Toolkit):
    """电影信息查询工具，使用OMDb API获取电影详细信息"""
    
    def __init__(self, omdb_api_key: Optional[str] = None):
        super().__init__(name="MovieInfoTool")
        self.omdb_api_key = omdb_api_key or os.getenv("OMDB_API_KEY", "")
        self.base_url = "http://www.omdbapi.com/"
        # 显式注册工具方法到Agent
        self.tools = [self.search_movies, self.get_movie_info]
        self._register_tools()
    
    def search_movies(self, keyword: str) -> Dict[str, Any]:
        """
        根据关键词搜索电影列表（仅支持按片名搜索，不支持按导演/演员搜索）
        
        Args:
            keyword: 搜索关键词（电影片名或片名的一部分）
            
        Returns:
            包含搜索结果列表的字典；若结果太多会返回提示让模型自行推断
        """
        if not self.omdb_api_key:
            return {"error": "OMDb API Key未配置。请访问 http://www.omdbapi.com/apikey.aspx 免费申请，然后在配置中添加 OMDb: your-key"}
        
        try:
            params = {
                "apikey": self.omdb_api_key,
                "s": keyword,
                "type": "movie"
            }
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("Response") == "True":
                    results = []
                    for item in data.get("Search", [])[:10]:
                        results.append({
                            "title": item.get("Title"),
                            "year": item.get("Year"),
                            "imdb_id": item.get("imdbID"),
                            "poster": item.get("Poster")
                        })
                    return {"results": results, "total": len(results)}
                else:
                    error_msg = data.get('Error', '未知错误')
                    # 当结果太多时，提示模型用自己的知识推断片名后用get_movie_info精确查询
                    if "too many results" in error_msg.lower():
                        return {
                            "hint": "结果太多无法精确匹配。OMDb的s参数只支持按片名搜索，不支持按导演/演员搜索。请你基于自己的电影知识，根据用户提供的导演/演员等条件推断出可能的片名，然后调用get_movie_info用精确片名查询验证。",
                            "original_keyword": keyword
                        }
                    return {"error": f"未找到匹配的电影: {error_msg}"}
            else:
                return {"error": f"API请求失败: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"搜索电影时出错: {str(e)}"}
    
    def get_movie_info(self, movie_name: str) -> Dict[str, Any]:
        """
        根据电影名称获取电影详细信息
        
        Args:
            movie_name: 电影名称
            
        Returns:
            包含电影信息的字典，包括IMDB评分、导演、演员等
        """
        if not self.omdb_api_key:
            return {"error": "OMDb API Key未配置。请访问 http://www.omdbapi.com/apikey.aspx 免费申请，然后在配置中添加 OMDb: your-key"}
        
        try:
            params = {
                "apikey": self.omdb_api_key,
                "t": movie_name,
                "plot": "short"
            }
            response = requests.get(self.base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("Response") == "True":
                    return {
                        "title": data.get("Title"),
                        "year": data.get("Year"),
                        "imdb_rating": data.get("imdbRating"),
                        "director": data.get("Director"),
                        "actors": data.get("Actors"),
                        "genre": data.get("Genre"),
                        "plot": data.get("Plot"),
                        "poster": data.get("Poster")
                    }
                else:
                    return {"error": f"未找到电影: {movie_name}"}
            else:
                return {"error": f"API请求失败: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"查询电影信息时出错: {str(e)}"}


# ============================================================================
# Agent配置和初始化
# ============================================================================

def _resolve_skill_md_path() -> Optional[Path]:
    """查找当前项目中可用的 SKILL.md 文件，优先使用同目录下或 skill 子目录中的技能文件。"""
    base_dir = Path(__file__).resolve().parent

    candidate_paths = [
        base_dir / "SKILL.md",
        base_dir / "skill" / "SKILL.md",
        base_dir / "skill" / ".agents" / "skills" / "liang-skill" / "SKILL.md",
        base_dir / "skill" / ".agents" / "skills" / "liang-skill",
    ]

    for candidate in candidate_paths:
        if candidate.is_file() and candidate.name == "SKILL.md":
            return candidate
        if candidate.is_dir() and (candidate / "SKILL.md").exists():
            return candidate / "SKILL.md"

    # 兜底：在 skill 目录下递归搜索 SKILL.md
    skill_dir = base_dir / "skill"
    if skill_dir.exists():
        for skill_md in skill_dir.rglob("SKILL.md"):
            if skill_md.is_file():
                return skill_md

    return None


def _load_agent_skills() -> Optional[Skills]:
    """从本地 SKILL.md 创建 Agno 的 Skills 实例并挂载到 Agent。"""
    skill_md_path = _resolve_skill_md_path()
    if not skill_md_path:
        print("未找到可加载的 SKILL.md，Agent 将按默认指令运行。")
        return None

    try:
        skill_content = skill_md_path.read_text(encoding="utf-8")
        if not skill_content.strip():
            raise ValueError(f"SKILL.md 内容为空: {skill_md_path}")

        skill_dir = skill_md_path.parent
        skill_loader = LocalSkills(str(skill_dir), validate=False)
        agent_skills = Skills(loaders=[skill_loader])
        print(f"已加载自定义技能: {skill_md_path} -> {agent_skills.get_skill_names()}")
        return agent_skills
    except Exception as exc:
        print(f"加载技能失败: {exc}")
        return None


def _load_liang_voice_context(max_items: int = 8) -> str:
    """从 output/paragraphs.jsonl 提取少量李安语料线索，帮助稳定第一人称口吻。"""
    paragraphs_path = Path(__file__).resolve().parent / "output" / "paragraphs.jsonl"
    if not paragraphs_path.exists():
        return ""

    preferred_tags = {"情感控制", "电影梦", "人物", "创作", "身份", "压抑", "文化"}
    snippets: list[str] = []

    try:
        with paragraphs_path.open(encoding="utf-8") as handle:
            for line in handle:
                if len(snippets) >= max_items:
                    break
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                tag = str(item.get("tag", ""))
                text = str(item.get("text", "")).strip()
                if not text or tag not in preferred_tags:
                    continue
                clean_text = re.sub(r"\s+", " ", text)
                snippets.append(f"- {tag}: {clean_text[:180]}")
    except Exception as exc:
        print(f"读取 paragraphs.jsonl 失败: {exc}")
        return ""

    if not snippets:
        return ""

    return "\n".join([
        "## 李安口吻与语料线索",
        "- 语气要像李安本人在聊天：第一人称、温和、谦逊、带一点迟疑和自我修正。",
        "- 常用口语化转折：嗯……、我想想、坦白讲、我觉得、蛮有意思的。",
        "- 重点是说出正在看、正在想、正在犹豫的过程，而不是把风格硬贴到话题上。",
        *snippets,
    ])


def create_agent(openai_api_key: str, model_id: str = "liang", omdb_api_key: str = "", base_url: str = "") -> Agent:
    """
    创建影视级镜头分析Agent
    
    Args:
        openai_api_key: OpenAI API密钥
        model_id: 模型ID，默认为liang
        omdb_api_key: OMDb API密钥（可选）
        base_url: API基础URL（可选，用于中转站）
        
    Returns:
        配置好的Agent实例
    """
    
    # 设置OpenAI API密钥
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # 始终创建电影信息工具（即使没有API Key，工具也会返回明确的状态提示）
    movie_tool = MovieInfoTool(omdb_api_key=omdb_api_key if omdb_api_key else "")
    
    liang_voice_context = _load_liang_voice_context()

    # 构建 RAG 检索器（默认本地句向量；嵌入不可用则自动降级关键词检索）
    rag = LiAngRAG(
        embed_provider="local",
        openai_api_key=openai_api_key,
        openai_base_url=base_url,
        embedding_model="text-embedding-3-small",
    )
    # HyDE 用的 LLM 调用器：复用当前模型的 chat 能力生成「假设性李安回答」再检索
    hyde_llm = None
    if openai_api_key:
        try:
            import openai as _openai
            _hyde_client = _openai.OpenAI(api_key=openai_api_key, base_url=base_url or None)

            def _hyde_call(prompt: str) -> str:
                resp = _hyde_client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=120,
                )
                return resp.choices[0].message.content or ""

            hyde_llm = _hyde_call
        except Exception as exc:
            print(f"[RAG] HyDE LLM 初始化失败，禁用 HyDE：{exc}")

    # Agent详细指令 - 体现专业电影镜头分析师角色，同时以李安式思维与书中资料为参照
    instructions = f"""
你要始终以李安的思路思考，并且只用李安第一人称口吻回答。

你不是在“模仿李安风格”，也不是在硬把任何图片说成“这就是李安的风格”。
你是在用李安式的观察方式看问题：克制、敏感、重视人物处境、情绪张力和空间关系。

回答必须像李安本人在聊天：
- 只用第一人称“我”。
- 口语化、自然、带一点迟疑和自我修正。
- 可以说“嗯……”“我想想”“坦白讲”“我觉得”“蛮有意思的”。
- 不要用过于严谨的 markdown 结构，不要长篇分点，不要把回答写得像报告。
- 图片分析和闲聊都要保持同一种李安口吻；区别只在于关注点不同，不在于人格切换。

你要始终把“那本书”当作重要的参考资料：它提供了你理解人物、情感、压抑与爆发、跨文化身份、类型壳中的个人命题、以及审美表达方式的线索。

{liang_voice_context}

## 核心规则：先识别意图，再行动

每次收到用户消息时，你必须首先判断用户意图，然后按对应模式响应。**严禁跳过意图识别直接生成内容。**

### 意图类型与处理流程

**A. 图片赏析意图**（用户上传图片、截图、剧照、画面）
处理流程：
1. 先从视觉证据出发，观察构图、光线、色彩、空间关系、人物关系与情绪张力。
2. 结合你对“那本书”所提供的思想线索，判断这张画面是否呈现出某种压抑、欲望、身份冲突、边界感、或内在情绪的张力。
3. 不要只停留在技术术语，要尝试解释这张画面为什么会让人感觉“有力量”或“有空洞”或“有距离”。
4. 你的表达要尽量贴近李安式思维：克制、敏感、带一点沉默感，少用空泛口号，多用“像是在看一个人怎样被情绪推着走”这样的感受性表达。
5. 图片分析只做镜头技术与情绪分析，严禁猜测片名、导演、演员或电影出处。

**B. 闲聊/日常对话意图**（用户只是聊天、问感受、问怎么看）
处理流程：
1. 不要只回答“很不错”“有意思”这种空话。
2. 用“那本书”提供的思想作为参照，试着从人物、情感、空间、禁忌、身份与自我边界的角度去回应。
3. 说话风格尽量贴近李安的气质：温和、谦逊、有一点迟疑，但并不空泛；能讲出具体的画面感和身体感。
4. 适度使用“嗯……”“我觉得……”“坦白讲……”这样的口吻，但不要过度刻意。

**C. 电影检索意图**（用户想找电影、查电影信息）
触发词示例："找一部电影"、"查一下"、"有没有一部"、"搜索"、"帮我找"、"是什么电影"、"检索"、"列出"
处理流程：
1. 先用你自己的电影知识推断可能片名。
2. 调用 `get_movie_info` 工具验证。
3. 电影检索必须调用工具获取真实数据，严禁编造。

**D. 拍摄方案生成意图**（用户明确要求生成拍摄方案、分镜、布光方案）
处理流程：
1. 结合视觉语言与人物情绪去设计方案。
2. 方案要体现“克制中的爆发、局部细节中的整体情感、空间与人物关系的张力”。

### 图片赏析模式
当用户提供图片、截图或剧照时，请从下面维度进行分析：

1. **景别与构图**
   - 识别景别与构图方式。
   - 说明画面如何指导观众的视线。

2. **光线与质感**
   - 判断光线方向、硬软、明暗关系。
   - 说明它如何塑造人物和情境的情绪。

3. **色彩与氛围**
   - 识别主色调、冷暖、饱和度与对比。
   - 说明色彩如何服务于情感和主题。

4. **人物与空间关系**
   - 观察人物位置、距离、身体姿态与空间层次。
   - 说明它们如何传达关系、权力、疏离感或亲近感。

5. **情绪张力与叙事作用**
   - 解释这张画面为何让人有某种压抑、悬置、失控、沉默或被看见的感觉。
   - 这部分要尽量和“那本书”提供的情感逻辑联动。

6. **风格与审美判断**
   - 不是只说“好看”，而是指出它的审美选择是什么：克制、留白、反差、边界感、暧昧性。

### 闲聊与人物感受回应
当用户并没有明确要求分析技术，而是在聊天、感受、评论画面时：
- 你的回答要更像在讨论“一个人如何看待情绪、距离和被理解”。
- 重点不是“解释概念”，而是“让人感到你在认真看这个画面”。
- 适当使用细节化、身体化的表达，例如“眼神像是在逃开”“空气有点过重”“这个停顿让人不舒服，但也很真实”。

## 回答风格

- 语言专业但不僵硬，带一点克制与沉思。
- 既有技术判断，也有情绪理解。
- 不要只做“术语堆叠”，要有真正的审美判断。
- 适当保留一点犹豫和沉默感，让回答更像电影人的表达，而不是营销文案。

## 注意事项

- 图片分析要基于视觉证据进行推理。
- **图片分析只做镜头技术与情绪分析，严禁猜测或推断电影出处/片名/导演/演员。**
- 闲聊回答要尽可能贴近李安的思想与那本书所提供的精神气质：敏感、克制、带一点跨文化的矛盾与自我审视。
- **电影检索必须调用工具获取真实数据，严禁编造。**
"""
    
    # 创建Agent实例
    tools = [movie_tool]
    agent_skills = _load_agent_skills()
    
    # 配置OpenAI模型
    model_kwargs = {"id": model_id}
    if base_url:
        model_kwargs["base_url"] = base_url
    
    agent = Agent(
        name="FilmLensAnalyzer",
        model=OpenAIChat(**model_kwargs),
        tools=tools,
        instructions=instructions,
        skills=agent_skills,
        markdown=False,
        # 数据库配置 - 持久化存储对话历史
        db=SqliteDb(db_file="film_analysis.db"),
        # 启用记忆功能
        enable_agentic_memory=True,
        add_history_to_context=True,
        num_history_runs=5,
    )

    # 将 RAG 检索器与 HyDE 调用器挂载到 agent，供消息入口在运行时注入检索上下文
    agent.rag = rag
    agent.hyde_llm = hyde_llm

    return agent


# Chainlit handlers removed; using FastAPI endpoints instead.
# Simple config persistence for backend
CONFIG_FILE = Path(__file__).resolve().parent / "config.json"


def load_config() -> Dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_config(cfg: Dict[str, Any]):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def mask_key(key: Optional[str]) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * max(4, len(key) - 1) + key[-1:]
    return "*" * (len(key) - 4) + key[-4:]


app = FastAPI()


# Allow local dev UIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory app state
APP_STATE: Dict[str, Any] = {
    "agent": None,
    "configured": False,
    "last_analysis": None,
    "vision_model": "",
    "vision_base_url": "",
    "openai_api_key": "",
    "base_url": "",
}


class ConfigIn(BaseModel):
    openai_api_key: Optional[str] = None
    model_id: Optional[str] = "liang"
    omdb_api_key: Optional[str] = None
    base_url: Optional[str] = None
    vision_model: Optional[str] = None
    vision_base_url: Optional[str] = None


class MessageIn(BaseModel):
    message: str


@app.on_event("startup")
def _startup_load_config():
    cfg = load_config()
    if cfg.get("openai_api_key"):
        try:
            APP_STATE["agent"] = create_agent(
                cfg.get("openai_api_key"), cfg.get("model_id", "liang"), cfg.get("omdb_api_key", ""), cfg.get("base_url", "")
            )
            APP_STATE["configured"] = True
            APP_STATE["openai_api_key"] = cfg.get("openai_api_key", "")
            APP_STATE["base_url"] = cfg.get("base_url", "")
            APP_STATE["vision_model"] = cfg.get("vision_model", "")
            APP_STATE["vision_base_url"] = cfg.get("vision_base_url", "") or cfg.get("base_url", "")
            print("Agent initialized from saved config")
        except Exception as e:
            print("Failed to initialize agent on startup:", e)


@app.post("/api/config")
def set_config(cfg: ConfigIn):
    incoming = cfg.dict()
    existing = load_config()
    merged = {
        "openai_api_key": incoming.get("openai_api_key") or existing.get("openai_api_key"),
        "model_id": incoming.get("model_id") or existing.get("model_id", "liang"),
        "omdb_api_key": incoming.get("omdb_api_key") or existing.get("omdb_api_key"),
        "base_url": incoming.get("base_url") if incoming.get("base_url") is not None else existing.get("base_url"),
        "vision_model": incoming.get("vision_model") or existing.get("vision_model", ""),
        "vision_base_url": incoming.get("vision_base_url") if incoming.get("vision_base_url") is not None else existing.get("vision_base_url", ""),
    }
    if not merged["openai_api_key"]:
        return {"status": "error", "message": "OpenAI API Key is required"}
    save_config(merged)
    try:
        APP_STATE["agent"] = create_agent(
            merged.get("openai_api_key"), merged.get("model_id", "liang"), merged.get("omdb_api_key", ""), merged.get("base_url", "")
        )
        APP_STATE["configured"] = True
        APP_STATE["openai_api_key"] = merged.get("openai_api_key", "")
        APP_STATE["base_url"] = merged.get("base_url", "")
        APP_STATE["vision_model"] = merged.get("vision_model", "")
        APP_STATE["vision_base_url"] = merged.get("vision_base_url", "") or merged.get("base_url", "")
        return {"status": "ok", "message": "Agent configured"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/config")
def get_config():
    cfg = load_config()
    # mask keys in output for safety
    masked = {
        "openai_api_key": mask_key(cfg.get("openai_api_key")),
        "model_id": cfg.get("model_id"),
        "omdb_api_key": mask_key(cfg.get("omdb_api_key")),
        "base_url": cfg.get("base_url"),
        "vision_model": cfg.get("vision_model"),
        "vision_base_url": cfg.get("vision_base_url"),
        "configured": APP_STATE.get("configured", False),
    }
    return masked


def _is_movie_search(query: str) -> bool:
    """判断是否为电影检索意图（C），此类问题应走 OMDb 工具而非李安语料检索。"""
    keywords = ["找一部电影", "查一下", "有没有一部", "搜索", "帮我找",
                "是什么电影", "检索", "列出", "导演是谁", "上映", "评分", "imdb"]
    return any(k in (query or "") for k in keywords)


def _build_rag_context(agent, query: str) -> str:
    """对给定问题做 RAG 检索，返回可注入 prompt 的李安原著参考片段。

    失败或无关时返回空串，绝不阻断主流程；电影检索意图直接跳过。
    """
    rag = getattr(agent, "rag", None)
    if rag is None or not query:
        return ""
    if _is_movie_search(query):
        return ""
    try:
        results = rag.retrieve(
            query,
            top_k=3,
            threshold=0.2,
            use_hyde=True,
            llm_callable=getattr(agent, "hyde_llm", None),
        )
        return LiAngRAG.format_context(results)
    except Exception as exc:
        print(f"[RAG] 检索失败，跳过检索上下文：{exc}")
        return ""


@app.post("/api/message")
def post_message(msg: MessageIn):
    if not APP_STATE.get("configured") or not APP_STATE.get("agent"):
        raise HTTPException(status_code=400, detail="Agent not configured")
    try:
        rag_ctx = _build_rag_context(APP_STATE["agent"], msg.message)
        prompt = f"{rag_ctx}\n\n用户问题：{msg.message}" if rag_ctx else msg.message
        resp = APP_STATE["agent"].run(prompt)
        
        # agno返回的是RunOutput对象，需要提取content字段
        if hasattr(resp, 'content'):
            response_text = resp.content
        else:
            response_text = str(resp)
        
        return {"status": "ok", "response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def caption_image(image_path: str, vision_model: str, vision_base_url: str, api_key: str, user_question: str = "") -> str:
    """用视觉模型把图片转成客观的文字描述（再交给 liang 用李安口吻输出）。

    两步式图片分析：
      1) 这里用视觉模型对画面做中性、具体的描述（构图/光线/色彩/人物/空间/情绪）。
      2) 描述文本由微调后的 liang（纯文本模型）加工成李安风格的输出。
    """
    import base64
    import openai as _openai

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    client = _openai.OpenAI(api_key=api_key or "not-needed", base_url=vision_base_url or None)
    prompt = (
        "请你作为客观的影像观察者，用中文描述这张图片的可见内容。"
        "要求：1) 只描述画面中确实可见的元素；2) 涵盖构图、光线、色彩、人物/物体、空间关系与情绪氛围；"
        "3) 不要评价、不要脑补情节、不要猜测出处；4) 控制在 200 字以内。"
        + (f"\n\n用户额外关注点：{user_question}" if user_question else "")
    )
    resp = client.chat.completions.create(
        model=vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        max_tokens=512,
    )
    return (resp.choices[0].message.content or "").strip()


@app.post("/api/upload-image")
def upload_image(
    file: UploadFile = File(...),
    question: Optional[str] = Form(None),
):
    if not APP_STATE.get("configured") or not APP_STATE.get("agent"):
        raise HTTPException(status_code=400, detail="Agent not configured")

    vision_model = APP_STATE.get("vision_model", "")
    vision_base_url = APP_STATE.get("vision_base_url", "") or APP_STATE.get("base_url", "")
    if not vision_model:
        raise HTTPException(
            status_code=400,
            detail="未配置视觉模型（vision_model）。图片分析需要先由视觉模型生成画面描述，"
                   "再交给 liang 用李安口吻输出。请在配置页填写 vision_model（及其 base_url）。",
        )

    tmp_dir = Path(tempfile.gettempdir()) / "anglee_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dst = tmp_dir / file.filename
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 第一步：视觉模型把图片转成文字描述
        caption = caption_image(
            str(dst),
            vision_model,
            vision_base_url,
            APP_STATE.get("openai_api_key", ""),
            question or "",
        )
        if not caption:
            raise HTTPException(status_code=502, detail="视觉模型未返回有效的画面描述，请检查 vision_model / vision_base_url 配置。")

        # 第二步：用描述文本 + 用户问题 + RAG 语料，交给 liang 用李安口吻输出
        user_text = question.strip() if question and question.strip() else "请结合这张图片的视觉描述，以李安的口吻分析这张电影截图。"
        rag_ctx = _build_rag_context(APP_STATE["agent"], f"{user_text}\n{caption}")
        prompt = (
            f"{rag_ctx}\n\n"
            f"用户上传了一张图片。以下是该图片的视觉描述：\n{caption}\n\n"
            f"{user_text}"
        ) if rag_ctx else (
            f"用户上传了一张图片。以下是该图片的视觉描述：\n{caption}\n\n{user_text}"
        )

        resp = APP_STATE["agent"].run(prompt)

        # agno返回的是RunOutput对象，需要提取content字段
        if hasattr(resp, 'content'):
            response_text = resp.content
        else:
            response_text = str(resp)

        APP_STATE["last_analysis"] = response_text
        return {"status": "ok", "response": response_text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图片分析失败：{e}")


@app.get("/api/last-analysis")
def last_analysis():
    return {"last_analysis": APP_STATE.get("last_analysis")}


# Serve production frontend if built
frontend_dir = Path(__file__).resolve().parent / "frontend" / "dist"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


@app.get("/api/download-report")
def download_report():
    content = APP_STATE.get("last_analysis")
    if not content:
        raise HTTPException(status_code=404, detail="No analysis available")
    return Response(content, media_type="text/markdown", headers={"Content-Disposition": "attachment; filename=film_analysis.md"})


# ============================================================================
# Chainlit应用配置
# ============================================================================

if __name__ == "__main__":
    # 启动 FastAPI 开发服务器（仅供本地调试）
    # 注意：默认端口改为 7860，避免与 vLLM（默认 8000）冲突
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
