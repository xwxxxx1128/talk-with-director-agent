
"""李安传记语料的运行时 RAG 检索器（替代原有的静态片段注入）。

设计要点：
- 数据来源：output/paragraphs.jsonl（每段带 tag + text）。
- 向量检索：用 OpenAI / Nomic 在线嵌入，缓存为 numpy 矩阵后做余弦相似度检索；
  若嵌入不可用（无 key / 网络失败），自动降级为关键词（lexical）检索，保证永远可运行、有结果。
- 标签过滤：可按意图缩小召回域，降低小语料检索噪声。
- HyDE：可选，用 LLM 先生成「假设性李安回答」再去检索，提升跨表述召回相关性。
- 相关性裁判：仅在相似度/得分超过阈值时返回，避免把无关片段硬塞进 prompt。

注意：本检索器把召回结果当作「参考锚点」交给微调后的模型判断引用与否，
因此只解决「书内有依据」的 in-corpus  grounding；书外事实幻觉仍由
SFT 诚实边界 + Skills 约束 + OMDb 工具负责。
"""

from __future__ import annotations

import json
import re
import hashlib
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

import numpy as np

try:
    from lancedb import connect as _lancedb_connect
except Exception:  # pragma: no cover - lancedb 可选
    _lancedb_connect = None

# 复用 process_mobi_book 中已有的嵌入实现，避免重复造轮子
from process_mobi_book import embed_with_openai, embed_with_nomic, store_embeddings_lancedb

DEFAULT_PARAGRAPHS = Path(__file__).resolve().parent / "output" / "paragraphs.jsonl"
DEFAULT_LANCEDB_DIR = Path(__file__).resolve().parent / "output" / "lancedb"
DEFAULT_EMBED_CACHE = Path(__file__).resolve().parent / "output" / "embed_cache"

# 意图 -> 标签过滤（None 表示不限）。用于收窄小语料召回域，降低噪声。
INTENT_TAG_MAP: Dict[str, Optional[List[str]]] = {
    "A": None,                              # 图片赏析：全标签
    "B": None,                              # 闲聊：全标签
    "C": None,                              # 电影检索：调用方应跳过 RAG
    "D": ["构图", "调度", "光线", "情感控制"],  # 拍摄方案：偏制作向标签
}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _simple_tokenize(text: str) -> List[str]:
    """中文按字、英文按词做轻量分词，用于 lexical 兜底检索。"""
    text = re.sub(r"\s+", "", text)
    return re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text)


class LiAngRAG:
    def __init__(
        self,
        paragraphs_path: Path = DEFAULT_PARAGRAPHS,
        lancedb_dir: Path = DEFAULT_LANCEDB_DIR,
        embed_provider: str = "local",        # "local" | "openai" | "nomic"
        openai_api_key: str = "",
        openai_base_url: str = "",
        embedding_model: str = "text-embedding-3-small",
        nomic_url: str = "http://127.0.0.1:3000/v1/embeddings",
        nomic_model: str = "nomic-embed-text",
        embed_local_model: str = "BAAI/bge-small-zh-v1.5",
        embed_cache_dir: Path = DEFAULT_EMBED_CACHE,
        persist: bool = True,
    ):
        self.paragraphs_path = Path(paragraphs_path)
        self.lancedb_dir = Path(lancedb_dir)
        self.embed_provider = embed_provider
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        self.embedding_model = embedding_model
        self.nomic_url = nomic_url
        self.nomic_model = nomic_model
        self.embed_local_model = embed_local_model
        self.embed_cache_dir = Path(embed_cache_dir)
        self.persist = persist

        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None  # (N, D)
        self._ready = False
        self._local_model = None
        self._local_lock = threading.Lock()

    # ------------------------- 数据加载 -------------------------
    def _load_chunks(self) -> None:
        if not self.paragraphs_path.exists():
            print(f"[RAG] 未找到语料文件：{self.paragraphs_path}，RAG 停用。")
            return
        with self.paragraphs_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                self.chunks.append({
                    "id": item.get("id"),
                    "tag": str(item.get("tag", "综合")),
                    "text": text,
                })

    # ------------------------- 嵌入 -------------------------
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.embed_provider == "nomic":
            return embed_with_nomic(texts, url=self.nomic_url, model=self.nomic_model)
        if self.embed_provider == "local":
            return self._embed_local(texts)
        if self.openai_base_url:
            import openai as _openai
            client = _openai.OpenAI(api_key=self.openai_api_key, base_url=self.openai_base_url)
            out: List[List[float]] = []
            for i in range(0, len(texts), 64):
                resp = client.embeddings.create(model=self.embedding_model, input=texts[i:i + 64])
                out.extend([d.embedding for d in resp.data])
            return out
        return embed_with_openai(texts, self.openai_api_key, model=self.embedding_model)

    # ------------------------- 本地句向量 -------------------------
    def _load_local_model(self):
        """惰性加载本地 sentence-transformers 模型（线程安全，只加载一次）。"""
        if self._local_model is not None:
            return self._local_model
        with self._local_lock:
            if self._local_model is not None:
                return self._local_model
            from sentence_transformers import SentenceTransformer
            print(f"[RAG] 加载本地嵌入模型：{self.embed_local_model} ...")
            model = SentenceTransformer(self.embed_local_model)
            self._local_model = model
            return model

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        model = self._load_local_model()
        vecs = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32).tolist()

    def _local_cache_paths(self):
        self.embed_cache_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", self.embed_local_model)
        npy = self.embed_cache_dir / f"emb_{safe}.npy"
        meta = self.embed_cache_dir / f"emb_{safe}.meta.json"
        return npy, meta

    def _local_meta_ok(self, npy, meta) -> bool:
        if not (npy.exists() and meta.exists()):
            return False
        try:
            meta_info = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            return False
        # 语料文件指纹（路径+大小+修改时间）与模型名都必须一致，否则重建
        try:
            st = self.paragraphs_path.stat()
            fingerprint = f"{self.paragraphs_path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
        except Exception:
            return False
        return (
            meta_info.get("model") == self.embed_local_model
            and meta_info.get("fingerprint") == fingerprint
            and meta_info.get("ndim") is not None
        )

    def ensure_ready(self) -> None:
        """准备检索索引；任何嵌入失败都降级到关键词检索，不抛异常。"""
        if self._ready:
            return
        self._load_chunks()
        if self.chunks:
            try:
                # 本地嵌入：优先命中磁盘缓存，避免每次冷启动重新计算
                if self.embed_provider == "local":
                    npy, meta = self._local_cache_paths()
                    if self._local_meta_ok(npy, meta):
                        print(f"[RAG] 命中本地嵌入缓存：{npy}")
                        self.embeddings = np.load(str(npy))
                        self._ready = True
                        return

                texts = [c["text"] for c in self.chunks]
                embs = self._embed_texts(texts)
                self.embeddings = np.asarray(embs, dtype=np.float32)

                if self.embed_provider == "local":
                    try:
                        npy, meta = self._local_cache_paths()
                        np.save(str(npy), self.embeddings)
                        st = self.paragraphs_path.stat()
                        meta.write_text(json.dumps({
                            "model": self.embed_local_model,
                            "fingerprint": f"{self.paragraphs_path.resolve()}|{st.st_size}|{int(st.st_mtime)}",
                            "ndim": int(self.embeddings.shape[1]),
                        }, ensure_ascii=False), encoding="utf-8")
                        print(f"[RAG] 本地嵌入已缓存到：{npy}")
                    except Exception as exc:
                        print(f"[RAG] 写入本地嵌入缓存失败（不影响检索）：{exc}")

                if self.persist:
                    self._maybe_persist(embs)
            except Exception as exc:
                print(f"[RAG] 嵌入不可用，降级为关键词检索：{exc}")
                self.embeddings = None
        self._ready = True

    def _maybe_persist(self, embs: List[List[float]]) -> None:
        try:
            store_embeddings_lancedb([c["text"] for c in self.chunks], self.lancedb_dir.parent, embs)
        except Exception:
            pass  # 持久化失败不影响本次检索

    # ------------------------- HyDE -------------------------
    def hyde(self, query: str, llm_callable: Optional[Callable[[str], str]] = None) -> str:
        if llm_callable is None:
            return query
        try:
            hypo = llm_callable(
                "你是李安。请以你第一人称、克制谦逊的口吻，假设你要回答下面这个问题，"
                "写一段简短（2-3 句）的回答，不要引用具体电影名。\n问题：" + query
            )
            if hypo and len(hypo.strip()) > 5:
                return hypo.strip()
        except Exception as exc:
            print(f"[RAG] HyDE 生成失败，使用原问题检索：{exc}")
        return query

    # ------------------------- 检索 -------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.15,
        tag_filter: Optional[List[str]] = None,
        use_hyde: bool = False,
        llm_callable: Optional[Callable[[str], str]] = None,
    ) -> List[Dict[str, Any]]:
        self.ensure_ready()
        if not self.chunks:
            return []

        effective_query = self.hyde(query, llm_callable) if use_hyde else query

        # 1) 向量检索（余弦相似度，口径一致、可靠）
        if self.embeddings is not None:
            q_emb = np.asarray(self._embed_query(effective_query), dtype=np.float32)
            sims = np.asarray([_cosine(q_emb, e) for e in self.embeddings])
            order = np.argsort(-sims)
            results: List[Dict[str, Any]] = []
            for idx in order:
                sim = float(sims[idx])
                if sim < threshold:
                    continue
                chunk = self.chunks[idx]
                if tag_filter and chunk["tag"] not in tag_filter:
                    continue
                results.append({**chunk, "score": round(sim, 4)})
                if len(results) >= top_k:
                    break
            if results:
                return results

        # 2) 关键词兜底检索
        return self._retrieve_lexical(effective_query, top_k, threshold, tag_filter)

    def _embed_query(self, text: str) -> List[float]:
        return self._embed_texts([text])[0]

    def _retrieve_lexical(self, query: str, top_k: int, threshold: float,
                          tag_filter: Optional[List[str]]) -> List[Dict[str, Any]]:
        q_tok = _simple_tokenize(query)
        if not q_tok:
            return []
        q_set = set(q_tok)
        scored: List[Dict[str, Any]] = []
        for c in self.chunks:
            if tag_filter and c["tag"] not in tag_filter:
                continue
            overlap = len(q_set & set(_simple_tokenize(c["text"])))
            if overlap == 0:
                continue
            score = overlap / max(1, len(q_tok))
            if score < threshold:
                continue
            scored.append({**c, "score": round(score, 4)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # ------------------------- 上下文拼装 -------------------------
    @staticmethod
    def format_context(results: List[Dict[str, Any]], max_chars_per: int = 400) -> str:
        if not results:
            return ""
        lines = ["## 李安原著参考（仅当相关时才引用，无关则忽略）"]
        for r in results:
            snippet = re.sub(r"\s+", " ", r["text"])[:max_chars_per]
            lines.append(f"- [{r.get('tag', '综合')}] {snippet}")
        lines.append(
            "- 注意：以上为李安原著片段，请仅在与问题相关时引用，"
            "不要编造其中未提及的具体电影、言论或数据。"
        )
        return "\n".join(lines)