#!/usr/bin/env python3
"""统计李安语料库（output/paragraphs.jsonl）的字符数 / Token 数，并与主流模型上下文窗口对比，
帮助判断「全量上下文注入」还是「RAG 向量检索」更合适。

用法：
    python count_corpus_tokens.py
    python count_corpus_tokens.py --method tiktoken
    python count_corpus_tokens.py --method transformers --model Qwen/Qwen2.5-7B-Instruct
    python count_corpus_tokens.py --system-budget 1500 --shot-budget 1500

说明：
- 默认 method=heuristic：中文/混合文本用「字符数 / 1.7」估算（Qwen2.5 系常见比率），
  无第三方依赖，适合快速评估。
- tiktoken：用 o200k_base（GPT-4o 同款，中文偏乐观），需 pip install tiktoken。
- transformers：加载真实 Qwen2.5 tokenizer（最准，但首次需下载）。
"""

import argparse
import json
from pathlib import Path

# 常见模型的安全上下文窗口（token 数）。训练窗口较小，长上下文多靠 YaRN/推理扩展。
CONTEXT_WINDOWS = {
    "Qwen2.5 (训练 32K)": 32_768,
    "Qwen2.5 (扩展 128K)": 131_072,
    "GPT-4o / 4o-mini": 128_000,
    "Qwen2.5-VL (多模态)": 32_768,
}

DEFAULT_PARAGRAPH_PATH = Path(__file__).resolve().parent / "output" / "paragraphs.jsonl"
CHARS_PER_TOKEN_HEURISTIC = 1.7  # 中文混合文本经验值


def load_paragraphs(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"未找到语料文件：{path}（请先运行 process_mobi_book.py）")
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def count_tokens_heuristic(text: str) -> int:
    return max(1, round(len(text) / CHARS_PER_TOKEN_HEURISTIC))


def count_tokens_tiktoken(text: str) -> int:
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


def count_tokens_transformers(text: str, model: str) -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    return len(tok.encode(text, add_special_tokens=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="统计语料库 Token 数并判断全量/检索策略")
    parser.add_argument("--paragraphs", type=Path, default=DEFAULT_PARAGRAPH_PATH)
    parser.add_argument("--method", choices=["heuristic", "tiktoken", "transformers"], default="heuristic")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="method=transformers 时使用的 tokenizer 模型")
    parser.add_argument("--system-budget", type=int, default=1500,
                        help="预留给 system prompt / 人格约束的 token 预算")
    parser.add_argument("--shot-budget", type=int, default=1500,
                        help="预留给 few-shot 口吻锚点的 token 预算")
    args = parser.parse_args()

    records = load_paragraphs(args.paragraphs)
    if not records:
        print("语料为空。")
        return

    total_chars = 0
    total_tokens = 0
    per_tag_chars: dict[str, int] = {}
    per_tag_tokens: dict[str, int] = {}

    for rec in records:
        text = rec.get("text", "")
        chars = len(text)
        if args.method == "tiktoken":
            tokens = count_tokens_tiktoken(text)
        elif args.method == "transformers":
            tokens = count_tokens_transformers(text, args.model)
        else:
            tokens = count_tokens_heuristic(text)
        total_chars += chars
        total_tokens += tokens
        tag = rec.get("tag", "综合")
        per_tag_chars[tag] = per_tag_chars.get(tag, 0) + chars
        per_tag_tokens[tag] = per_tag_tokens.get(tag, 0) + tokens

    avg_tokens = total_tokens / len(records)
    overhead = args.system_budget + args.shot_budget
    corpus_with_overhead = total_tokens + overhead

    print("=" * 64)
    print("语料库规模统计")
    print("=" * 64)
    print(f"段落数量        : {len(records)}")
    print(f"总字符数        : {total_chars:,} 字符")
    print(f"总 Token 数     : {total_tokens:,} token  (method={args.method})")
    print(f"平均每段        : {avg_tokens:,.1f} token / {total_chars/len(records):,.0f} 字符")
    print()
    print("按标签分布（tag 过滤检索的召回域参考）：")
    for tag in sorted(per_tag_tokens, key=lambda t: per_tag_tokens[t], reverse=True):
        print(f"  - {tag:<8} {per_tag_tokens[tag]:>8,} token  ({per_tag_chars[tag]:>8,} 字符)")
    print()

    print("=" * 64)
    print("与上下文窗口对比（含 system/few-shot 预留）")
    print("=" * 64)
    for name, window in CONTEXT_WINDOWS.items():
        fits_full = corpus_with_overhead <= window
        remainder = window - corpus_with_overhead
        status = "✅ 可全量注入" if fits_full else f"❌ 超窗 {abs(remainder):,} token"
        print(f"  {name:<22} 窗口 {window:>8,} | 语料+预留 {corpus_with_overhead:>8,} | {status}")
    print()

    print("=" * 64)
    print("策略建议")
    print("=" * 64)
    if corpus_with_overhead <= CONTEXT_WINDOWS["Qwen2.5 (训练 32K)"]:
        print("结论：语料可完整塞进模型上下文窗口（32K 训练窗口内）。")
        print("  → 推荐「全量上下文注入」，而非 RAG 向量检索：")
        print("     1) 零检索噪声、零生硬拼接；")
        print("     2) 覆盖所有在书内出现过的话题（覆盖率上限最高）；")
        print("     3) 真正会'瞎编'的只剩书外话题——那靠 SFT 谦逊边界 + Skills 约束 + OMDb 工具，而非 RAG。")
    elif corpus_with_overhead <= CONTEXT_WINDOWS["Qwen2.5 (扩展 128K)"]:
        print("结论：语料在 32K 训练窗口之外，但 128K 扩展窗口内。")
        print("  → 关键决策：是否开长上下文推理（YaRN/推理扩展）。")
        print("     若开启：仍可全量注入，注意长上下文可能退化，需测试；")
        print("     若不开：改用 RAG，但必须配合 tag 过滤 + HyDE + 相关性裁判，避免小语料检索噪声。")
    else:
        print("结论：语料超过 128K，必须做 RAG 检索（无法全量注入）。")
        print("  → 建议检索结果仅作'参考锚点'、由模型判断是否引用，降低生硬与噪声。")


if __name__ == "__main__":
    main()
