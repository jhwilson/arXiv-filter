import os
import re
import argparse
from typing import Dict, List, Tuple

import numpy as np
import yaml
from sklearn.metrics.pairwise import cosine_similarity

from compute_embeddings import load_preprocessed_papers, generate_embeddings


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")


def _read_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _title_for_file(arxiv_dir: str, filename: str) -> str:
    p = os.path.join(arxiv_dir, filename)
    if not os.path.exists(p):
        return filename
    try:
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
        start = content.find("Title:")
        if start == -1:
            return filename
        start += len("Title:")
        end = content.find("\n", start)
        if end == -1:
            end = len(content)
        title = content[start:end].strip()
        return title or filename
    except Exception:
        return filename


def _rank_arxiv(
    arxiv_emb: np.ndarray,
    my_emb: np.ndarray,
    filenames_arxiv: List[str],
    top_n: int,
) -> List[Tuple[str, float]]:
    sim = cosine_similarity(arxiv_emb, my_emb)
    max_sim = np.max(sim, axis=1)
    order = np.argsort(max_sim)[::-1]
    return [(filenames_arxiv[i], float(max_sim[i])) for i in order[:top_n]]


def _compute_model_rankings(
    model_name: str,
    papers_my: List[str],
    papers_arxiv: List[str],
    filenames_arxiv: List[str],
    top_n: int,
    batch_size: int,
    use_adapters: bool = False,
    adapter_base_dir: str = "",
    adapter_dir: str = "",
) -> List[Tuple[str, float]]:
    emb_my = generate_embeddings(
        papers_my,
        model_name=model_name,
        batch_size=batch_size,
        use_adapters=use_adapters,
        adapter_base_dir=adapter_base_dir,
        adapter_dir=adapter_dir,
    )
    emb_arxiv = generate_embeddings(
        papers_arxiv,
        model_name=model_name,
        batch_size=batch_size,
        use_adapters=use_adapters,
        adapter_base_dir=adapter_base_dir,
        adapter_dir=adapter_dir,
    )
    return _rank_arxiv(np.array(emb_arxiv), np.array(emb_my), filenames_arxiv, top_n=top_n)


def _format_top_block(
    top: List[Tuple[str, float]],
    arxiv_dir: str,
    header: str,
) -> str:
    out = [f"### {header}", ""]
    for i, (fn, score) in enumerate(top, start=1):
        out.append(f"{i}. **{_title_for_file(arxiv_dir, fn)}**")
        out.append(f"   - File: `{fn}`")
        out.append(f"   - Score: `{score:.4f}`")
    out.append("")
    return "\n".join(out)


def _summarize_overlap(
    top_a: List[Tuple[str, float]],
    top_b: List[Tuple[str, float]],
) -> Dict[str, object]:
    set_a = {x[0] for x in top_a}
    set_b = {x[0] for x in top_b}
    overlap = sorted(set_a & set_b)
    union = sorted(set_a | set_b)
    jaccard = (len(overlap) / len(union)) if union else 0.0
    return {
        "overlap_count": len(overlap),
        "jaccard": jaccard,
        "overlap_files": overlap,
    }


def main():
    parser = argparse.ArgumentParser(
        description="A/B test abstract similarity rankings across two embedding models."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--model_b",
        type=str,
        default="thellert/accphysbert_cased",
        help="Second model to compare against the config model (default: PhysBERT variant).",
    )
    parser.add_argument("--top_n", type=int, default=20, help="Top N arXiv items to compare.")
    parser.add_argument("--batch_size", type=int, default=16, help="Embedding batch size.")
    parser.add_argument(
        "--max_my",
        type=int,
        default=0,
        help="Optional cap on number of your abstracts (0 = all).",
    )
    parser.add_argument(
        "--max_arxiv",
        type=int,
        default=0,
        help="Optional cap on number of arXiv abstracts (0 = all).",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default="recommendations/model_ab_test.md",
        help="Markdown output path for A/B report.",
    )
    args = parser.parse_args()

    cfg = _read_config(args.config)
    model_a = cfg.get("embedding_model", "allenai/specter2_aug2023refresh")
    use_adapters_a = bool(cfg.get("specter2_use_adapters", False))
    adapter_base_a = cfg.get("local_specter2_base_dir", "")
    adapter_dir_a = cfg.get("local_specter2_adapter_dir", "")

    processed_dir = cfg.get("processed_data_dir", "data/processed")
    arxiv_dir = cfg.get("arxiv_abstracts_dir", "data/arxiv_papers")
    my_path = os.path.join(processed_dir, "my_abstracts.pkl")
    arxiv_path = os.path.join(processed_dir, "arxiv_papers.pkl")

    if not os.path.exists(my_path):
        raise FileNotFoundError(
            f"Missing `{my_path}`. Run preprocess/pipeline first to create processed datasets."
        )
    if not os.path.exists(arxiv_path):
        raise FileNotFoundError(
            f"Missing `{arxiv_path}`. Run preprocess/pipeline first to create processed datasets."
        )

    papers_my, _ = load_preprocessed_papers(my_path)
    papers_arxiv, filenames_arxiv = load_preprocessed_papers(arxiv_path)

    if args.max_my > 0:
        papers_my = papers_my[: args.max_my]
    if args.max_arxiv > 0:
        papers_arxiv = papers_arxiv[: args.max_arxiv]
        filenames_arxiv = filenames_arxiv[: args.max_arxiv]

    print(f"Model A: {model_a} (adapters={use_adapters_a})")
    print(f"Model B: {args.model_b}")
    print(f"Corpus sizes -> my: {len(papers_my)}, arXiv: {len(papers_arxiv)}")

    print("Embedding/ranking with Model A...")
    top_a = _compute_model_rankings(
        model_name=model_a,
        papers_my=papers_my,
        papers_arxiv=papers_arxiv,
        filenames_arxiv=filenames_arxiv,
        top_n=args.top_n,
        batch_size=args.batch_size,
        use_adapters=use_adapters_a,
        adapter_base_dir=adapter_base_a,
        adapter_dir=adapter_dir_a,
    )

    print("Embedding/ranking with Model B...")
    top_b = _compute_model_rankings(
        model_name=args.model_b,
        papers_my=papers_my,
        papers_arxiv=papers_arxiv,
        filenames_arxiv=filenames_arxiv,
        top_n=args.top_n,
        batch_size=args.batch_size,
        use_adapters=False,
        adapter_base_dir="",
        adapter_dir="",
    )

    overlap = _summarize_overlap(top_a, top_b)

    lines = [
        "# Model A/B Test Report",
        "",
        "## Setup",
        "",
        f"- Model A: `{model_a}`",
        f"- Model A adapters: `{use_adapters_a}`",
        f"- Model A adapter base: `{adapter_base_a}`",
        f"- Model A adapter dir: `{adapter_dir_a}`",
        f"- Model B: `{args.model_b}`",
        f"- Top N compared: `{args.top_n}`",
        f"- My abstracts used: `{len(papers_my)}`",
        f"- arXiv abstracts used: `{len(papers_arxiv)}`",
        "",
        "## Agreement",
        "",
        f"- Overlap in top-{args.top_n}: `{overlap['overlap_count']}` papers",
        f"- Jaccard(top-{args.top_n}): `{overlap['jaccard']:.3f}`",
        "",
        "### Overlap files",
        "",
    ]
    if overlap["overlap_files"]:
        lines.extend([f"- `{fn}`" for fn in overlap["overlap_files"]])
    else:
        lines.append("- (none)")
    lines.extend(
        [
            "",
            _format_top_block(top_a, arxiv_dir, f"Top {args.top_n} for Model A"),
            _format_top_block(top_b, arxiv_dir, f"Top {args.top_n} for Model B"),
        ]
    )

    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"A/B report written to {args.output_md}")


if __name__ == "__main__":
    main()
