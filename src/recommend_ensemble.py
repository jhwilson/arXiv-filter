import os
import pickle
import argparse
import numpy as np


def load_similarity_data(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["similarity_matrix"], data["filenames_arxiv"], data["filenames_my"]


def extract_field(content, field_name):
    start = content.find(f"{field_name}")
    if start == -1:
        return ""
    start += len(field_name)
    end = content.find("\n", start)
    if end == -1:
        end = len(content)
    return content[start:end].strip()


def extract_multiline_field(content, start_field, end_field):
    start = content.find(f"{start_field}")
    if start == -1:
        return ""
    start += len(start_field)
    end = content.find(f"{end_field}", start)
    if end == -1:
        end = len(content)
    return content[start:end].strip()


def display_recommendations(top_papers, arxiv_abstracts_dir, output_file=None):
    output = []
    for filename, score in top_papers:
        filepath = os.path.join(arxiv_abstracts_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        title = extract_field(content, "Title:")
        authors = extract_field(content, "Authors:")
        abstract = extract_multiline_field(content, "Abstract:", "URL:")
        url = extract_field(content, "URL:")
        date = extract_field(content, "Date:")

        recommendation = f"### [{title}]({url})\n"
        recommendation += f"**Authors:** {authors}\n"
        recommendation += f"**Date:** {date}\n"
        recommendation += f"**Similarity Score:** {score:.4f}\n"
        recommendation += f"**Abstract:**\n{abstract}\n\n"
        recommendation += "---\n\n"
        output.append(recommendation)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("## Top Recommendations:\n\n")
            f.writelines(output)
        print(f"Recommendations saved to {output_file}")
    else:
        print("## Top Recommendations:\n")
        for rec in output:
            print(rec)


def rrf_fuse(similarity_matrices, threshold=0.0, k=60):
    # Per-model max similarities for each arXiv paper
    per_model_max = [np.max(m, axis=1) for m in similarity_matrices]
    n = per_model_max[0].shape[0]
    rrf_scores = np.zeros(n, dtype=float)

    for max_sim in per_model_max:
        order = np.argsort(max_sim)[::-1]
        # rank starts at 1
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(1, n + 1)
        rrf_scores += 1.0 / (k + ranks)

    # Keep papers that pass threshold on at least one selected model
    keep = np.zeros(n, dtype=bool)
    for max_sim in per_model_max:
        keep |= (max_sim >= threshold)

    indices = np.where(keep)[0]
    if len(indices) == 0:
        return []
    sorted_indices = indices[np.argsort(rrf_scores[indices])[::-1]]
    return sorted_indices, rrf_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ensemble recommendations from multiple similarity matrices.")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top recommendations to display.")
    parser.add_argument("--similarity_threshold", type=float, default=0.0, help="Minimum per-model similarity to consider.")
    parser.add_argument(
        "--similarity_data_paths",
        nargs="+",
        required=True,
        help="Space-separated similarity pickle paths (one per model).",
    )
    parser.add_argument("--arxiv_abstracts_dir", type=str, default="data/arxiv_papers", help="Directory containing arXiv abstracts.")
    parser.add_argument("--output_file", type=str, default=None, help="File to save recommendations (Markdown format).")
    args = parser.parse_args()

    matrices = []
    filenames_arxiv_ref = None
    filenames_my_ref = None

    for path in args.similarity_data_paths:
        sim, filenames_arxiv, filenames_my = load_similarity_data(path)
        if filenames_arxiv_ref is None:
            filenames_arxiv_ref = filenames_arxiv
            filenames_my_ref = filenames_my
        else:
            if filenames_arxiv != filenames_arxiv_ref:
                raise ValueError("Filenames mismatch across similarity matrices; cannot ensemble safely.")
            if filenames_my != filenames_my_ref:
                raise ValueError("My-paper filenames mismatch across similarity matrices; cannot ensemble safely.")
        matrices.append(sim)

    result = rrf_fuse(matrices, threshold=args.similarity_threshold, k=60)
    if not result:
        print("No papers found above the similarity threshold.")
    else:
        sorted_indices, rrf_scores = result
        top_indices = sorted_indices[: args.top_n]
        top_papers = [(filenames_arxiv_ref[i], float(rrf_scores[i])) for i in top_indices]
        display_recommendations(top_papers, args.arxiv_abstracts_dir, output_file=args.output_file)
