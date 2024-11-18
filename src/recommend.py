import os
import pickle
import numpy as np
import argparse

def load_similarity_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['similarity_matrix'], data['filenames_arxiv'], data['filenames_my']

def get_top_recommendations(similarity_matrix, filenames_arxiv, top_n=10, threshold=0.0):
    # Compute the maximum similarity score for each arXiv paper across your papers
    max_similarities = np.max(similarity_matrix, axis=1)
    # Filter papers above the threshold
    indices = np.where(max_similarities >= threshold)[0]
    # Sort indices based on similarity scores
    sorted_indices = indices[np.argsort(max_similarities[indices])[::-1]]
    # Get top N papers
    top_indices = sorted_indices[:top_n]
    # Create a list of tuples (filename, similarity score)
    top_papers = [(filenames_arxiv[i], max_similarities[i]) for i in top_indices]
    return top_papers

def display_recommendations(top_papers, arxiv_abstracts_dir, output_file=None):
    output = []
    for filename, score in top_papers:
        filepath = os.path.join(arxiv_abstracts_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        # Extract relevant information from the content
        title = extract_field(content, 'Title:')
        authors = extract_field(content, 'Authors:')
        abstract = extract_multiline_field(content, 'Abstract:', 'URL:')
        url = extract_field(content, 'URL:')
        date = extract_field(content, 'Date:')

        # Format output in Markdown
        recommendation = f"### [{title}]({url})\n"
        recommendation += f"**Authors:** {authors}\n"
        recommendation += f"**Date:** {date}\n"
        recommendation += f"**Similarity Score:** {score:.4f}\n"
        recommendation += f"**Abstract:**\n{abstract}\n\n"
        recommendation += "---\n\n"
        output.append(recommendation)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("## Top Recommendations:\n\n")
            f.writelines(output)
        print(f"Recommendations saved to {output_file}")
    else:
        print("## Top Recommendations:\n")
        for rec in output:
            print(rec)

def extract_field(content, field_name):
    start = content.find(f"{field_name}")
    if start == -1:
        return ''
    start += len(field_name)
    end = content.find('\n', start)
    if end == -1:
        end = len(content)
    return content[start:end].strip()

def extract_multiline_field(content, start_field, end_field):
    start = content.find(f"{start_field}")
    if start == -1:
        return ''
    start += len(start_field)
    end = content.find(f"{end_field}", start)
    if end == -1:
        end = len(content)
    return content[start:end].strip()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate recommendations based on similarity scores.')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top recommendations to display.')
    parser.add_argument('--similarity_threshold', type=float, default=0.0, help='Minimum similarity score to consider.')
    parser.add_argument('--similarity_data_path', type=str, default='models/similarity_matrix.pkl', help='Path to similarity data pickle file.')
    parser.add_argument('--arxiv_abstracts_dir', type=str, default='data/arxiv_papers', help='Directory containing arXiv abstracts.')
    parser.add_argument('--output_file', type=str, default=None, help='File to save recommendations (Markdown format).')
    args = parser.parse_args()

    # Load similarity data
    print("Loading similarity data...")
    similarity_matrix, filenames_arxiv, filenames_my = load_similarity_data(args.similarity_data_path)

    # Get top N recommendations with threshold
    top_n = args.top_n
    threshold = args.similarity_threshold
    print(f"Retrieving top {top_n} recommendations with similarity threshold {threshold}...")
    top_papers = get_top_recommendations(similarity_matrix, filenames_arxiv, top_n=top_n, threshold=threshold)

    if not top_papers:
        print("No papers found above the similarity threshold.")
    else:
        # Display recommendations
        display_recommendations(top_papers, args.arxiv_abstracts_dir, output_file=args.output_file)