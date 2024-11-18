import os
import pickle
import numpy as np

def load_similarity_data(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['similarity_matrix'], data['filenames_arxiv'], data['filenames_my']

def get_top_recommendations(similarity_matrix, filenames_arxiv, top_n=10):
    # Compute the maximum similarity score for each arXiv paper across your papers
    max_similarities = np.max(similarity_matrix, axis=1)
    # Get indices of top N papers
    top_indices = np.argsort(max_similarities)[::-1][:top_n]
    # Create a list of tuples (filename, similarity score)
    top_papers = [(filenames_arxiv[i], max_similarities[i]) for i in top_indices]
    return top_papers

def display_recommendations(top_papers, arxiv_abstracts_dir):
    for filename, score in top_papers:
        filepath = os.path.join(arxiv_abstracts_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        # Extract relevant information from the content
        title = extract_field(content, 'Title:')
        authors = extract_field(content, 'Authors:')
        abstract = extract_field(content, 'Abstract:')
        url = extract_field(content, 'URL:')
        date = extract_field(content, 'Date:')

        print(f"Filename: {filename}")
        print(f"Similarity Score: {score:.4f}")
        print(f"Title: {title}")
        print(f"Authors: {authors}")
        print(f"Date: {date}")
        print(f"URL: {url}")
        print(f"Abstract:\n{abstract}\n")
        print("="*80)

def extract_field(content, field_name):
    start = content.find(f"{field_name}")
    if start == -1:
        return ''
    start += len(field_name)
    end = content.find('\n', start)
    if end == -1:
        end = len(content)
    return content[start:end].strip()

if __name__ == '__main__':
    # Path to similarity data
    similarity_data_path = 'models/similarity_matrix.pkl'

    # Directory containing arXiv abstracts
    arxiv_abstracts_dir = 'data/arxiv_papers'

    # Load similarity data
    print("Loading similarity data...")
    similarity_matrix, filenames_arxiv, filenames_my = load_similarity_data(similarity_data_path)

    # Get top N recommendations
    top_n = 10  # You can adjust this number
    print(f"Retrieving top {top_n} recommendations...")
    top_papers = get_top_recommendations(similarity_matrix, filenames_arxiv, top_n=top_n)

    # Display recommendations
    print("Top recommendations:")
    display_recommendations(top_papers, arxiv_abstracts_dir)