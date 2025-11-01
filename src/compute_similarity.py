import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse

def load_embeddings(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return np.array(data['embeddings']), data['filenames']

def compute_similarity(embeddings_arxiv, embeddings_my):
    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings_arxiv, embeddings_my)
    return similarity_matrix

def save_similarity(similarity_matrix, filenames_arxiv, filenames_my, output_filepath):
    # Save the similarity matrix and filenames as a pickle file
    with open(output_filepath, 'wb') as f:
        pickle.dump({
            'similarity_matrix': similarity_matrix,
            'filenames_arxiv': filenames_arxiv,
            'filenames_my': filenames_my
        }, f)
    print(f"Similarity matrix saved to {output_filepath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute similarity matrix.')
    parser.add_argument('--embeddings_arxiv_path', type=str, default='models/arxiv_abstracts_embeddings.pkl', help='Path to arXiv embeddings pkl')
    parser.add_argument('--embeddings_my_path', type=str, default='models/my_abstracts_embeddings.pkl', help='Path to my abstracts embeddings pkl')
    parser.add_argument('--similarity_output_path', type=str, default='models/similarity_matrix.pkl', help='Output path for similarity pkl')
    args = parser.parse_args()

    # Load embeddings
    print("Loading embeddings...")
    embeddings_arxiv, filenames_arxiv = load_embeddings(args.embeddings_arxiv_path)
    embeddings_my, filenames_my = load_embeddings(args.embeddings_my_path)

    if embeddings_arxiv.shape[1] != embeddings_my.shape[1]:
        print("Embedding dimensions do not match!")

    # Compute similarity matrix
    print("Computing similarity matrix...")
    similarity_matrix = compute_similarity(embeddings_arxiv, embeddings_my)

    # Save similarity matrix
    save_similarity(similarity_matrix, filenames_arxiv, filenames_my, args.similarity_output_path)

    print("Similarity computation completed.")