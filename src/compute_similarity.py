import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    # Paths to embeddings
    embeddings_arxiv_path = 'models/arxiv_abstracts_embeddings.pkl'
    embeddings_my_path = 'models/my_abstracts_embeddings.pkl'

    # Output path for similarity matrix
    similarity_output_path = 'models/similarity_matrix.pkl'

    # Load embeddings
    print("Loading embeddings...")
    embeddings_arxiv, filenames_arxiv = load_embeddings(embeddings_arxiv_path)
    embeddings_my, filenames_my = load_embeddings(embeddings_my_path)

    if embeddings_arxiv.shape[1] != embeddings_my.shape[1]:
        print("Embedding dimensions do not match!")

    # Compute similarity matrix
    print("Computing similarity matrix...")
    similarity_matrix = compute_similarity(embeddings_arxiv, embeddings_my)

    # Save similarity matrix
    save_similarity(similarity_matrix, filenames_arxiv, filenames_my, similarity_output_path)

    print("Similarity computation completed.")