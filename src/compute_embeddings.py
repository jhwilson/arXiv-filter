import os
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse

def load_preprocessed_abstracts(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['abstracts'], data['filenames']

def generate_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    # Load the pre-trained model
    model = SentenceTransformer(model_name)
    # Generate embeddings for the texts
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=False)
    return embeddings

def save_embeddings(embeddings, filenames, output_filepath):
    # Save embeddings and filenames as a pickle file
    with open(output_filepath, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'filenames': filenames}, f)
    print(f"Embeddings saved to {output_filepath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute embeddings.')
    parser.add_argument('--dataset', type=str, choices=['my_abstracts', 'arxiv_papers'], default='my_abstracts', help='Dataset to compute embeddings for.')
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2', help='Embedding model name.')
    args = parser.parse_args()

    # Set paths based on dataset
    if args.dataset == 'my_abstracts':
        preprocessed_path = 'data/processed/my_abstracts.pkl'
        embeddings_output_path = 'models/my_abstracts_embeddings.pkl'
        print("Generating embeddings for your abstracts...")
    elif args.dataset == 'arxiv_papers':
        preprocessed_path = 'data/processed/arxiv_papers.pkl'
        embeddings_output_path = 'models/arxiv_abstracts_embeddings.pkl'
        print("Generating embeddings for arXiv papers...")

    # Load preprocessed abstracts
    abstracts, filenames = load_preprocessed_abstracts(preprocessed_path)

    # Generate embeddings
    embeddings = generate_embeddings(abstracts, model_name=args.model_name)

    # Save embeddings
    save_embeddings(embeddings, filenames, embeddings_output_path)

    print("Embeddings have been generated and saved.")
# if __name__ == '__main__':
#     # Paths to preprocessed data
#     preprocessed_my_abstracts = 'data/processed/my_abstracts.pkl'
#     preprocessed_arxiv_abstracts = 'data/processed/arxiv_abstracts.pkl'

#     # Output directories for embeddings
#     embeddings_dir = 'models'
#     os.makedirs(embeddings_dir, exist_ok=True)

#     # Generate embeddings for your abstracts
#     print("Generating embeddings for your abstracts...")
#     abstracts_my, filenames_my = load_preprocessed_abstracts(preprocessed_my_abstracts)
#     embeddings_my = generate_embeddings(abstracts_my)
#     save_embeddings(embeddings_my, filenames_my, os.path.join(embeddings_dir, 'my_abstracts_embeddings.pkl'))

#     # Generate embeddings for arXiv papers
#     print("Generating embeddings for arXiv papers...")
#     abstracts_arxiv, filenames_arxiv = load_preprocessed_abstracts(preprocessed_arxiv_abstracts)
#     embeddings_arxiv = generate_embeddings(abstracts_arxiv)
#     save_embeddings(embeddings_arxiv, filenames_arxiv, os.path.join(embeddings_dir, 'arxiv_abstracts_embeddings.pkl'))

#     print("All embeddings have been generated and saved.")