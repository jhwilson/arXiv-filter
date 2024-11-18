import os
import pickle
import argparse
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_preprocessed_papers(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['papers'], data['filenames']

def generate_embeddings(texts, model_name='allenai-specter'):
    # Load the pre-trained SPECTER model
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
    parser.add_argument('--model_name', type=str, default='allenai-specter', help='Embedding model name.')
    args = parser.parse_args()

    # Set paths based on dataset
    if args.dataset == 'my_abstracts':
        preprocessed_path = 'data/processed/my_abstracts.pkl'
        embeddings_output_path = 'models/my_abstracts_embeddings.pkl'
        print("Generating embeddings for your papers...")
    elif args.dataset == 'arxiv_papers':
        preprocessed_path = 'data/processed/arxiv_papers.pkl'
        embeddings_output_path = 'models/arxiv_abstracts_embeddings.pkl'
        print("Generating embeddings for arXiv papers...")

    # Load preprocessed papers
    papers, filenames = load_preprocessed_papers(preprocessed_path)

    # Generate embeddings
    embeddings = generate_embeddings(papers, model_name=args.model_name)

    # Save embeddings
    save_embeddings(embeddings, filenames, embeddings_output_path)

    print("Embeddings have been generated and saved.")