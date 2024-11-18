import os
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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
    # Paths to preprocessed data
    preprocessed_my_abstracts = 'data/processed/my_abstracts.pkl'
    preprocessed_arxiv_abstracts = 'data/processed/arxiv_abstracts.pkl'

    # Output directories for embeddings
    embeddings_dir = 'models'
    os.makedirs(embeddings_dir, exist_ok=True)

    # Generate embeddings for your abstracts
    print("Generating embeddings for your abstracts...")
    abstracts_my, filenames_my = load_preprocessed_abstracts(preprocessed_my_abstracts)
    embeddings_my = generate_embeddings(abstracts_my)
    save_embeddings(embeddings_my, filenames_my, os.path.join(embeddings_dir, 'my_abstracts_embeddings.pkl'))

    # Generate embeddings for arXiv papers
    print("Generating embeddings for arXiv papers...")
    abstracts_arxiv, filenames_arxiv = load_preprocessed_abstracts(preprocessed_arxiv_abstracts)
    embeddings_arxiv = generate_embeddings(abstracts_arxiv)
    save_embeddings(embeddings_arxiv, filenames_arxiv, os.path.join(embeddings_dir, 'arxiv_abstracts_embeddings.pkl'))

    print("All embeddings have been generated and saved.")