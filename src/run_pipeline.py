import os
import shutil
import yaml
import argparse
import hashlib
import pickle
from datetime import datetime
from tqdm import tqdm

# Import your existing scripts as modules
import subprocess

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def compute_directory_hash(directory):
    hash_md5 = hashlib.md5()
    for root, dirs, files in os.walk(directory):
        for names in files:
            filepath = os.path.join(root, names)
            with open(filepath, 'rb') as f:
                hash_md5.update(f.read())
    return hash_md5.hexdigest()

def main():
    parser = argparse.ArgumentParser(description='Run the full pipeline.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file.')
    parser.add_argument('--check_my_papers', action='store_true', help='Check if my papers have updated.')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config if argument is provided
    if args.check_my_papers:
        config['check_my_papers'] = True

     # Get current date
    today_date = datetime.now().strftime('%Y-%m-%d')

    # Paths
    my_abstracts_dir = config.get('my_abstracts_dir', 'data/abstracts')
    arxiv_abstracts_dir = config.get('arxiv_abstracts_dir', 'data/arxiv_papers')
    processed_data_dir = config.get('processed_data_dir', 'data/processed')
    embeddings_dir = config.get('embeddings_dir', 'models')
    similarity_data_path = config.get('similarity_data_path', 'models/similarity_matrix.pkl')

     # Modify recommendations output to include the date
    recommendations_base = config.get('recommendations_output_base', 'recommendations')
    recommendations_output = f"{recommendations_base}_{today_date}.md"

    # Create necessary directories if they don't exist
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)

    # Check if my papers have updated
    my_papers_updated = False
    hash_file = os.path.join(processed_data_dir, 'my_abstracts_hash.txt')
    current_hash = compute_directory_hash(my_abstracts_dir)

    if config.get('check_my_papers', True):
        if os.path.exists(hash_file):
            with open(hash_file, 'r') as f:
                previous_hash = f.read()
            if current_hash != previous_hash:
                print("Your abstracts have changed. Reprocessing...")
                my_papers_updated = True
            else:
                print("Your abstracts have not changed. Skipping reprocessing.")
        else:
            print("No previous hash found. Processing your abstracts for the first time...")
            my_papers_updated = True

        if my_papers_updated:
            # Save the new hash
            with open(hash_file, 'w') as f:
                f.write(current_hash)

            # Preprocess your abstracts
            print("Preprocessing your abstracts...")
            subprocess.run(['python', 'src/preprocess.py', '--dataset', 'my_abstracts'], check=True)

            # Generate embeddings for your abstracts
            print("Generating embeddings for your abstracts...")
            subprocess.run(['python', 'src/compute_embeddings.py', '--dataset', 'my_abstracts', '--model_name', config['embedding_model']], check=True)
    else:
        print("Skipping check for updates to your abstracts.")

    # Clear arXiv data
    if os.path.exists(arxiv_abstracts_dir):
        print("Clearing old arXiv abstracts...")
        shutil.rmtree(arxiv_abstracts_dir)
    os.makedirs(arxiv_abstracts_dir, exist_ok=True)

    # Remove processed arXiv data and embeddings
    arxiv_processed_path = os.path.join(processed_data_dir, 'arxiv_abstracts.pkl')
    if os.path.exists(arxiv_processed_path):
        os.remove(arxiv_processed_path)

    arxiv_embeddings_path = os.path.join(embeddings_dir, 'arxiv_abstracts_embeddings.pkl')
    if os.path.exists(arxiv_embeddings_path):
        os.remove(arxiv_embeddings_path)

    # Fetch new arXiv papers
    print("Fetching new arXiv papers...")
    fetch_args = ['python', 'src/fetch_arxiv_papers.py',
                  '--categories'] + config['categories'] + [
                  '--days', str(config['days']),
                  '--output_dir', arxiv_abstracts_dir]
    subprocess.run(fetch_args, check=True)

    # Preprocess arXiv papers
    print("Preprocessing arXiv papers...")
    subprocess.run(['python', 'src/preprocess.py', '--dataset', 'arxiv_papers'], check=True)

    # Generate embeddings for arXiv papers
    print("Generating embeddings for arXiv papers...")
    subprocess.run(['python', 'src/compute_embeddings.py', '--dataset', 'arxiv_papers', '--model_name', config['embedding_model']], check=True)

    # Compute similarities
    print("Computing similarities...")
    subprocess.run(['python', 'src/compute_similarity.py'], check=True)

    # Generate recommendations
    print("Generating recommendations...")
    recommend_args = ['python', 'src/recommend.py',
                      '--top_n', str(config['top_n']),
                      '--similarity_threshold', str(config.get('similarity_threshold', 0.0)),
                      '--output_file', recommendations_output]
    subprocess.run(recommend_args, check=True)

    print("Pipeline completed successfully.")
    print(f"Recommendations saved to {recommendations_output}")

if __name__ == '__main__':
    main()