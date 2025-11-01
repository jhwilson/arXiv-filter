import os
import shutil
import yaml
import argparse
import hashlib
from datetime import datetime

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
    local_model_dir = config.get('local_model_dir', '')
    use_adapters = config.get('specter2_use_adapters', False)
    adapter_base_dir = config.get('local_specter2_base_dir', '')
    adapter_dir = config.get('local_specter2_adapter_dir', '')
    recommendations_dir = config.get('recommendations_dir', 'recommendations')
    embeddings_my_path = os.path.join(embeddings_dir, 'my_abstracts_embeddings.pkl')
    last_model_file = os.path.join(embeddings_dir, 'last_model.txt')
    model_fingerprint = f"{config['embedding_model']}|adapters={use_adapters}|base={adapter_base_dir}|adapter={adapter_dir}"

    # Modify recommendations output to include the date and directory
    recommendations_base = config.get('recommendations_output_base', 'recommendations')
    os.makedirs(recommendations_dir, exist_ok=True)
    recommendations_output = os.path.join(recommendations_dir, f"{recommendations_base}_{today_date}.md")

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

        # Force (re)embedding if missing embeddings or model changed
        if not os.path.exists(embeddings_my_path):
            print("No existing embeddings found for your abstracts. Generating...")
            my_papers_updated = True
        elif os.path.exists(last_model_file):
            with open(last_model_file, 'r') as f:
                prev_fingerprint = f.read().strip()
            if prev_fingerprint != model_fingerprint:
                print("Embedding model configuration changed. Re-embedding your abstracts...")
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
            cmd = ['python', 'src/compute_embeddings.py', '--dataset', 'my_abstracts', '--model_name', config['embedding_model']]
            if local_model_dir:
                cmd += ['--local_model_dir', local_model_dir]
            if use_adapters:
                cmd += ['--use_adapters']
                if adapter_base_dir:
                    cmd += ['--adapter_base_dir', adapter_base_dir]
                if adapter_dir:
                    cmd += ['--adapter_dir', adapter_dir]
            subprocess.run(cmd, check=True)

            # Record the model fingerprint to detect future changes
            with open(last_model_file, 'w') as f:
                f.write(model_fingerprint)
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

    # --- Author filters (blacklist/whitelist) ---
    def normalize_name(name: str) -> str:
        s = name.lower()
        for ch in [',', '.', '"', "'", '-', '_', '(', ')']:
            s = s.replace(ch, ' ')
        s = ' '.join(s.split())
        return s

    def parse_first_last(name: str):
        parts = normalize_name(name).split()
        if not parts:
            return ('', '')
        if len(parts) == 1:
            return (parts[0], '')
        return (parts[0], parts[-1])

    def author_matches(target: str, candidate: str) -> bool:
        tf, tl = parse_first_last(target)
        cf, cl = parse_first_last(candidate)
        if not tl or not cl:
            return False
        if tl != cl:
            return False
        # First name full match or initial match
        if tf == cf:
            return True
        if tf and cf and tf[0] == cf[0]:
            # Allow first-initial vs full-first-name
            if len(tf) == 1 or len(cf) == 1:
                return True
        return False

    def load_author_list(path: str):
        if not os.path.exists(path):
            return []
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def authors_line_matches_list(authors_line: str, names: list) -> bool:
        # authors_line like: "Authors: A, B, C"
        raw = authors_line.replace('Authors:', '').strip()
        candidates = [a.strip() for a in raw.split(',') if a.strip()]
        for name in names:
            for cand in candidates:
                if author_matches(name, cand):
                    return True
        return False

    blacklist_path = os.path.join('config', 'blacklist_authors.txt')
    whitelist_path = os.path.join('config', 'whitelist_authors.txt')
    os.makedirs('config', exist_ok=True)
    # Ensure files exist (blacklist blank as requested)
    if not os.path.exists(blacklist_path):
        with open(blacklist_path, 'w', encoding='utf-8') as f:
            f.write('')
    if not os.path.exists(whitelist_path):
        with open(whitelist_path, 'w', encoding='utf-8') as f:
            f.write('')

    blacklist = load_author_list(blacklist_path)
    whitelist = load_author_list(whitelist_path)

    # Paths for subsets
    whitelist_dir = os.path.join(arxiv_abstracts_dir, '..', 'arxiv_papers_whitelist')
    whitelist_dir = os.path.normpath(whitelist_dir)
    rest_dir = os.path.join(arxiv_abstracts_dir, '..', 'arxiv_papers_rest')
    rest_dir = os.path.normpath(rest_dir)
    # Reset subsets
    if os.path.exists(whitelist_dir):
        shutil.rmtree(whitelist_dir)
    if os.path.exists(rest_dir):
        shutil.rmtree(rest_dir)
    os.makedirs(whitelist_dir, exist_ok=True)
    os.makedirs(rest_dir, exist_ok=True)

    # Filter fetched files
    removed_blacklisted = 0
    copied_whitelist = 0
    copied_rest = 0
    for filename in os.listdir(arxiv_abstracts_dir):
        if not filename.endswith('.txt'):
            continue
        src = os.path.join(arxiv_abstracts_dir, filename)
        with open(src, 'r', encoding='utf-8') as f:
            content = f.read()
        # Extract authors line
        start = content.find('Authors:')
        authors_line = content[start:content.find('\n', start)] if start != -1 else 'Authors:'
        # Blacklist filter
        if blacklist and authors_line_matches_list(authors_line, blacklist):
            removed_blacklisted += 1
            continue
        # Whitelist routing
        if whitelist and authors_line_matches_list(authors_line, whitelist):
            shutil.copy2(src, os.path.join(whitelist_dir, filename))
            copied_whitelist += 1
        else:
            shutil.copy2(src, os.path.join(rest_dir, filename))
            copied_rest += 1

    print(f"Blacklisted removed: {removed_blacklisted}; Whitelist files: {copied_whitelist}; Rest files: {copied_rest}")

    # Helper to run a subset pipeline
    def run_subset(arxiv_dir: str, tag: str):
        print(f"Preprocessing arXiv papers ({tag})...")
        preproc_out = os.path.join(processed_data_dir, f'arxiv_papers_{tag}.pkl')
        subprocess.run(['python', 'src/preprocess.py', '--dataset', 'arxiv_papers', '--input_dir', arxiv_dir, '--output_pickle', preproc_out], check=True)

        print(f"Generating embeddings for arXiv papers ({tag})...")
        emb_out = os.path.join(embeddings_dir, f'arxiv_abstracts_embeddings_{tag}.pkl')
        cmd = ['python', 'src/compute_embeddings.py', '--dataset', 'arxiv_papers', '--preprocessed_path', preproc_out, '--embeddings_output_path', emb_out, '--model_name', config['embedding_model']]
        if local_model_dir:
            cmd += ['--local_model_dir', local_model_dir]
        if use_adapters:
            cmd += ['--use_adapters']
            if adapter_base_dir:
                cmd += ['--adapter_base_dir', adapter_base_dir]
            if adapter_dir:
                cmd += ['--adapter_dir', adapter_dir]
        subprocess.run(cmd, check=True)

        print(f"Computing similarities ({tag})...")
        sim_out = os.path.join(embeddings_dir, f'similarity_matrix_{tag}.pkl')
        subprocess.run(['python', 'src/compute_similarity.py', '--embeddings_arxiv_path', emb_out, '--embeddings_my_path', os.path.join(embeddings_dir, 'my_abstracts_embeddings.pkl'), '--similarity_output_path', sim_out], check=True)

        return preproc_out, emb_out, sim_out

    # Run whitelist first, then rest
    wl_pre, wl_emb, wl_sim = run_subset(whitelist_dir, 'whitelist') if copied_whitelist > 0 else (None, None, None)
    rest_pre, rest_emb, rest_sim = run_subset(rest_dir, 'rest') if copied_rest > 0 else (None, None, None)

    # Compute similarities: handled per-subset above (whitelist/rest)

    # Generate recommendations: combine whitelist (if any) + rest
    print("Generating recommendations...")
    tmp_wl = None
    if wl_sim:
        tmp_wl = os.path.join(recommendations_dir, f"{recommendations_base}_wl_tmp.md")
        subprocess.run(['python', 'src/recommend.py', '--top_n', str(config['top_n']), '--similarity_threshold', str(config.get('similarity_threshold', 0.0)), '--similarity_data_path', wl_sim, '--arxiv_abstracts_dir', whitelist_dir, '--output_file', tmp_wl], check=True)

    tmp_rest = None
    if rest_sim:
        tmp_rest = os.path.join(recommendations_dir, f"{recommendations_base}_rest_tmp.md")
        subprocess.run(['python', 'src/recommend.py', '--top_n', str(config['top_n']), '--similarity_threshold', str(config.get('similarity_threshold', 0.0)), '--similarity_data_path', rest_sim, '--arxiv_abstracts_dir', rest_dir, '--output_file', tmp_rest], check=True)

    # Merge outputs with headers
    with open(recommendations_output, 'w', encoding='utf-8') as out:
        if tmp_wl and os.path.exists(tmp_wl):
            out.write("## Priority (whitelist authors)\n\n")
            with open(tmp_wl, 'r', encoding='utf-8') as f:
                out.write(f.read())
            out.write('\n')
        if tmp_rest and os.path.exists(tmp_rest):
            out.write("## Other recommendations\n\n")
            with open(tmp_rest, 'r', encoding='utf-8') as f:
                out.write(f.read())
            out.write('\n')

    # Cleanup tmp files
    if tmp_wl and os.path.exists(tmp_wl):
        os.remove(tmp_wl)
    if tmp_rest and os.path.exists(tmp_rest):
        os.remove(tmp_rest)

    print("Pipeline completed successfully.")
    print(f"Recommendations saved to {recommendations_output}")

if __name__ == '__main__':
    main()