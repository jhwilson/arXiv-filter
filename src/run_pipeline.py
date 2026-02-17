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


def list_txt_files(directory):
    if not os.path.isdir(directory):
        return []
    return [f for f in os.listdir(directory) if f.endswith('.txt')]


def get_selected_models(config):
    selected = config.get('enabled_recommendation_models', ['specter2_refresh'])
    if not selected:
        selected = ['specter2_refresh']

    model_specs = []
    for key in selected:
        if key == 'specter2_refresh':
            model_specs.append({
                'key': 'specter2_refresh',
                'model_name': config.get('embedding_model', 'allenai/specter2_aug2023refresh'),
                'local_model_dir': config.get('local_model_dir', ''),
                'use_adapters': bool(config.get('specter2_use_adapters', False)),
                'adapter_base_dir': config.get('local_specter2_base_dir', ''),
                'adapter_dir': config.get('local_specter2_adapter_dir', ''),
            })
        elif key == 'physbert':
            model_specs.append({
                'key': 'physbert',
                'model_name': config.get('physbert_model', 'thellert/accphysbert_cased'),
                'local_model_dir': config.get('local_physbert_model_dir', ''),
                'use_adapters': False,
                'adapter_base_dir': '',
                'adapter_dir': '',
            })
        else:
            print(f"Unknown model key in enabled_recommendation_models: {key} (skipping)")

    if not model_specs:
        model_specs.append({
            'key': 'specter2_refresh',
            'model_name': config.get('embedding_model', 'allenai/specter2_aug2023refresh'),
            'local_model_dir': config.get('local_model_dir', ''),
            'use_adapters': bool(config.get('specter2_use_adapters', False)),
            'adapter_base_dir': config.get('local_specter2_base_dir', ''),
            'adapter_dir': config.get('local_specter2_adapter_dir', ''),
        })
    return model_specs


def model_section_title(model_key: str) -> str:
    if model_key == 'specter2_refresh':
        return 'Specter2 recommendations'
    if model_key == 'physbert':
        return 'PhysBERT recommendations'
    return f'{model_key} recommendations'


def read_recommendation_body(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read()
    lines = txt.splitlines()
    if lines and lines[0].strip().startswith('## Top Recommendations'):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)
    body = '\n'.join(lines).strip()
    return (body + '\n') if body else ''


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
    recommendations_dir = config.get('recommendations_dir', 'recommendations')
    selected_models = get_selected_models(config)
    print("Selected models:", ", ".join(m['key'] for m in selected_models))

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
    if not os.path.isdir(my_abstracts_dir):
        os.makedirs(my_abstracts_dir, exist_ok=True)

    my_abstract_files = list_txt_files(my_abstracts_dir)
    if not my_abstract_files:
        raise FileNotFoundError(
            f"No abstract files found in `{my_abstracts_dir}`.\n"
            "Add one or more `.txt` abstracts (see README format), or run:\n"
            "  python src/fetch_abstracts.py --author_id <your_arxiv_author_id>"
        )

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

        # Force (re)embedding if missing embeddings or any selected model changed
        for model_spec in selected_models:
            key = model_spec['key']
            embeddings_my_path = os.path.join(embeddings_dir, f'my_abstracts_embeddings_{key}.pkl')
            last_model_file = os.path.join(embeddings_dir, f'last_model_{key}.txt')
            model_fingerprint = (
                f"{model_spec['model_name']}"
                f"|adapters={model_spec['use_adapters']}"
                f"|base={model_spec['adapter_base_dir']}"
                f"|adapter={model_spec['adapter_dir']}"
            )
            if not os.path.exists(embeddings_my_path):
                print(f"No embeddings found for model '{key}'. Generating...")
                my_papers_updated = True
                break
            if os.path.exists(last_model_file):
                with open(last_model_file, 'r') as f:
                    prev_fingerprint = f.read().strip()
                if prev_fingerprint != model_fingerprint:
                    print(f"Embedding model config changed for '{key}'. Re-embedding...")
                    my_papers_updated = True
                    break

        if my_papers_updated:
            # Save the new hash
            with open(hash_file, 'w') as f:
                f.write(current_hash)

            # Preprocess your abstracts
            print("Preprocessing your abstracts...")
            subprocess.run(['python', 'src/preprocess.py', '--dataset', 'my_abstracts'], check=True)

            # Generate embeddings for each selected model
            for model_spec in selected_models:
                key = model_spec['key']
                print(f"Generating embeddings for your abstracts ({key})...")
                out_path = os.path.join(embeddings_dir, f'my_abstracts_embeddings_{key}.pkl')
                cmd = [
                    'python', 'src/compute_embeddings.py',
                    '--dataset', 'my_abstracts',
                    '--model_name', model_spec['model_name'],
                    '--embeddings_output_path', out_path,
                ]
                if model_spec['local_model_dir']:
                    cmd += ['--local_model_dir', model_spec['local_model_dir']]
                if model_spec['use_adapters']:
                    cmd += ['--use_adapters']
                    if model_spec['adapter_base_dir']:
                        cmd += ['--adapter_base_dir', model_spec['adapter_base_dir']]
                    if model_spec['adapter_dir']:
                        cmd += ['--adapter_dir', model_spec['adapter_dir']]
                subprocess.run(cmd, check=True)

                model_fingerprint = (
                    f"{model_spec['model_name']}"
                    f"|adapters={model_spec['use_adapters']}"
                    f"|base={model_spec['adapter_base_dir']}"
                    f"|adapter={model_spec['adapter_dir']}"
                )
                last_model_file = os.path.join(embeddings_dir, f'last_model_{key}.txt')
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

        model_outputs = {}
        for model_spec in selected_models:
            key = model_spec['key']
            print(f"Generating embeddings for arXiv papers ({tag}, {key})...")
            emb_out = os.path.join(embeddings_dir, f'arxiv_abstracts_embeddings_{tag}_{key}.pkl')
            cmd = [
                'python', 'src/compute_embeddings.py',
                '--dataset', 'arxiv_papers',
                '--preprocessed_path', preproc_out,
                '--embeddings_output_path', emb_out,
                '--model_name', model_spec['model_name'],
            ]
            if model_spec['local_model_dir']:
                cmd += ['--local_model_dir', model_spec['local_model_dir']]
            if model_spec['use_adapters']:
                cmd += ['--use_adapters']
                if model_spec['adapter_base_dir']:
                    cmd += ['--adapter_base_dir', model_spec['adapter_base_dir']]
                if model_spec['adapter_dir']:
                    cmd += ['--adapter_dir', model_spec['adapter_dir']]
            subprocess.run(cmd, check=True)

            print(f"Computing similarities ({tag}, {key})...")
            sim_out = os.path.join(embeddings_dir, f'similarity_matrix_{tag}_{key}.pkl')
            emb_my = os.path.join(embeddings_dir, f'my_abstracts_embeddings_{key}.pkl')
            subprocess.run([
                'python', 'src/compute_similarity.py',
                '--embeddings_arxiv_path', emb_out,
                '--embeddings_my_path', emb_my,
                '--similarity_output_path', sim_out
            ], check=True)
            model_outputs[key] = {'emb': emb_out, 'sim': sim_out}

        return preproc_out, model_outputs

    # Run whitelist first, then rest
    wl_pre, wl_outputs = run_subset(whitelist_dir, 'whitelist') if copied_whitelist > 0 else (None, None)
    rest_pre, rest_outputs = run_subset(rest_dir, 'rest') if copied_rest > 0 else (None, None)

    # Compute similarities: handled per-subset above (whitelist/rest)

    # Generate recommendations: combine whitelist (if any) + rest
    print("Generating recommendations...")
    tmp_wl = None
    if wl_outputs:
        tmp_wl = os.path.join(recommendations_dir, f"{recommendations_base}_wl_tmp.md")
        if len(selected_models) == 1:
            model_key = selected_models[0]['key']
            subprocess.run([
                'python', 'src/recommend.py',
                '--top_n', str(config['top_n']),
                '--similarity_threshold', str(config.get('similarity_threshold', 0.0)),
                '--similarity_data_path', wl_outputs[model_key]['sim'],
                '--arxiv_abstracts_dir', whitelist_dir,
                '--output_file', tmp_wl
            ], check=True)
        else:
            sim_paths = [wl_outputs[m['key']]['sim'] for m in selected_models]
            subprocess.run([
                'python', 'src/recommend_ensemble.py',
                '--top_n', str(config['top_n']),
                '--similarity_threshold', str(config.get('similarity_threshold', 0.0)),
                '--arxiv_abstracts_dir', whitelist_dir,
                '--output_file', tmp_wl,
                '--similarity_data_paths',
                *sim_paths,
            ], check=True)

    tmp_rest = None
    tmp_rest_by_model = {}
    if rest_outputs:
        if len(selected_models) == 1:
            tmp_rest = os.path.join(recommendations_dir, f"{recommendations_base}_rest_tmp.md")
            model_key = selected_models[0]['key']
            subprocess.run([
                'python', 'src/recommend.py',
                '--top_n', str(config['top_n']),
                '--similarity_threshold', str(config.get('similarity_threshold', 0.0)),
                '--similarity_data_path', rest_outputs[model_key]['sim'],
                '--arxiv_abstracts_dir', rest_dir,
                '--output_file', tmp_rest
            ], check=True)
        else:
            for model_spec in selected_models:
                model_key = model_spec['key']
                model_tmp = os.path.join(
                    recommendations_dir,
                    f"{recommendations_base}_rest_{model_key}_tmp.md",
                )
                subprocess.run([
                    'python', 'src/recommend.py',
                    '--top_n', str(config['top_n']),
                    '--similarity_threshold', str(config.get('similarity_threshold', 0.0)),
                    '--similarity_data_path', rest_outputs[model_key]['sim'],
                    '--arxiv_abstracts_dir', rest_dir,
                    '--output_file', model_tmp
                ], check=True)
                tmp_rest_by_model[model_key] = model_tmp

    # Merge outputs with headers
    with open(recommendations_output, 'w', encoding='utf-8') as out:
        if tmp_wl and os.path.exists(tmp_wl):
            out.write("## Priority (whitelist authors)\n\n")
            out.write(read_recommendation_body(tmp_wl))
            out.write('\n')
        if tmp_rest and os.path.exists(tmp_rest):
            out.write("## Other recommendations\n\n")
            out.write(read_recommendation_body(tmp_rest))
            out.write('\n')
        elif tmp_rest_by_model:
            out.write("## Other recommendations\n\n")
            for model_spec in selected_models:
                model_key = model_spec['key']
                model_tmp = tmp_rest_by_model.get(model_key)
                if not model_tmp or not os.path.exists(model_tmp):
                    continue
                out.write(f"### {model_section_title(model_key)}\n\n")
                out.write(read_recommendation_body(model_tmp))
                out.write('\n')

    # Cleanup tmp files
    if tmp_wl and os.path.exists(tmp_wl):
        os.remove(tmp_wl)
    if tmp_rest and os.path.exists(tmp_rest):
        os.remove(tmp_rest)
    for model_tmp in tmp_rest_by_model.values():
        if os.path.exists(model_tmp):
            os.remove(model_tmp)

    print("Pipeline completed successfully.")
    print(f"Recommendations saved to {recommendations_output}")

if __name__ == '__main__':
    main()