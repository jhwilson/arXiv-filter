import os
import pickle
import argparse
from tqdm import tqdm

def load_papers(input_dir):
    papers = []
    filenames = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            title = extract_field(content, 'Title:')
            abstract = extract_multiline_field(content, 'Abstract:', 'Date:')
            # Combine title and abstract with [SEP]
            paper_text = f"{title} [SEP] {abstract}"
            papers.append(paper_text)
            filenames.append(filename)
    return papers, filenames

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
    parser = argparse.ArgumentParser(description='Preprocess abstracts.')
    parser.add_argument('--dataset', type=str, choices=['my_abstracts', 'arxiv_papers'], default='my_abstracts', help='Dataset to preprocess.')
    args = parser.parse_args()

    # Use args.dataset to set input and output directories
    if args.dataset == 'my_abstracts':
        input_dir = 'data/abstracts'
        output_dir = 'data/processed'
        dataset_name = 'your abstracts'
    elif args.dataset == 'arxiv_papers':
        input_dir = 'data/arxiv_papers'
        output_dir = 'data/processed'
        dataset_name = 'arXiv abstracts'

    print(f"Preprocessing {dataset_name}...")

    papers, filenames = load_papers(input_dir)

    # Save the preprocessed data
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{args.dataset}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump({'papers': papers, 'filenames': filenames}, f)
    print(f'Preprocessed {len(papers)} papers saved to {output_file}.')