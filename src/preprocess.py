import os
import pickle
import argparse

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
    parser.add_argument('--input_dir', type=str, default='', help='Optional input directory to override dataset location.')
    parser.add_argument('--output_pickle', type=str, default='', help='Optional full path to save the output pickle.')
    args = parser.parse_args()

    # Determine input and output
    if args.input_dir:
        input_dir = args.input_dir
        output_file = args.output_pickle or os.path.join('data/processed', f'{args.dataset}.pkl')
        dataset_name = input_dir
    else:
        if args.dataset == 'my_abstracts':
            input_dir = 'data/abstracts'
            output_dir = 'data/processed'
            dataset_name = 'your abstracts'
        elif args.dataset == 'arxiv_papers':
            input_dir = 'data/arxiv_papers'
            output_dir = 'data/processed'
            dataset_name = 'arXiv abstracts'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{args.dataset}.pkl')

    print(f"Preprocessing {dataset_name}...")

    papers, filenames = load_papers(input_dir)

    # Save the preprocessed data
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump({'papers': papers, 'filenames': filenames}, f)
    print(f'Preprocessed {len(papers)} papers saved to {output_file}.')