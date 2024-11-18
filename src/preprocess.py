import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import pickle
import argparse

def download_nltk_resource(resource_name):
    try:
        nltk.data.find(resource_name)
    except LookupError:
        # Extract the word after the last slash
        resource_to_download = resource_name.split('/')[-1]
        nltk.download(resource_name)

# Download necessary NLTK resources
download_nltk_resource('tokenizers/punkt')
download_nltk_resource('corpora/stopwords')
download_nltk_resource('corpora/wordnet')
download_nltk_resource('tokenizers/punkt_tab')

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove non-alphanumeric characters and lowercase the text
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    return text

def tokenize_and_remove_stopwords(text):
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized

def preprocess_text(text):
    text = clean_text(text)
    tokens = tokenize_and_remove_stopwords(text)
    lemmatized_tokens = lemmatize_tokens(tokens)
    return ' '.join(lemmatized_tokens)

# def preprocess_abstracts(input_dir, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     abstracts = []
#     filenames = []

#     for filename in tqdm(os.listdir(input_dir)):
#         if filename.endswith('.txt'):
#             filepath = os.path.join(input_dir, filename)
#             with open(filepath, 'r', encoding='utf-8') as f:
#                 content = f.read()
#                 # Extract the abstract
#                 abstract = extract_abstract(content)
#                 # Preprocess the abstract
#                 preprocessed = preprocess_text(abstract)
#                 # Save preprocessed abstract
#                 output_filepath = os.path.join(output_dir, filename)
#                 with open(output_filepath, 'w', encoding='utf-8') as out_f:
#                     out_f.write(preprocessed)
#                 abstracts.append(preprocessed)
#                 filenames.append(filename)
#     return abstracts, filenames

def extract_abstract(content):
    # Assuming the abstract starts after 'Abstract:\n' and ends before 'URL:'
    start = content.find('Abstract:\n') + len('Abstract:\n')
    end = content.find('URL:')
    abstract = content[start:end].strip()
    return abstract

def preprocess_dataset(input_dir, output_dir, dataset_name):
    os.makedirs(output_dir, exist_ok=True)
    abstracts = []
    filenames = []

    print(f"Preprocessing {dataset_name}...")
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith('.txt'):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract the abstract
                abstract = extract_abstract(content)
                # Preprocess the abstract
                preprocessed = preprocess_text(abstract)
                # Save preprocessed abstract
                output_filepath = os.path.join(output_dir, filename)
                with open(output_filepath, 'w', encoding='utf-8') as out_f:
                    out_f.write(preprocessed)
                abstracts.append(preprocessed)
                filenames.append(filename)
    return abstracts, filenames


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

    # Preprocess the dataset
    abstracts, filenames = preprocess_dataset(input_dir, output_dir, dataset_name)

    # Construct the output file path correctly
    output_file = os.path.join(output_dir, f'{args.dataset}.pkl')

    # Save the preprocessed data
    with open(output_file, 'wb') as f:
        pickle.dump({'abstracts': abstracts, 'filenames': filenames}, f)
    print(f'Preprocessed abstracts saved to {output_file}.')
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Preprocess abstracts.')
#     parser.add_argument('--dataset', type=str, choices=['my_abstracts', 'arxiv_papers'], default='my_abstracts', help='Dataset to preprocess.')
#     args = parser.parse_args()

#     # Use args.dataset to set input and output directories
#     if args.dataset == 'my_abstracts':
#         input_dir = 'data/abstracts'
#         output_dir = 'data/processed'
#         dataset_name = 'your abstracts'
#     elif args.dataset == 'arxiv_papers':
#         input_dir = 'data/arxiv_papers'
#         output_dir = 'data/processed'
#         dataset_name = 'arXiv abstracts'

#     abstracts, filenames = preprocess_dataset(input_dir, output_dir, dataset_name)
#     output_file = os.path.join(output_dir, args.dataset, '.pkl')

#     with open(output_file, 'wb') as f:
#         pickle.dump({'abstracts': abstracts, 'filenames': filenames}, f)
#     print('Preprocessed abstracts.')

    # if args.dataset in ['arxiv_papers', 'both']:
    #     # Paths for arXiv papers
    #     input_dir_arxiv = 'data/arxiv_papers'
    #     output_dir_arxiv = 'data/processed/arxiv_papers'

    #     # Preprocess arXiv papers
    #     abstracts_arxiv, filenames_arxiv = preprocess_dataset(input_dir_arxiv, output_dir_arxiv, 'arXiv papers')
    #     with open('data/processed/arxiv_abstracts.pkl', 'wb') as f:
    #         pickle.dump({'abstracts': abstracts_arxiv, 'filenames': filenames_arxiv}, f)
    #     print('Preprocessed arXiv papers.')