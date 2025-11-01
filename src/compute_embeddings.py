import os
import pickle
import argparse
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

def load_preprocessed_papers(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['papers'], data['filenames']

def _mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

def generate_embeddings(texts, model_name='allenai/specter2', backend='auto', batch_size=16,
                        use_adapters=False, adapter_base_dir='', adapter_dir=''):
    # Increase HF timeouts and enable faster transfer if available
    os.environ.setdefault('HF_HUB_READ_TIMEOUT', '60')
    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')

    # If adapter-based flow requested and dirs provided, use adapters library
    if use_adapters and adapter_base_dir and adapter_dir:
        try:
            from adapters import AutoAdapterModel
            # Prefer GPU/MPS if available; we will retry on CPU if needed
            preferred_device = (
                'cuda' if torch.cuda.is_available() else (
                    'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'
                )
            )

            def encode_with_adapters(device: str):
                tokenizer = AutoTokenizer.from_pretrained(adapter_base_dir)
                model = AutoAdapterModel.from_pretrained(adapter_base_dir)
                model.load_adapter(adapter_dir, load_as="proximity")
                model.set_active_adapters("proximity")
                model.to(device).eval()

                # Replace literal [SEP] with tokenizer.sep_token to match SPECTER2 guidance
                processed_texts = [
                    t.replace(" [SEP] ", f" {tokenizer.sep_token} ").replace("[SEP]", tokenizer.sep_token)
                    for t in texts
                ]

                chunks = []
                with torch.no_grad():
                    for i in range(0, len(processed_texts), batch_size):
                        batch = processed_texts[i:i + batch_size]
                        enc = tokenizer(
                            batch,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors='pt',
                            return_token_type_ids=False
                        )
                        enc = {k: v.to(device) for k, v in enc.items()}
                        out = model(**enc)
                        cls = out.last_hidden_state[:, 0, :]
                        cls = torch.nn.functional.normalize(cls, p=2, dim=1)
                        chunks.append(cls.cpu())
                return torch.cat(chunks, dim=0).numpy()

            try:
                return encode_with_adapters(preferred_device)
            except RuntimeError as rt_err:
                # Common on Apple MPS when some tensors remain on CPU; retry on CPU
                if 'expected on mps' in str(rt_err).lower() or 'device type mps' in str(rt_err).lower():
                    return encode_with_adapters('cpu')
                raise
        except Exception as e:
            print(f"Adapters path failed, falling back: {e}")

    # Try SentenceTransformers first
    if backend in ('auto', 'st'):
        try:
            model = SentenceTransformer(model_name)
            return model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_tensor=False,
                normalize_embeddings=True,
            )
        except Exception as e:
            if backend == 'st':
                raise
            print(f"Falling back to Transformers loader for {model_name}: {e}")

    # Transformers fallback with mean pooling and L2 normalization
    device = (
        'cuda' if torch.cuda.is_available() else (
            'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    embeddings_chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            pooled = _mean_pooling(out.last_hidden_state, enc['attention_mask'])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings_chunks.append(pooled.cpu())

    return torch.cat(embeddings_chunks, dim=0).numpy()

def save_embeddings(embeddings, filenames, output_filepath):
    # Save embeddings and filenames as a pickle file
    with open(output_filepath, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'filenames': filenames}, f)
    print(f"Embeddings saved to {output_filepath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute embeddings.')
    parser.add_argument('--dataset', type=str, choices=['my_abstracts', 'arxiv_papers'], default='my_abstracts', help='Dataset to compute embeddings for.')
    parser.add_argument('--preprocessed_path', type=str, default='', help='Optional path to preprocessed .pkl to override dataset mapping.')
    parser.add_argument('--embeddings_output_path', type=str, default='', help='Optional path to write embeddings .pkl.')
    parser.add_argument('--model_name', type=str, default='allenai/specter2', help='Embedding model name or local dir path.')
    parser.add_argument('--local_model_dir', type=str, default='', help='Optional local directory of a pre-downloaded HF model (avoids network).')
    parser.add_argument('--use_adapters', action='store_true', help='Use SPECTER2 base + proximity adapter via adapters library.')
    parser.add_argument('--adapter_base_dir', type=str, default='', help='Local dir for SPECTER2 base (e.g., models/specter2_base).')
    parser.add_argument('--adapter_dir', type=str, default='', help='Local dir for proximity adapter (e.g., models/specter2_adapter).')
    args = parser.parse_args()

    # Set paths based on dataset or overrides
    if args.preprocessed_path:
        preprocessed_path = args.preprocessed_path
        embeddings_output_path = args.embeddings_output_path or (
            'models/my_abstracts_embeddings.pkl' if args.dataset == 'my_abstracts' else 'models/arxiv_abstracts_embeddings.pkl'
        )
        print("Generating embeddings from custom preprocessed dataset...")
    else:
        if args.dataset == 'my_abstracts':
            preprocessed_path = 'data/processed/my_abstracts.pkl'
            embeddings_output_path = 'models/my_abstracts_embeddings.pkl'
            print("Generating embeddings for your papers...")
        elif args.dataset == 'arxiv_papers':
            preprocessed_path = 'data/processed/arxiv_papers.pkl'
            embeddings_output_path = 'models/arxiv_abstracts_embeddings.pkl'
            print("Generating embeddings for arXiv papers...")

    # Resolve model id or local directory
    model_id = args.model_name
    if args.local_model_dir and os.path.isdir(args.local_model_dir):
        model_id = args.local_model_dir

    # Load preprocessed papers
    papers, filenames = load_preprocessed_papers(preprocessed_path)

    # Generate embeddings
    embeddings = generate_embeddings(
        papers,
        model_name=model_id,
        use_adapters=args.use_adapters,
        adapter_base_dir=args.adapter_base_dir,
        adapter_dir=args.adapter_dir,
    )

    # Save embeddings
    save_embeddings(embeddings, filenames, embeddings_output_path)

    print("Embeddings have been generated and saved.")