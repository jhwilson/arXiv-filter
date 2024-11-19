import feedparser
import os
import argparse
import yaml

def load_config(config_file):
    """Load configuration from a YAML file."""
    if os.path.exists(config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)
    else:
        print(f"Configuration file {config_file} not found. Using defaults.")
        return {}

def fetch_papers_from_rss(author_id):
    feed_url = f"https://arxiv.org/a/{author_id}.atom2"
    feed = feedparser.parse(feed_url)
    return feed.entries

def save_abstracts_from_rss(entries, abstracts_dir):
    os.makedirs(abstracts_dir, exist_ok=True)
    for entry in entries:
        paper_id = entry.id.split('/')[-1]
        title = entry.title.replace('/', '_').replace(':', '_')
        filename = f"{paper_id}_{title}.txt"
        filepath = os.path.join(abstracts_dir, filename)
        
        if not os.path.exists(filepath):
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Title: {entry.title}\n")
                f.write(f"Authors: {entry.author}\n")
                f.write(f"Abstract:\n{entry.summary}\n")
                f.write(f"URL: {entry.link}\n")
                f.write(f"Date: {entry.published}\n")
            print(f"Saved abstract to {filepath}")
        else:
            print(f"Abstract already exists: {filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and save abstracts from arXiv RSS feed")
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--author_id",
        type=str,
        help="The arXiv author ID (e.g., wilson_j_3)"
    )
    parser.add_argument(
        "--abstracts_dir",
        type=str,
        help="Directory to save abstracts"
    )

    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config_file)

    # Resolve values with precedence: command-line > YAML > hardcoded defaults
    author_id = args.author_id or config.get('default_author_id', 'wilson_j_3')
    abstracts_dir = args.abstracts_dir or config.get('my_abstracts_dir', 'data/abstracts')

    # Fetch papers and save abstracts
    entries = fetch_papers_from_rss(author_id)
    print(f"Found {len(entries)} papers by author ID {author_id}.")
    save_abstracts_from_rss(entries, abstracts_dir)