import feedparser
import os

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
    # Your arXiv author ID
    author_id = "wilson_j_3"

    # Directory to store abstracts
    abstracts_dir = "data/abstracts"

    # Fetch papers and save abstracts
    entries = fetch_papers_from_rss(author_id)
    print(f"Found {len(entries)} papers by author ID {author_id}.")
    save_abstracts_from_rss(entries, abstracts_dir)