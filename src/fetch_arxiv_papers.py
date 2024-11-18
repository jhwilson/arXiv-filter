import arxiv
import os
from tqdm import tqdm
from datetime import datetime, timedelta, timezone

def fetch_recent_papers(categories, days=1):
    # Construct the search query
    category_query = ' OR '.join([f'cat:{cat}' for cat in categories])
    search_query = f'({category_query})'

    # Calculate date range using timezone-aware datetime objects
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    # Set max_results to a high number to cover all expected papers
    max_results = 200  # Adjust as needed

    print(f"Fetching papers from {start_date.date()} to {end_date.date()} in categories: {categories}")

    # Create a Client instance
    client = arxiv.Client(page_size=100, delay_seconds=3)

    # Create a Search instance
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    all_results = []

    # Iterate over results using the Client
    for result in client.results(search):
        # Ensure result.published is timezone-aware
        if result.published.tzinfo is None:
            result_published = result.published.replace(tzinfo=timezone.utc)
        else:
            result_published = result.published
        
        print(f"Result published date: {result_published}, Start date: {start_date}")

        # Stop if paper is older than start_date
        if result_published < start_date:
            break
        all_results.append(result)

    print(f"Total papers fetched: {len(all_results)}")
    return all_results

def save_papers(papers, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for paper in tqdm(papers, desc="Saving papers"):
        paper_id = paper.get_short_id()
        title = paper.title.replace('/', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_')
        filename = f"{paper_id}_{title}.txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Title: {paper.title}\n")
            f.write(f"Authors: {', '.join(author.name for author in paper.authors)}\n")
            f.write(f"Abstract:\n{paper.summary}\n")
            f.write(f"URL: {paper.pdf_url}\n")
            # Ensure paper.published is timezone-aware
            if paper.published.tzinfo is None:
                paper_published = paper.published.replace(tzinfo=timezone.utc)
            else:
                paper_published = paper.published
            published_date = paper_published.strftime('%Y-%m-%d')
            f.write(f"Date: {published_date}\n")
    print(f"Saved {len(papers)} papers to {output_dir}")

if __name__ == '__main__':
    # Define search parameters
    categories = ['cond-mat', 'math-ph', 'quant-ph']
    days = 4  # Number of days to look back
    output_dir = 'data/arxiv_papers'

    # Fetch and save papers
    papers = fetch_recent_papers(categories, days)
    save_papers(papers, output_dir)