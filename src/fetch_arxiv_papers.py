import arxiv
import os
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
import argparse

def fetch_most_recent_paper_date(categories):
    # Construct the search query
    category_query = ' OR '.join([f'cat:{cat}' for cat in categories])
    search_query = f'({category_query})'
    # Create a Client instance
    client = arxiv.Client(page_size=1, delay_seconds=3)
    # Create a Search instance to fetch only the most recent paper
    search = arxiv.Search(
        query=search_query,
        max_results=1,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    # Get the most recent paper
    results = list(client.results(search))
    if results:
        most_recent_paper = results[0]
        if most_recent_paper.published.tzinfo is None:
            end_date = most_recent_paper.published.replace(tzinfo=timezone.utc)
        else:
            end_date = most_recent_paper.published
        return end_date
    else:
        return None

def fetch_papers_in_date_range(categories, start_date, end_date):
    # Construct the search query
    category_query = ' OR '.join([f'cat:{cat}' for cat in categories])
    search_query = f'({category_query})'
    # Initialize variables
    all_results = []
    max_results = 1000  # Upper bound; client handles paging
    print(f"Fetching papers from {start_date.date()} to {end_date.date()} in categories: {categories}")
    # Create a Client instance
    # Smaller page size is more reliable with some arxiv lib versions
    client = arxiv.Client(page_size=100, delay_seconds=3)
    # Create a Search instance
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    # Iterate over results using the Client
    try:
        for result in client.results(search):
            # Ensure result.published is timezone-aware
            if result.published.tzinfo is None:
                result_published = result.published.replace(tzinfo=timezone.utc)
            else:
                result_published = result.published
            # Only include papers within the date range
            if start_date <= result_published <= end_date:
                all_results.append(result)
            elif result_published < start_date:
                # Since results are sorted in descending order, we can break early
                break
    except arxiv.UnexpectedEmptyPageError as e:
        # Benign: happens when we page beyond available results; keep what we have
        print(f"Warning: encountered empty page while fetching results; proceeding with {len(all_results)} items.")
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
    parser = argparse.ArgumentParser(description='Fetch recent arXiv papers.')
    parser.add_argument('--categories', nargs='+', default=[
        'cond-mat.dis-nn',
        'cond-mat.mes-hall',
        'cond-mat.mtrl-sci',
        'cond-mat.other',
        'cond-mat.quant-gas',
        'cond-mat.soft',
        'cond-mat.stat-mech',
        'cond-mat.str-el',
        'cond-mat.supr-con',
        'math-ph',
        'quant-ph'
    ], help='List of arXiv categories to search.')
    parser.add_argument('--days', type=int, default=1, help='Number of days before the date of the most recent article.')
    parser.add_argument('--output_dir', type=str, default='data/arxiv_papers', help='Directory to save fetched papers.')

    args = parser.parse_args()

    # Fetch the most recent paper date
    most_recent_date = fetch_most_recent_paper_date(args.categories)
    if most_recent_date is None:
        # If no papers found, use current date
        end_date = datetime.now(timezone.utc)
        print("Warning: Could not find any recent papers. Using current date as end_date.")
    else:
        end_date = most_recent_date
        print(f"Most recent paper date: {end_date.date()}")

    # Calculate start date
    start_date = end_date - timedelta(days=args.days)

    # Fetch papers in date range
    papers = fetch_papers_in_date_range(args.categories, start_date, end_date)

    # Save fetched papers
    save_papers(papers, args.output_dir)