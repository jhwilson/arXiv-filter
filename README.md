# ArXiv Paper Recommendation Pipeline

This project is a pipeline that fetches recent papers from arXiv in specified categories, preprocesses the abstracts, computes embeddings, calculates similarities with your own papers, and generates recommendations of papers most similar to your work.

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Scripts Description](#scripts-description)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Quick Start

After following the installation section and modifying `config.yaml` (especially `categories`, plus `default_author_id` if you want auto-fetch), you can run these from the top directory:

```bash
# Optional: only if you configured default_author_id in config.yaml (or pass --author_id)
python src/fetch_abstracts.py
python src/run_pipeline.py
```

Then use your favorite markdown viewer to inspect the new `recommendations/recommendations_YYYY-MM-DD.md` file created.

## Overview

The pipeline automates the process of staying updated with the latest research relevant to your interests by:

- Fetching the most recent arXiv papers in specified categories.
- Preprocessing abstracts into title + abstract text payloads.
- Generating embeddings using a pre-trained language model.
- Computing similarities between arXiv papers and your own papers.
- Generating recommendations in Markdown format with hyperlinks to the papers.

## Features

- Customizable categories and date ranges for fetching arXiv papers.
- Automatic detection of updates to your own papers.
- Configurable preprocessing and embedding generation.
- Command-line interface and configuration file for easy customization.
- Recommendations saved with timestamps for historical tracking.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jhwilson/arXiv-filter.git
cd arXiv-filter
```

### 2. Create a Python environment

#### 2a. Conda (recommended)

```bash
conda create -n arxiv-filter python=3.10 -y
conda activate arxiv-filter
```

#### 2b. Virtualenv (alternative)

```bash
python3 -m venv env
source env/bin/activate  # On Windows, use 'env\Scripts\activate'
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. First-run model download note

On first run, Hugging Face models may be downloaded (often several GB total depending on selected models). This can take a while.
## Configuration

Customize the pipeline by editing the config.yaml file:

```yaml
# config.yaml

# Author ID for arXiv
default_author_id: ''  # Optional; set this to your arXiv author ID to enable auto-fetch

# General settings
check_my_papers: true  # Set to false to skip checking your own papers

# Paths
my_abstracts_dir: 'data/abstracts'
arxiv_abstracts_dir: 'data/arxiv_papers'
processed_data_dir: 'data/processed'
embeddings_dir: 'models'

# arXiv fetch settings
categories:
  - 'cond-mat.dis-nn'
  - 'cond-mat.mes-hall'
  - 'cond-mat.mtrl-sci'
  - 'cond-mat.other'
  - 'cond-mat.quant-gas'
  - 'cond-mat.soft'
  - 'cond-mat.stat-mech'
  - 'cond-mat.str-el'
  - 'cond-mat.supr-con'
  - 'math-ph'
  - 'quant-ph'
days: 1  # Number of days before the most recent arXiv paper

# Embedding model
embedding_model: 'allenai/specter2_aug2023refresh'
enabled_recommendation_models:
  - 'specter2_refresh'   # add 'physbert' to compare both
physbert_model: 'thellert/accphysbert_cased'

# SPECTER2 adapters mode
specter2_use_adapters: true
local_specter2_base_dir: 'allenai/specter2_aug2023refresh_base'
local_specter2_adapter_dir: 'allenai/specter2_aug2023refresh'

# Recommendations
top_n: 10  # Number of top recommendations to display
similarity_threshold: 0.0  # Minimum similarity score to consider
recommendations_dir: 'recommendations'  # Directory to save recommendation files
recommendations_output_base: 'recommendations'

# Cleanup settings
cleanup_old_recommendations: false
days_to_keep: 7  # Number of recent recommendations files to keep
```
## Usage

### 1. Prepare Your Abstracts

- Place your paper abstracts in the data/abstracts directory.
- Each abstract should be in a separate .txt file.
- The file should include the following fields:
```
Title: Your Paper Title
Authors: Your Name, Collaborator Name
Abstract:
Your abstract text goes here.
Date: YYYY-MM-DD
```

### **Optional**: Auto-generate abstracts

```bash
python src/fetch_abstracts.py
```

1. YAML Configuration
    - By default, the script reads settings from `config.yaml` file. This should specify the author's arXiv ID and the directory where abstracts will be saved. 
    - Example: `default_author_id` and `my_abstracts_dir: data/abstracts`
2. Command-line overrides
   - You can override the YAML settings by providing arguments via the command line:
    - --author_id: Specify the arXiv author ID (e.g., wilson_j_3).
	- --abstracts_dir: Set the directory where abstracts should be saved.
	- --config_file: Specify an alternative YAML configuration file. 
3. Fetching and Saving:
   - The script fetches the RSS feed for the given author ID, parses the abstracts, and saves them as .txt files in the specified directory. Each file includes the paper’s title, authors, abstract, URL, and publication date.
4. If you skip auto-fetch:
   - Manually place your own abstracts in `data/abstracts` before running the pipeline.


### 2. Run the Pipeline

```bash
python src/run_pipeline.py
```

### 2b. Optional: Run the UI

After installing requirements, activate your environment and run:

```bash
# Conda
conda activate arxiv-filter
streamlit run app/ui.py
```

```bash
# Virtualenv
source env/bin/activate
streamlit run app/ui.py
```

The UI provides:
- A sidebar listing past `recommendations_YYYY-MM-DD.md`
- Tabs for Priority (whitelist authors) and Other recommendations
- Buttons to reload your papers and to run the pipeline
- A settings page to edit key values in `config.yaml` including model selection (SPECTER2 refresh, PhysBERT, or both)

Whitelist/blacklist author files are stored under `config/` and are intentionally gitignored so private names are not committed.

The pipeline will process your abstracts (if updated), fetch new arXiv papers, process them, compute similarities, and generate recommendations.

### 2c. Optional: A/B test embedding models

Compare your current config model (now SPECTER2 refresh + adapters) against a PhysBERT variant:

```bash
# Quick smoke test on subsets first
python src/ab_test_models.py --model_b thellert/accphysbert_cased --max_my 50 --max_arxiv 500 --top_n 20

# Full run
python src/ab_test_models.py --model_b thellert/accphysbert_cased --top_n 20
```

This writes `recommendations/model_ab_test.md` with:
- top-N lists from each model,
- top-N overlap count / Jaccard agreement,
- per-paper scores for side-by-side comparison.

### 3. View Recommendations

- The recommendations will be saved under `recommendations/` with a filename like `recommendations_YYYY-MM-DD.md`.
- Open the Markdown file with a viewer or editor to see your personalized recommendations.

### 4. Command-Line Options

- To force checking for updates to your papers:
```bash
python src/run_pipeline.py --check_my_papers
```
- To specify a different configuration file:
```bash
python src/run_pipeline.py --config my_config.yaml
```

## Scripts Description

- src/run_pipeline.py: Orchestrates the entire pipeline.
- src/fetch_arxiv_papers.py: Fetches recent arXiv papers based on categories and date range.
- src/preprocess.py: Preprocesses abstracts into title + abstract text payloads.
- src/compute_embeddings.py: Generates embeddings for abstracts using a pre-trained model.
- src/compute_similarity.py: Computes similarity scores between your papers and arXiv papers.
- src/recommend.py: Generates recommendations based on similarity scores and outputs them in Markdown format.
- src/recommend_ensemble.py: Generates recommendations by fusing multiple model rankings when more than one model is selected.
- app/ui.py: Streamlit UI to browse results, run the pipeline, and adjust settings.

## Dependencies

- Python 3.8+
- Required Python packages (from `requirements.txt`):
```
arxiv
feedparser
numpy
scikit-learn
sentence-transformers
transformers
torch
adapters
streamlit
tqdm
pyyaml
watchdog
```

## Troubleshooting

### Missing your abstracts

If `src/run_pipeline.py` reports no abstracts in `data/abstracts`:
- add your own `.txt` abstracts (see format above), or
- run `python src/fetch_abstracts.py --author_id <your_arxiv_author_id>`.
### Timezone and Date Issues

Ensure your system’s date and time settings are correct to avoid issues with fetching papers based on dates.

### arXiv Fetching Limitations

- arXiv may limit the number of queries you can make in a short period.
- If you encounter errors fetching papers, try increasing the delay_seconds parameter in fetch_arxiv_papers.py or reduce the frequency of your requests.

### Memory Errors

- For large datasets, you may encounter memory issues.
- Consider increasing your system’s memory or modifying the scripts to process data in smaller batches.

## License

This project is licensed under the MIT License - see the LICENSE file for details.