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

After following the installation section and modifying `config.yaml` (the `default_author_id` and `categories` are the most personalized elements), you can run these from the top directory

```bash
python src/fetch_abstracts.py # Only if you configured default_author_id in config.yaml
python src/run_pipeline.py
```

Then use your favorite markdown viewer to inspect the new `recommendations/recommendations_YYYY-MM-DD.md` file created.

## Overview

The pipeline automates the process of staying updated with the latest research relevant to your interests by:

- Fetching the most recent arXiv papers in specified categories.
- Preprocessing abstracts (tokenization, stopword removal, lemmatization).
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

### 2. Set Up a Virtual Environment (Optional but Recommended)

```bash
python3 -m venv env
source env/bin/activate  # On Windows, use 'env\Scripts\activate'
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Resources

Run the following commands to download necessary NLTK data:

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```
## Configuration

Customize the pipeline by editing the config.yaml file:

```yaml
# config.yaml

# Author ID for arXiv
default_author_id: 'wilson_j_3'

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
embedding_model: 'allenai/specter2'

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
    - Example: `default_author_id` and `abstracts_dir: data/abstracts`)
2. Command-line overrides
   - You can override the YAML settings by providing arguments via the command line:
    - --author_id: Specify the arXiv author ID (e.g., wilson_j_3).
	- --abstracts_dir: Set the directory where abstracts should be saved.
	- --config_file: Specify an alternative YAML configuration file. 
3. Fetching and Saving:
   - The script fetches the RSS feed for the given author ID, parses the abstracts, and saves them as .txt files in the specified directory. Each file includes the paper’s title, authors, abstract, URL, and publication date.


### 2. Run the Pipeline

```bash
python src/run_pipeline.py
### 2b. Optional: Run the UI

After installing requirements:

```bash
source env/bin/activate
streamlit run app/ui.py
```

The UI provides:
- A sidebar listing past `recommendations_YYYY-MM-DD.md`
- Tabs for Priority (whitelist authors) and Other recommendations
- Buttons to reload your papers and to run the pipeline
- A settings page to edit key values in `config.yaml`

```
- The pipeline will process your abstracts (if updated), fetch new arXiv papers, process them, compute similarities, and generate recommendations.

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
- src/preprocess.py: Preprocesses abstracts (tokenization, stopword removal, lemmatization).
- src/compute_embeddings.py: Generates embeddings for abstracts using a pre-trained model.
- src/compute_similarity.py: Computes similarity scores between your papers and arXiv papers.
- src/recommend.py: Generates recommendations based on similarity scores and outputs them in Markdown format.

## Dependencies

- Python 3.6+
- Required Python Packages (listed in requirements.txt):
```
arxiv
numpy
scikit-learn
sentence-transformers
transformers
torch
nltk
tqdm
pyyaml
```
- NLTK Data Packages:
  - punkt
  - stopwords
  - wordnet

## Troubleshooting

### NLTK Data Errors

If you encounter errors related to NLTK data not being found, ensure you’ve downloaded the necessary resources:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```
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