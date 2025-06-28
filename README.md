## arXiv Curator

**arXiv Curator** is an automated system for curating personalized lists of research papers from arXiv. It generates daily recommendations by analyzing your research interests and adapting to your feedback.

This project utilized the Gemini advanced model during its development, particularly for aspects involving advanced AI capabilities for paper analysis and recommendation. It also drew inspiration from [ArxRec](https://pppoe.github.io/ArxRec/).

### Workflow Status
  * **Bootstrap from Scholar**: [![Bootstrap from Scholar](https://github.com/yilu/arxiv-curator/actions/workflows/1-bootstrap-from-scholar.yml/badge.svg)](https://github.com/yilu/arxiv-curator/actions/workflows/1-bootstrap-from-scholar.yml)
  * **Batch Process Feedback**: [![Batch Process Feedback](https://github.com/yilu/arxiv-curator/actions/workflows/2-batch-process-feedback.yml/badge.svg)](https://github.com/yilu/arxiv-curator/actions/workflows/2-batch-process-feedback.yml)
  * **Daily Recommender**: [![Daily Recommender](https://github.com/yilu/arxiv-curator/actions/workflows/3-daily-recommender.yml/badge.svg)](https://github.com/yilu/arxiv-curator/actions/workflows/3-daily-recommender.yml)
  
### Operational Overview

The system operates by establishing and maintaining a "taste profile" for research papers, which is then used to filter and rank new submissions. The key processes are:

1.  **Taste Profile Initialization:**

      * The system can be initialized by retrieving publications from a Google Scholar profile.
      * These publications are converted into numerical vector embeddings using a Sentence Transformer model. These embeddings constitute the initial "taste profile".

2.  **Daily Paper Recommendation Generation:**

      * Recent papers are fetched from specified arXiv categories.
      * An initial filtering step is applied based on predefined keywords to select relevant papers.
      * For selected candidates, an AI model (e.g., the Gemini Large Language Model) analyzes the paper's content in relation to the existing taste profile. This analysis yields a relevance score, a summary of the recommendation's basis, and suggested keywords. This process refines the ranking of potential recommendations.
      * An HTML report is then generated to display the curated list of papers.

3.  **Feedback Integration for Continuous Learning:**

      * Users can provide feedback on recommended papers by creating "Like" or "Unlike" issues on GitHub.
      * The system processes this feedback periodically, updating the "taste profile" to incorporate new preferences. This mechanism allows for continuous adaptation of recommendations.

### Automation via GitHub Actions

The curation workflow is automated through GitHub Actions:

  * **`1-bootstrap-from-scholar.yml`**: A workflow for initial taste profile bootstrapping from Google Scholar.
  * **`2-batch-process-feedback.yml`**: A daily workflow to process feedback issues and update the taste profile.
  * **`3-daily-recommender.yml`**: A daily workflow that fetches new papers, performs analysis, and generates updated recommendations.
