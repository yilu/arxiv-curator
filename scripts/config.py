# scripts/config.py

# --- Your Personal Configuration ---

# Replace this with your Google Scholar User ID.
# You can find this in the URL of your Google Scholar profile
# (e.g., [https://scholar.google.com/citations?user=YOUR_ID_HERE](https://scholar.google.com/citations?user=YOUR_ID_HERE))
GOOGLE_SCHOLAR_USER_ID = 'yBtX0C4AAAAJ'

# A list of arXiv categories you are interested in.
# Examples: 'cs.AI', 'cs.CV', 'cs.LG', 'cs.CL', 'stat.ML', 'eess.IV'
ARXIV_CATEGORIES = [
    'cond-mat.str-el',
    'cond-mat.super-con',
    'cond-mat.mtrl-sci'
]

# --- AI Model and System Configuration ---

# The name of the Sentence Transformer model to use for embeddings.
# 'all-MiniLM-L6-v2' is a great default: fast, high-quality, and runs locally.
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# The file path for storing the vector embeddings of your liked papers.
# This acts as your personal "taste" database.
TASTE_PROFILE_PATH = 'liked_vectors.json'

# The path where the final HTML page will be generated.
# The 'dist' directory is standard for distribution outputs.
GENERATED_HTML_PATH = 'dist/index.html'

# The maximum number of papers to display on each monthly archive page.
RECOMMENDATION_LIMIT = 150

# The Requests Per Minute (RPM) limit for your Gemini API tier.
# The script will automatically calculate the necessary delay based on this.
LLM_RPM_LIMIT = 15

# --- External Paper Sources ---
# URL for the manually curated list of DMRG-related preprints.
DMRG_URL = 'http://quattro.phys.sci.kobe-u.ac.jp/dmrg/condmat.html'

# The tag to display on the website for papers from this source.
DMRG_SOURCE_TAG = 'dmrg_preprints'
DMRG_SITE_LIMIT = 30