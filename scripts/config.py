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

# A list of keywords to perform an initial filtering on new papers.
# The script will check if any of these keywords (case-insensitive)
# are in the paper's title or abstract.
KEYWORDS = [
    # materials
    'nickelate',
    'cuprate',

    # models
    'hubbard model',
    'anderson impurity model',
    't-J model',

    #theory methods
    'matrix product state',
    'tensor network',
    'numerical renormalization group',
    'density matrix renormalization group',
    'green\'s function',
    'dynamical mean field theory',
    'machine learning',
    'neural network',
    'neural quantum state',
    'quantum monte carlo'

    #experiment methods
    'resonant inelastic x-ray scattering',
    'RIXS',
    'pump-probe',
    'nonequilibrium'
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