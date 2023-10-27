##########################################################################################
# Description: High-level project configurations.
##########################################################################################

from radqg.utils.general_utils import redirect_path
from radqg.settings.apis import POURIA_OPENAI_API_KEY

# ----------------------------------------------------------------------------------------
# API and token configs
# ----------------------------------------------------------------------------------------

OPENAI_API_KEY = POURIA_OPENAI_API_KEY  # Enter your OpenAI API key here

# ----------------------------------------------------------------------------------------
# Logging configs
# ----------------------------------------------------------------------------------------

VERBOSE = True

# ----------------------------------------------------------------------------------------
# Path configs
# ----------------------------------------------------------------------------------------

TOY_DATA_DIR = redirect_path("data/brain_tumor_articles")
VECTOR_DB_DIR = redirect_path("data/vector_db")

# ----------------------------------------------------------------------------------------
# Document extraction configs
# ----------------------------------------------------------------------------------------

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# ----------------------------------------------------------------------------------------
# Retrieval augmentation arguments
# ----------------------------------------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-ada-002"
VECTOR_DB = "in-memory"
SEARCH_TYPE = "similarity"
K = 6
FETCH_K = None
COMPRESSOR = None
TEMPERATURE = 0.2
CHAIN_TYPE = "stuff"
# MODEL = "gpt-3.5-turbo"
# MODEL = "gpt-4-32k"
MODEL = "gpt-4"

# ----------------------------------------------------------------------------------------
# GradIO arguments
# ----------------------------------------------------------------------------------------

GR_PORT_NUMBER = 1901
GR_SERVER_NAME = "0.0.0.0"
GR_PUBLIC_SHARE = True
GR_CONCURRENCY_COUNT = 30
