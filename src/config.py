from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "NLP_avito_interns"
CACHE_DIR = ROOT / "cache"
OUTPUT_DIR = ROOT

TRAIN_PATH = DATA_DIR / "train.parquet"
QUERIES_PATH = DATA_DIR / "benchmark_queries.parquet"
ITEMS_PATH = DATA_DIR / "benchmark_items.parquet"
ANSWER_PATH = OUTPUT_DIR / "answer.csv"

EMBEDDINGS_PATH = CACHE_DIR / "item_embeddings.npy"
ITEM_IDS_CACHE = CACHE_DIR / "item_ids.txt"

# multilingual-e5-small: 384 dim, нормально лезет в 6 ГБ VRAM
MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_DIR = CACHE_DIR / "multilingual-e5-small"
RANDOM_SEED = 42

TOP_K = 50
# сколько кандидатов с каждой стороны перед RRF
CANDIDATE_N = 100
RRF_K = 60
BM25_WEIGHT = 1.0
# чуть выше BM25, если dense на hold-out сильнее — подкрутим после сравнения
DENSE_WEIGHT = 1.2

EMBED_BATCH = 256
EMBED_MAX_SEQ = 80
# для e5 описание короче BM25: в эмбеддинг и так уезжает мало токенов
EMBED_DESC_CHARS = 200
QUERY_BATCH = 32

HISTORY_TOP_N = 12
CLICK_PER_TOKEN = 40

# Hold-out по тексту запроса, без подглядывания в те же клики
MAX_VAL_QUERIES = 400

# Описание режем, иначе индекс раздувается. 420 символов хватает, чтобы слово из запроса
# чаще попадало в документ, и при этом RAM остаётся в разумных пределах.
MAX_DESC_CHARS = 420
MAX_ITEM_TOKENS = 80

ITEM_COLS = [
    "item_id",
    "item_title_raw",
    "item_description_raw",
    "item_infm_params_text",
    "item_category_id",
    "item_location_id",
    "item_rating",
]

TRAIN_COLS = [
    "search_query",
    "search_location_id",
    "search_category",
    "search_infm_params_text",
    "search_is_delivery_search",
    "item_id",
]

# устаревшие имена — чтобы старый код/ноутбук не падал при импорте
BM25_QUOTA = 30
SEMANTIC_PREFETCH = CANDIDATE_N
