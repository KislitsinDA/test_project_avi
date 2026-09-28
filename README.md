# Генерация кандидатов (Avito, Recall@50)

## Данные

Распаковать архив в `data/NLP_avito_interns/`:

- `train.parquet`
- `benchmark_queries.parquet`
- `benchmark_items.parquet`

## Окружение

Пошагово для venv на Windows: **[env/README.md](env/README.md)**.

Кратко ( из папки `avito`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
python download_model.py
```

- `requirements.txt` — torch **CUDA 12.4** + остальное.
- `requirements-cpu.txt` — если GPU не нужен.

Проверка CUDA: `python -c "import torch; print(torch.cuda.is_available())"`.

## Запуск

```bash
python run_solution.py --validate                 # только BM25
python run_solution.py --compare --semantic       # Recall@50: bm25 / dense / hybrid
python run_solution.py --validate --semantic      # BM25 + dense e5, RRF
python run_solution.py --validate --mode dense --semantic
python run_solution.py --submit --semantic        # answer.csv
```

Без `--semantic` Python не тянет torch и держит RAM в 2–4 ГБ (инвертированный BM25).
С `--semantic` эмбеддинги корпуса считаются один раз в `cache/item_embeddings.npy` (fp16) и дальше подхватываются из кэша.

## Подход

1. **BM25** по инвертированному индексу внутри `search_category`. Текст стеммится, описание обрезано. Буст полного совпадения в заголовке и локации. Клики из train по тексту запроса (не хардкод `query_id`).
2. **Dense**: `intfloat/multilingual-e5-small` (384 dim) через `sentence-transformers`. Документ: `passage: title + desc[:200] + params`. Запрос: `query: search_query [+ filters]`. Косинус: GPU matmul по корпусу или CPU чанками (без FAISS).
3. **RRF**: топ-100 BM25 + топ-100 dense → `w/(60+rank)` → топ-50 уникальных `item_id`.

`rank-bm25` на всём корпусе не используем: он хранит токены всех документов и на этой выборке съедал ~11 ГБ RAM.

Открытые компоненты: `intfloat/multilingual-e5-small`, `sentence-transformers`, `torch`. Инференс полностью локальный.
