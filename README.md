# Data Science Docs Assistant

A retrieval-augmented generation (RAG) assistant over the reference docs for **pandas, NumPy, SciPy, scikit-learn, matplotlib and Plotly**. It answers only from the documentation, cites the library, section and page for every answer, and refuses when the docs don't cover the question.

**Live app:** https://data-science-library.streamlit.app

| | |
|---|---|
| Knowledge base | 6 reference PDFs, 1,040 pages, ~2.1M characters |
| Right source in the top 5 results | 100% (57 answerable eval questions) |
| Out-of-scope questions refused | 15 / 15, with 0 false refusals |
| Answers judged faithful to the context | 97% (blind LLM-as-judge, 3 runs) |

## How it works

```
question ─► embed ─► Supabase pgvector (top 5) ─► similarity gate ─► LLM (structured output) ─► citations from metadata
                                                     │                    │
                                                  refuse              refuse if in_context = False
```

1. **Ingestion.** PDFs are extracted with PyMuPDF. pypdf was rejected after it glued 14,341 words together (vs 104 real long class names with PyMuPDF). Running page headers are parsed into `library`, `version` and `section` metadata and then stripped; unmapped font glyphs are translated back to text; text is NFKC-normalized.
2. **Chunking and embedding.** Page-level chunks (`RecursiveCharacterTextSplitter`, 2,000 / 200 characters), embedded with `text-embedding-3-large` at 1,536 dimensions and stored in Supabase pgvector with an HNSW index and a `match_documents` RPC.
3. **Retrieval.** The question is embedded and matched against the top 5 chunks, optionally filtered to one library.
4. **Two refusal layers.**
   - *Gate:* if the best similarity is below `MIN_SIM = 0.40`, the app refuses without calling the LLM (cost: $0).
   - *LLM:* the structured `Answer` schema puts `in_context: bool` **first**, so the model judges coverage before it writes. It must set it False when the context only mentions a library by name.
5. **Citations.** The model returns only the numbers of the context blocks it used. Library, version, section, page and docs URL are filled in by code from stored metadata, so a citation can never point to a page that doesn't exist.

Answers stream token by token (structured output via a JSON-schema dict), and every answer shows its token count and cost.

## Evaluation

A 72-question set (57 answerable with an expected library and section, 15 deliberately out of scope, e.g. XGBoost, Dask, seaborn, statsmodels) drove every design decision.

**Retrieval, page vs section chunking (n = 57)**

| Metric | Page | Section |
|---|---|---|
| hit@1 | **0.842** | 0.789 |
| hit@3 | 0.947 | **0.965** |
| hit@5 | 1.000 | 1.000 |
| MRR | **0.899** | 0.873 |

**Answer quality**, judged by a stronger model than the one that wrote the answers (to avoid self-preference bias), blind to the strategy, with A/B order randomized, over 3 runs:

| | Page | Section |
|---|---|---|
| Faithful | **0.970** | 0.952 |
| Completeness (1–5) | **4.76** | 4.66 |
| Input tokens per 56 answers | **131K** | 162K |

The difference wasn't statistically significant (sign test p = 0.405), but page-level chunking was never behind and used about 19% fewer input tokens, so it shipped.

**Refusal threshold.** A cutoff sweep set `MIN_SIM = 0.40`: it keeps every answerable question, while the gate alone rejects 10 of 15 out-of-scope ones. The LLM layer catches the remaining 5, including statsmodels, which leaked until the `in_context` description was tightened.

**Known limits.** Most faithfulness flags were correct facts the model added from its own training rather than from the context. 0 leaks out of 15 still allows a true leak rate of up to ~20% (rule of three), so more near-miss questions would tighten that bound. Each question is answered on its own, with no follow-up memory.

## Repository

| File | Purpose |
|---|---|
| `datascience_rag.ipynb` | Full build: ingestion, chunking, schema, embedding, evaluation, LLM-as-judge |
| `rag.py` | Retrieval and answer pipeline shared by the notebook and the app (`ask`, `ask_stream`) |
| `app.py` | Streamlit chat UI: citation cards, library filter, refusal state, per-answer cost |
| `tests/test_rag.py` | Unit tests with all API clients mocked (no network calls, no cost) |
| `judge_answers.parquet`, `judge_results.parquet`, `eval_vectors.parquet` | Frozen eval inputs and results, so the judge run never has to be repeated |
| `data-science-knowledge-base/` | The six reference PDFs |

## Run locally

Python 3.12 (the pinned dependencies have no wheels for 3.14 yet).

```sh
pip install -r requirements.txt
```

Create `.env` (or `.streamlit/secrets.toml`):

```
OPENAI_API_KEY=...
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_KEY=...   # publishable/anon key for the app; service-role key only for ingestion
```

```sh
streamlit run app.py
python -m pytest -q tests/
```

The deployed app reads with the anon key under a read-only RLS policy; the service-role key is used only by the notebook for ingestion and is never deployed.

## Stack

Python · LangChain · OpenAI · Supabase pgvector · PyMuPDF · Streamlit · pytest
