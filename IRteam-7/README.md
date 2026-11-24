# IRteam-7: TF vs TF-IDF Comparative Information Retrieval Project

## Project objective
Compare **Term Frequency (TF)** and **TF-IDF** ranking over document **titles and abstracts**. Test broad and topic-specific queries and evaluate the top five results using Precision@5.

## What is included
- `dataset.json`: 36 records with title, abstract, category, source URL and source note.
- `queries.json`: 10 prepared broad and topic-specific queries with relevance judgments (`relevant_ids`).
- `search_engine.py`: shared preprocessing and two ranking methods.
- `streamlit_app.py`: interactive side-by-side search, evaluation dashboard and collection browser.
- `tests/`: basic correctness tests.

The abstracts are concise educational summaries written for this project. The source links are background references, not a claim that each summary is a verbatim abstract from the linked page. Verify and replace/add source URLs if your instructor requires record-by-record bibliographic citations.

## Methodology
1. Both methods use the same 36 records, query set, lowercase conversion, English stop-word removal, and unigram tokenization.
2. Each record is represented by its title and abstract. The title is repeated once, giving title terms a consistent 2x weight in both methods.
3. **TF:** CountVectorizer produces raw term counts; query-document scores are the dot product of the query count vector and document count vectors.
4. **TF-IDF:** TfidfVectorizer applies term-frequency and inverse-document-frequency weighting; cosine similarity ranks documents.
5. Results are sorted by score descending; document ID breaks ties.
6. Relevance judgments were assigned before running either method. Precision@5 = relevant documents among the first five / 5. If fewer than five results are returned, missing positions count as non-relevant.

## Dashboard navigation
The dashboard workspace navigation (Search comparison, Evaluation, Collection) is displayed horizontally at the top of the page, above the dashboard metrics.

## Run on Windows
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```
Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Run tests
```powershell
python -m pytest -v
```

## Reproduce evaluation from command line
```powershell
python evaluate.py
```
This prints per-query ranked IDs and Precision@5, plus mean Precision@5 for each method.

## Evaluation discussion to include in report
- **Where TF may help:** broad queries with repeated terms can reward documents that discuss those terms repeatedly.
- **Where TF may fail:** long records or repeated generic terms can dominate the ranking even if the terms are not distinctive.
- **Where TF-IDF may help:** focused queries containing distinctive terms benefit when rare terms receive more weight.
- **Where TF-IDF may fail / offer little improvement:** queries with rare terms already have similar rankings under both methods; lexical TF-IDF also misses semantic matches expressed using synonyms.
- Report the actual measured scores from `evaluate.py` rather than claiming one method always wins.

## Limitations
This is a small educational collection and a lexical baseline. Relevance judgments are manually curated for demonstration and should ideally be reviewed by multiple assessors. The source links are background references. This is not a production search engine.
