import json
from pathlib import Path
from search_engine import ComparativeSearchEngine, evaluate
ROOT=Path(__file__).resolve().parents[1]
DOCS=json.loads((ROOT/"dataset.json").read_text(encoding="utf-8"))
QUERIES=json.loads((ROOT/"queries.json").read_text(encoding="utf-8"))
def test_collection_size_and_fields():
 assert 30 <= len(DOCS) <= 50
 assert all(d.get("title") and d.get("abstract") and d.get("source") for d in DOCS)
def test_both_methods_return_ranked_results():
 e=ComparativeSearchEngine(DOCS)
 assert e.search("inverse document frequency","tf")
 assert e.search("inverse document frequency","tfidf")
def test_empty_query():
 assert ComparativeSearchEngine(DOCS).search("   ","tfidf")==[]
def test_ten_queries_have_judgments():
 assert len(QUERIES)>=10
 assert all(q["relevant_ids"] for q in QUERIES)
def test_evaluation_has_both_methods():
 r=evaluate(ComparativeSearchEngine(DOCS),QUERIES)
 assert len(r)==len(QUERIES)
 assert all("tf" in x and "tfidf" in x for x in r)
