"""Side-by-side term-frequency and TF-IDF retrieval over document titles + abstracts."""
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _make_text(doc):
    # Repeating title gives it a fixed 2x field weight; both methods use the same text.
    return f'{doc["title"]} {doc["title"]} {doc["abstract"]}'

class ComparativeSearchEngine:
    def __init__(self, documents):
        self.documents = documents
        self.texts = [_make_text(d) for d in documents]
        common = dict(lowercase=True, stop_words="english", ngram_range=(1, 1), token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
        self.tf_vectorizer = CountVectorizer(**common)
        self.tfidf_vectorizer = TfidfVectorizer(**common, norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False)
        self.tf_matrix = self.tf_vectorizer.fit_transform(self.texts)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.texts)

    def search(self, query, method="tfidf", limit=5):
        query = (query or "").strip()
        if not query:
            return []
        if method.lower() in ("tf", "term-frequency", "term_frequency"):
            vectorizer, matrix = self.tf_vectorizer, self.tf_matrix
        elif method.lower() in ("tfidf", "tf-idf"):
            vectorizer, matrix = self.tfidf_vectorizer, self.tfidf_matrix
        else:
            raise ValueError("method must be 'tf' or 'tfidf'")
        qv = vectorizer.transform([query])
        # TF uses raw weighted term counts; TF-IDF uses cosine similarity.
        if method.lower() in ("tf", "term-frequency", "term_frequency"):
            scores = (matrix @ qv.T).toarray().ravel()
        else:
            scores = cosine_similarity(qv, matrix).ravel()
        ranked = []
        for doc, score in zip(self.documents, scores):
            if score > 0:
                item = dict(doc)
                item["score"] = float(score)
                ranked.append(item)
        ranked.sort(key=lambda x: (-x["score"], x["id"]))
        return ranked[:limit]

def evaluate(engine, query_set, k=5):
    report=[]
    for q in query_set:
        relevant=set(q["relevant_ids"])
        row={"query":q["query"],"type":q["type"],"relevant_count":len(relevant)}
        for method in ("tf","tfidf"):
            results=engine.search(q["query"],method=method,limit=k)
            ids=[r["id"] for r in results]
            hits=sum(1 for i in ids if i in relevant)
            row[method]={"ids":ids,"titles":[r["title"] for r in results],
                         "relevant_retrieved":hits,"precision_at_5":hits/k}
        report.append(row)
    return report

def mean_p5(report, method):
    return sum(r[method]["precision_at_5"] for r in report)/len(report) if report else 0.0
