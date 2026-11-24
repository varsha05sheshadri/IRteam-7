import json
from pathlib import Path
from search_engine import ComparativeSearchEngine, evaluate, mean_p5

root = Path(__file__).resolve().parent
docs = json.loads((root / 'dataset.json').read_text(encoding='utf-8'))
queries = json.loads((root / 'queries.json').read_text(encoding='utf-8'))
report = evaluate(ComparativeSearchEngine(docs), queries)
for row in report:
    print(f"\n{row['type']}: {row['query']}")
    for method in ('tf', 'tfidf'):
        print(f"  {method.upper():5} IDs={row[method]['ids']} P@5={row[method]['precision_at_5']:.2f}")
print(f"\nMean TF P@5: {mean_p5(report, 'tf'):.3f}")
print(f"Mean TF-IDF P@5: {mean_p5(report, 'tfidf'):.3f}")
