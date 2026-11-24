import json
from pathlib import Path
import pandas as pd
import streamlit as st
from search_engine import ComparativeSearchEngine, evaluate, mean_p5

ROOT = Path(__file__).resolve().parent
DOCS = json.loads((ROOT / "dataset.json").read_text(encoding="utf-8"))
QUERIES = json.loads((ROOT / "queries.json").read_text(encoding="utf-8"))
ENGINE = ComparativeSearchEngine(DOCS)

st.set_page_config(
    page_title="TF vs TF-IDF | IR Team 7",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp { background: #0b1020; color: #e8edf8; }
    .block-container { padding: 1.5rem 2.2rem 3rem; max-width: 1500px; }
    h1, h2, h3 { color: #f3f6ff; }
    p, label, .stMarkdown { color: #d5deef; }
    div[data-testid="stMetric"] {
        background: #151e33; border: 1px solid #293653;
        padding: 16px; border-radius: 12px;
    }
    .result {
        background: #151e33; border: 1px solid #293653;
        padding: 14px 16px; border-radius: 12px; margin: 9px 0;
    }
    .small { color: #9eacc5; font-size: 13px; }
    .pill { color: #8ee5cf; font-weight: 700; }
    div[role="radiogroup"] {
        background: #151e33; border: 1px solid #293653;
        border-radius: 13px; padding: 10px 14px;
        gap: 12px; width: fit-content;
    }
    div[role="radiogroup"] label {
        background: transparent; padding: 7px 14px;
        border-radius: 8px; color: #e8edf8 !important;
    }
    div[role="radiogroup"] label:hover { background: #263653; }
    div[data-testid="stRadio"] > label { display: none; }
    a { color: #78b7ff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔎 Retrieval Ranking Lab")
st.caption(
    "A controlled comparison of Term Frequency (TF) and TF-IDF over document titles and abstracts."
)

# Workspace navigation is displayed horizontally at the top of the dashboard.
page = st.radio(
    "Workspace",
    ["Search comparison", "Evaluation", "Collection"],
    horizontal=True,
    label_visibility="collapsed",
    key="workspace_navigation",
)

st.markdown("---")

# Compact collection overview below the horizontal navigation.
metric1, metric2, metric3 = st.columns(3)
metric1.metric("Documents", len(DOCS))
metric2.metric("Test queries", len(QUERIES))
metric3.metric("Ranking methods", "2", help="Term Frequency and TF-IDF")
st.markdown("")

if page == "Search comparison":
    st.subheader("Run the same query through both ranking methods")
    q = st.text_input(
        "Search query",
        placeholder="e.g. term frequency inverse document frequency",
    )
    examples = [item["query"] for item in QUERIES]
    selected = st.selectbox(
        "Or choose a prepared test query", ["— Select —"] + examples
    )
    if selected != "— Select —":
        q = selected
    k = st.slider("Results to display", 3, 10, 5)

    if q.strip():
        left, right = st.columns(2, gap="large")
        for col, method, label in [
            (left, "tf", "Term Frequency (TF)"),
            (right, "tfidf", "TF-IDF"),
        ]:
            with col:
                st.markdown(f"### {label}")
                results = ENGINE.search(q, method, k)
                if not results:
                    st.info("No matching documents.")
                for rank, doc in enumerate(results, 1):
                    st.markdown(
                        f"""<div class="result">
                        <div class="pill">#{rank} · score {doc['score']:.4f}</div>
                        <b>{doc['title']}</b>
                        <div class="small">{doc['category']} · Record {doc['id']}</div>
                        <p>{doc['abstract']}</p>
                        <a href="{doc['source']}" target="_blank">Reference source ↗</a>
                        </div>""",
                        unsafe_allow_html=True,
                    )
    else:
        st.info(
            "Enter a query or select one of the prepared test queries to compare ranked outputs."
        )

elif page == "Evaluation":
    st.subheader("Offline evaluation — Precision@5")
    st.write(
        "Relevance IDs were assigned before ranking, using the intended information need for each query. "
        "Both methods use the same 36-record collection and preprocessing."
    )
    report = evaluate(ENGINE, QUERIES, 5)
    summary = pd.DataFrame(
        [
            {
                "Query": row["query"],
                "Type": row["type"],
                "TF P@5": row["tf"]["precision_at_5"],
                "TF-IDF P@5": row["tfidf"]["precision_at_5"],
                "TF relevant": row["tf"]["relevant_retrieved"],
                "TF-IDF relevant": row["tfidf"]["relevant_retrieved"],
            }
            for row in report
        ]
    )
    a, b = st.columns(2)
    a.metric("Mean TF Precision@5", f"{mean_p5(report, 'tf'):.3f}")
    b.metric("Mean TF-IDF Precision@5", f"{mean_p5(report, 'tfidf'):.3f}")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.download_button(
        "Download evaluation CSV",
        summary.to_csv(index=False).encode("utf-8"),
        "precision_at_5.csv",
        "text/csv",
    )
    st.markdown("#### Per-query ranked document IDs")
    for row in report:
        with st.expander(row["query"]):
            st.write("Relevant IDs:", row["relevant_count"])
            st.write(
                "TF:", row["tf"]["ids"],
                " · P@5 =", round(row["tf"]["precision_at_5"], 3),
            )
            st.write(
                "TF-IDF:", row["tfidf"]["ids"],
                " · P@5 =", round(row["tfidf"]["precision_at_5"], 3),
            )
    st.markdown("#### Interpretation")
    st.write(
        "TF rewards repeated query terms and can over-rank longer records or documents that repeat common query words. "
        "TF-IDF reduces the influence of terms that occur in many documents, often helping topic-specific queries "
        "with distinctive vocabulary."
    )
    st.write(
        "TF-IDF may offer little improvement when query terms are already rare and highly focused. "
        "It can also fail when a relevant record uses synonyms or different wording, because this lexical baseline "
        "does not understand meaning."
    )

else:
    st.subheader("Document collection")
    st.write(
        "The collection contains 36 educational records across six information-retrieval themes. "
        "Each record has a title, abstract, category, and source reference."
    )
    categories = ["All"] + sorted({doc["category"] for doc in DOCS})
    category = st.selectbox("Filter category", categories)
    rows = [
        doc for doc in DOCS
        if category == "All" or doc["category"] == category
    ]
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "ID": doc["id"],
                    "Title": doc["title"],
                    "Category": doc["category"],
                    "Abstract": doc["abstract"],
                    "Source": doc["source"],
                }
                for doc in rows
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
