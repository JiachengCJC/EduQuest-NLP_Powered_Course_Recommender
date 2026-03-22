import time
import os
import json
import asyncio
import threading
from queue import Queue

import streamlit as st
import pandas as pd
import numpy as np
import re

# ---- your project modules ----
# Ensure PYTHONPATH includes the folder with these files (client.py, similarity.py, recommender/recommender.py)
from EduQuest.recommend import EmbeddingRecommender
from EduQuest.client import LocalVLLMClient
from EduQuest.similarity import CosineSimilarityCalculator  # assuming you have this concrete class

# ---------------------- helpers ----------------------
def arun(coro):
    """Run async coroutines safely from Streamlit."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_queue: Queue = Queue(maxsize=1)

    def _runner():
        try:
            result_queue.put((True, asyncio.run(coro)))
        except Exception as exc:
            result_queue.put((False, exc))

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()

    ok, payload = result_queue.get()
    if ok:
        return payload
    raise payload

def sanitize_query(query: str, max_len: int = 30) -> str:
    """
    Cleans the query text for safe filename use:
    - Removes non-alphanumeric characters
    - Replaces spaces with underscores
    - Trims to a reasonable length (default 30 chars)
    """
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', query)   # keep only letters, numbers, spaces
    clean = "_".join(clean.split())                 # replace spaces with underscores
    return clean[:max_len]     

def to_markdown_filename(query: str, prefix: str = "recommendations"):
    """
    Generate a Markdown filename that includes the query snippet and timestamp.
    """
    qpart = sanitize_query(query)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{qpart}_{ts}.md"


def to_csv_filename(query: str, prefix: str = "results"):
    """
    Generate a CSV filename that includes the query snippet and timestamp.
    """
    qpart = sanitize_query(query)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{qpart}_{ts}.csv"

# ---------------------- page config ----------------------
st.set_page_config(page_title="EduQuest-NUS Recommender", layout="wide")
st.title("EduQuest-NUS Recommender")
st.caption("Hybrid interface: vLLM + Cosine vs FAISS | Type 0 = no rationales, Type 1 = with rationales")

# ---------------------- SIDEBAR: Data & Index (unchanged in spirit) ----------------------
st.sidebar.header("Data & FAISS Index")

default_data_path = "cleaned_nusmods_with_embeddings.pkl"
if not os.path.exists(default_data_path):
    fallback_data_path = os.path.join("src", default_data_path)
    if os.path.exists(fallback_data_path):
        default_data_path = fallback_data_path

data_path = st.sidebar.text_input("Courses PKL path", value=default_data_path)

st.sidebar.header("LLM Using")
with st.sidebar.expander("Model settings", expanded=False):
    
    generator_model = st.text_input(
        "Generator model name (LLM)",
        value=os.getenv("VLLM_GENERATOR_MODEL", "gemma-3-4b-it"),
    )
    rec_model = st.text_input(
        "Recommendation model name (LLM)",
        value=os.getenv("VLLM_REC_MODEL", "gemma-3-4b-it"),
    )
    embedding_model = st.text_input(
        "Embedding model name",
        value=os.getenv("VLLM_EMBEDDING_MODEL", "bge-small-en-v1.5"),
    )

    vllm_chat_base_url = st.text_input(
        "vLLM chat base URL",
        value=os.getenv("VLLM_CHAT_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    vllm_embedding_base_url = st.text_input(
        "vLLM embedding base URL",
        value=os.getenv("VLLM_EMBEDDING_BASE_URL", vllm_chat_base_url),
    )

@st.cache_resource(show_spinner=True)
def load_system(
    csv_path: str,
    generator_model_name: str,
    rec_model_name: str,
    embedding_model_name: str,
    chat_base_url: str,
    embedding_base_url: str,
):
    # 1) Load data
    if csv_path.endswith(".pkl"):
        df = pd.read_pickle(csv_path)
    else:
        raise ValueError("Unsupported file format. Please use .pkl")

    # 2) Init models
    client = LocalVLLMClient(
        generator_model=generator_model_name,
        rec_model=rec_model_name,
        embedding_model=embedding_model_name,
        chat_base_url=chat_base_url,
        embedding_base_url=embedding_base_url,
        api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
    )
    sim = CosineSimilarityCalculator()
    rec = EmbeddingRecommender(client, sim)
    rec.load_courses(df.to_dict(orient="records"))
    return rec, df

recommender, courses_df = load_system(
    data_path,
    generator_model,
    rec_model,
    embedding_model,
    vllm_chat_base_url,
    vllm_embedding_base_url,
)

# =============== NEW: Embeddings Utility ===============
st.sidebar.header("Embeddings Utility")

# 1) Check if the file has embedding column
if st.sidebar.button("🔎 Check for 'embedding' column", use_container_width=True):
    if "embedding" in courses_df.columns and not courses_df["embedding"].isna().any():
        st.sidebar.success("✅ Embeddings column found.")
        st.sidebar.caption("You can build the FAISS index directly.")
    elif "embedding" in courses_df.columns:
        st.sidebar.warning("❗ 'embedding' column exists but contains missing/invalid values.")
    else:
        st.sidebar.error("❌ 'embedding' column NOT found.")

# 2) If missing, allow generating and saving embeddings
if ("embedding" not in courses_df.columns) or courses_df["embedding"].isna().any():
    st.sidebar.write("No valid embeddings detected. You can generate and save them below.")
    out_fmt = st.sidebar.selectbox("Output format", ["Pickle (.pkl)", "CSV (.csv)"], index=0)

    if data_path.endswith(".csv"):
        out_path = data_path.replace(".csv", "_with_embeddings.csv")
        save_fmt = "csv"
    elif data_path.endswith(".pkl"):
        out_path = data_path.replace(".pkl", "_with_embeddings.pkl")
        save_fmt = "pkl"
    else:
        out_path = data_path + "_with_embeddings.pkl"
        save_fmt = "pkl"

    out_path = st.sidebar.text_input("Output file path", value=out_path)

    # Async embedding generation with a progress bar (Streamlit-friendly)
    async def generate_all_embeddings(df: pd.DataFrame):
        descs = df["description"].astype(str).tolist()
        new_embeddings = []
        progress = st.sidebar.progress(0, text="Generating embeddings…")
        total = len(descs)

        # Embed sequentially; (you can batch or parallelize with care if your backend supports it)
        for i, desc in enumerate(descs, start=1):
            emb = await recommender.openai_client.generate_embedding(desc)
            new_embeddings.append(emb)
            if i % 5 == 0 or i == total:
                progress.progress(i / total, text=f"Generating embeddings… {i}/{total}")
        progress.progress(1.0, text="Done.")
        return new_embeddings

    if st.sidebar.button("⚙️ Generate embeddings & save", type="secondary", use_container_width=True):
        try:
            t0 = time.time()
            # Generate
            embs = arun(generate_all_embeddings(courses_df))
            # Attach & reorder columns
            out_df = courses_df.copy()
            out_df["embedding"] = embs
            cols = ["course", "title", "description", "embedding", "level", "prefix", "ori_course_code"]
            existing_cols = [c for c in cols if c in out_df.columns]
            out_df = out_df[existing_cols]

            # Save
            if out_fmt.startswith("Pickle"):
                out_df.to_pickle(out_path)
            else:
                # For CSV, store embedding lists as JSON strings
                out_df_csv = out_df.copy()
                out_df_csv["embedding"] = out_df_csv["embedding"].apply(json.dumps)
                out_df_csv.to_csv(out_path, index=False)

            st.sidebar.success(f"Saved with embeddings → {out_path} (in {time.time() - t0:.2f}s)")
            st.sidebar.info("Reload this path in the 'Courses CSV path' box to work with the new file.")
        except Exception as e:
            st.sidebar.error(f"Embedding generation failed: {e}")

if st.sidebar.button("Build FAISS index", type="primary", use_container_width=True):
    t0 = time.time()
    arun(recommender.build_faiss_index())
    st.success(f"FAISS built in {time.time()-t0:.2f}s")


# ---------------------- MAIN CONTROLS ----------------------
st.subheader("Search")

query = st.text_area(
    "Student query",
    placeholder="e.g., I want to learn machine learning and deep learning for real-world applications.",
    height=90
)

model_choice = st.radio(
    "Choose model",
    options=[
        "LLM + Cosine Similarity (recommend)",
        "FAISS + Explanation (recommend_deterministic)"
    ],
    index=1,
    help="LLM+Cosine: richer reasoning (slower). FAISS: very fast, deterministic."
)

# Filters
prefixes = sorted(courses_df["prefix"].dropna().unique().tolist()) if "prefix" in courses_df.columns else []
levels = sorted(courses_df["level"].dropna().unique().tolist()) if "level" in courses_df.columns else []

sel_prefix = st.multiselect("Filter by prefix", options=prefixes)

if "sel_levels" not in st.session_state:
    st.session_state.sel_levels = []

col1, col2 = st.columns([4, 1])

with col1:
    st.session_state.sel_levels = st.multiselect(
        "Filter by level", 
        options=levels, 
        default=st.session_state.sel_levels
    )

with col2:
    if st.button("🎓 Undergrad"):
        undergrad_levels = [lvl for lvl in levels if 1000 <= int(lvl) <= 4000]

        # Toggle: if all undergrad levels already selected → clear them
        if all(lvl in st.session_state.sel_levels for lvl in undergrad_levels):
            st.session_state.sel_levels = [
                lvl for lvl in st.session_state.sel_levels if lvl not in undergrad_levels
            ]
        else:
            # Otherwise, add undergrad levels
            st.session_state.sel_levels = sorted(list(set(st.session_state.sel_levels + undergrad_levels)))

        st.rerun()

# Now you can use:
sel_levels = st.session_state.sel_levels

#sel_levels = st.multiselect("Filter by level", options=levels)

# Output Type
type_choice = st.radio(
    "Output Type",
    options=["Without Rationales", "Rationales"],
    index=0,
    help="Without Rationales = Show only top courses ranked by similarity. Rationales = Include LLM rationales."
)

rationales_mode = "Cosine similarity ranking"  # default to cosine-similarity ranking
if model_choice.startswith("LLM + Cosine") and type_choice == "Rationales":
    rationales_mode = st.radio(
        "Rationales ranking mode",
        options=["Cosine similarity ranking", "LLM ranking"],
        index=0,
        help=("Cosine similarity ranking: LLM generates rationales but courses are ranked by cosine similarity (faster).\n"
              "LLM ranking: LLM generates rationales and also ranks courses based on relevance to the query (slower, more holistic).")
    )

# FAISS extras
faiss_k = st.slider("Top-K results", min_value=5, max_value=50, value=10, step=1)

# Optional enrichment toggle for FAISS
with st.expander("Optional: FAISS enrichment settings"):
    enrich = st.checkbox(
        "Enrich query with LLM example description (slower, sometimes better)",
        value=False
    )
    combine = st.selectbox(
        "Enrichment strategy",
        ["concat", "replace"],
        index=0,
        disabled=not enrich,
        help=(
            "concat: Append the generated example to your original query.\n"
            "replace: Use only the generated example as the query embedding."
        )
        
    )
    if enrich:
        if combine == "concat":
            st.caption("**concat** → The expanded example is *added* to your query, "
                       "so the embedding captures both your intent and the generated context.")
        elif combine == "replace":
            st.caption("**replace** → The expanded example *replaces* your query entirely, "
                       "so only the generated example is embedded for FAISS search.")

# Placeholders for downloads
download_md_bytes = None
download_csv_bytes = None
download_md_name = None
download_csv_name = None

# ---------------------- RUN ----------------------
if st.button("Run recommendation", type="primary", use_container_width=True) and q.strip():
    # Ensure embeddings exist on courses_df (both paths depend on them for similarity display)
    if "embedding" not in courses_df.columns:
        st.warning("Embeddings not found on courses_df. Build or load the FAISS index from the sidebar first.")
        st.stop()

    use_levels = sel_levels or None
    use_prefix = sel_prefix or None

    if model_choice.startswith("LLM + Cosine"):
        st.info(f"Running LLM + Cosine Similarity (type={type_choice})…")
        t0 = time.time()
        try:
            rec_text_or_msg, ranked_df = arun(
                recommender.recommend(
                    query,
                    levels=use_levels,
                    prefix=use_prefix,
                    top_k_rank=faiss_k,
                    type=type_choice,
                    rationales=rationales_mode
                )
            )
            elapsed = time.time() - t0
            st.success(f"Done in {elapsed:.2f}s")

            # Show outputs
            if type_choice == "Without Rationales":
                st.markdown("### Top Courses (by similarity)")
                cols = ["course", "title", "prefix", "level", "similarity"] if "similarity" in ranked_df.columns else["course", "title", "prefix", "level"]
                table = ranked_df[cols].head(faiss_k)
                st.dataframe(table)
                # Prepare downloads
                download_csv_name = to_csv_filename(query=query, prefix="llm_cosine_top")
                download_csv_bytes = table.to_csv(index=False).encode("utf-8")
            else:
                st.markdown("### Recommendations (with rationales, return at most 10 courses)")
                st.markdown(rec_text_or_msg)
                st.markdown("### Ranked Courses (Top-K)")
                cols = ["course", "title", "prefix", "level", "similarity"] if "similarity" in ranked_df.columns else ["course", "title", "prefix", "level"]
                table = ranked_df[cols].head(faiss_k)
                st.dataframe(table)
                # Prepare downloads
                download_md_name = to_markdown_filename(query=query, prefix="llm_cosine_rationales")
                download_md_bytes = rec_text_or_msg.encode("utf-8")
                download_csv_name = to_csv_filename(query=query, prefix="llm_cosine_top")
                download_csv_bytes = table.to_csv(index=False).encode("utf-8")

        except Exception as e:
            st.error(str(e))

    else:
        # FAISS path
        if not hasattr(recommender, "faiss_index"):
            st.warning("FAISS index not found. Build or load it from the sidebar.")
            st.stop()

        st.info(f"Running FAISS (type={type_choice})…")
        t0 = time.time()

        # Optional enrichment
        text_for_embedding = query
        example = None
        if enrich:
            try:
                example = arun(recommender.generate_example_description(query))
                if combine == "replace":
                    text_for_embedding = example
                elif combine == "concat":
                    text_for_embedding = f"{query}\n\nExpanded intent:\n{example}"
            except Exception as e:
                st.warning(f"Enrichment failed: {e}")

        try:
            rec_text_or_msg, top_df = arun(
                recommender.recommend_deterministic(
                    text_for_embedding,
                    top_k=faiss_k,
                    levels=use_levels,
                    prefix=use_prefix,
                    type=type_choice
                )
            )
            elapsed = time.time() - t0
            st.success(f"Done in {elapsed:.2f}s")

            if type_choice == "Without Rationales":
                st.markdown("### Top Courses (by similarity)")
                table = top_df[["course", "title", "prefix", "level", "similarity"]].head(faiss_k)
                st.dataframe(table)
                # Prepare downloads
                download_csv_name = to_csv_filename(query=query, prefix="faiss_top")
                download_csv_bytes = table.to_csv(index=False).encode("utf-8")
            else:
                st.markdown("### Recommendations (with rationales)")
                st.markdown(rec_text_or_msg)
                st.markdown("### Ranked Courses")
                table = top_df[["course", "title", "prefix", "level", "similarity"]].head(faiss_k)
                st.dataframe(table)
                # Prepare downloads
                download_md_name = to_markdown_filename(query=query, prefix="faiss_rationales")
                download_md_bytes = rec_text_or_msg.encode("utf-8")
                download_csv_name = to_csv_filename(query=query, prefix="faiss_top")
                download_csv_bytes = table.to_csv(index=False).encode("utf-8")

        except Exception as e:
            st.error(str(e))

# ---------------------- DOWNLOAD BUTTONS ----------------------
st.divider()
st.subheader("Export")
col1, col2 = st.columns(2)
with col1:
    if download_csv_bytes is not None:
        st.download_button(
            label="⬇️ Download results table (CSV)",
            data=download_csv_bytes,
            file_name=download_csv_name,
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.caption("Run a query to enable CSV download.")
with col2:
    if download_md_bytes is not None:
        st.download_button(
            label="⬇️ Download rationales (Markdown)",
            data=download_md_bytes,
            file_name=download_md_name,
            mime="text/markdown",
            use_container_width=True
        )
    else:
        st.caption("Select Output Type = Rationales to generate rationales.")
        
# ---------------------- Footer ----------------------
with st.expander("About the models"):
    st.write("""
- **LLM + Cosine Similarity**: expands short queries with an LLM, then computes cosine similarity; richer reasoning but slower.
- **FAISS**: pre-indexed vector search for millisecond retrieval; optionally add LLM explanations.
    """)
