# %%
from EduQuest.client import LocalOllamaClient
from EduQuest.similarity import CosineSimilarityCalculator
from EduQuest.recommend import EmbeddingRecommender

import asyncio
import json
import re
import time
import pandas as pd
import matplotlib.pyplot as plt

# %%
# -------- Helper: extract course codes like CS3244, DSA2101 --------
    
def extract_course_codes(text):
    Sim_top10_courses = text
    Sim_results = []
    for course_code in Sim_top10_courses["course"]:
        result_code = course_code.split(" ")[0]
        Sim_results.append(result_code)

    return Sim_results


# -------- Helper: compute precision and latency for both recommenders --------
async def evaluate_one_query(courses, Sim_recommender, LLM_recommender, query, relevant_courses, top_k=10):
    row = {"query": query}
    row["len_actual_courses"] = len(relevant_courses)

    # LLM recommender
    start = time.time()
    rec_llm, rec_df = await LLM_recommender.recommend(query, type=0)

    rec_df = rec_df # no cutoff

    latency = []
    row["latency_llm"] = time.time() - start
    llm_courses = rec_df["course"].tolist()
    #llm_courses = extract_course_codes(rec_df)
    llm_relevant_courses = list(set(llm_courses) & set(relevant_courses))
    row["len_llm_relevant"] = len(llm_relevant_courses)
    row["is_llm_in_df"] = all(pd.Series(llm_courses).isin(courses["course"]))  # should be True

    # FAISS recommender
    start = time.time()
    recommend, topk_results = await Sim_recommender.recommend_deterministic(query, top_k=top_k, type=0)

    topk_results = topk_results # no cutoff

    row["latency_faiss"] = time.time() - start
    faiss_courses = topk_results["course"].tolist()
    #faiss_courses = extract_course_codes(topk_results)
    faiss_relevant_courses = list(set(faiss_courses) & set(relevant_courses))
    row["len_faiss_relevant"] = len(faiss_relevant_courses)
    row["is_sim_in_df"] = all(pd.Series(faiss_courses).isin(courses["course"]))  # should be True

    lmm_faiss_overlap = list(set(llm_courses) & set(faiss_courses))
    row["len_llm_faiss_overlap"] = len(lmm_faiss_overlap)

    return llm_courses, faiss_courses, row

# %%
async def main():
    # Load test data
    test_data = pd.read_csv("../data/tidy_student_course_selection.csv")

    # Initialize recommender
    client = LocalOllamaClient(
        generator_model="mistral",
        rec_model="qwen2.5:7b-instruct",
        embedding_model="nomic-embed-text"
    )
    sim_calc = CosineSimilarityCalculator()
    Sim_recommender = EmbeddingRecommender(client, sim_calc)
    LLM_recommender = EmbeddingRecommender(client, sim_calc)

    # Load your course dataset (replace with real data)
#    courses = pd.read_csv("../data/tidy_course_information.csv")
    embeddings = pd.read_pickle("embeddings_outsidedata.pkl")

    Sim_recommender.load_courses(embeddings)
    LLM_recommender.load_courses(embeddings)
    
    # Build Faiss Index
    await Sim_recommender.build_faiss_index()

    # Evaluate each query
    courses_results = []
    all_results_dict_s = []
    for item in test_data.itertuples(index=False):
        llm_courses, faiss_courses, all_results_dict= await evaluate_one_query(embeddings, Sim_recommender, LLM_recommender, item.query, item.CourseId)
        time.sleep(1)  # to avoid potential rate limits
        all_results_dict_s.append(all_results_dict)
  
        courses_results.append({
            "query": item.query,
            "llm_courses": llm_courses,
            "faiss_courses": faiss_courses,
        })
        print(f"✅ Finished: {item.query}, results:")
        print(list(all_results_dict.items())[1:])   # skip the first item
    
    df_with_list_courses = pd.DataFrame(courses_results)
    df_results_data = pd.DataFrame(all_results_dict_s)

    # Save to CSV
    df_results_data.to_csv("evaluation_results_data.csv", index=False)
    df_with_list_courses.to_csv("evaluation_courseslist_results.csv", index=False)


if __name__ == "__main__":
    asyncio.run(main())



