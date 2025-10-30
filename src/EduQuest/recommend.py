# recommender/recommender.py
import numpy as np
import pandas as pd
import heapq
import faiss
import asyncio
import logging
from typing import List, Optional, Dict, Any

from EduQuest.client import LocalOllamaClient
from EduQuest.similarity import SimilarityCalculator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class EmbeddingRecommender:
    """
    Core recommender class that uses FAISS for deterministic retrieval
    and Ollama for explanation generation.
    """

    def __init__(self, openai_client: LocalOllamaClient, similarity_calculator: SimilarityCalculator):
        self.openai_client = openai_client
        self.similarity_calculator = similarity_calculator
        self.courses_df: Optional[pd.DataFrame] = None

    # -------------------------------------------------------------------
    # Load and setup
    # -------------------------------------------------------------------
    def load_courses(self, courses_data: List[Dict[str, Any]]):
        """Load courses into memory as a pandas DataFrame."""
        self.courses_df = pd.DataFrame(courses_data)

    # -------------------------------------------------------------------
    # Step 1: Build FAISS index
    # -------------------------------------------------------------------
    async def build_faiss_index(self):
        """Compute embeddings for all courses and build FAISS index."""
        if self.courses_df is None:
            raise ValueError("Courses not loaded. Call load_courses() first.")
        
        print("🔍 Checking for existing embeddings...")
        if "embedding" in self.courses_df.columns and not self.courses_df["embedding"].isna().any():
            all_embeddings = self.courses_df["embedding"].tolist()
        else:
            all_embeddings = []
            for _, row in self.courses_df.iterrows():
                emb = await self.openai_client.generate_embedding(row["description"])
                all_embeddings.append(emb)
            self.courses_df["embedding"] = all_embeddings

        # Build FAISS index
        matrix = np.array(all_embeddings).astype("float32")
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        self.faiss_index = index
        print("✅ FAISS index built successfully.")

    # -------------------------------------------------------------------
    # Step 2: Retrieve top-k most similar courses
    # -------------------------------------------------------------------
    async def retrieve_similar_courses(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """Retrieve top-k courses most similar to the query using FAISS."""
        if not hasattr(self, "faiss_index"):
            raise ValueError("FAISS index not built. Run build_faiss_index() first.")

        query_emb = await self.openai_client.generate_embedding(query)
        query_vec = np.array([query_emb]).astype("float32")
        faiss.normalize_L2(query_vec)

        D, I = self.faiss_index.search(query_vec, top_k)
        top_courses = self.courses_df.iloc[I[0]].copy()
        top_courses["similarity"] = D[0]
        return top_courses
    
    async def retrieve_similar_courses_from_df(self, query: str, df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
        """Search top-k similar courses within a filtered DataFrame using precomputed embeddings."""
        if "embedding" not in df.columns:
            raise ValueError("Embeddings not found. Build FAISS index first.")

        all_embeddings = df["embedding"].tolist()
    
        # Build FAISS index
        matrix = np.array(all_embeddings).astype("float32")
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

        query_emb = await self.openai_client.generate_embedding(query)
        query_vec = np.array([query_emb]).astype("float32")
        faiss.normalize_L2(query_vec)

        D, I = index.search(query_vec, min(top_k, len(df)))
        top_courses = df.iloc[I[0]].copy()
        top_courses["similarity"] = D[0]
        return top_courses


    # -------------------------------------------------------------------
    # Step 3: Explain top courses
    # -------------------------------------------------------------------
    async def explain_courses(self, query: str, top_courses: pd.DataFrame) -> str:
        """Generate rationale for each course using LLM."""
        course_text = "\n".join(
            f"{row['course']}: {row['title']}\n{row['description']}"
            for _, row in top_courses.iterrows()
        )

        prompt = f"""
    You are an academic advisor.

    Student profile: {query}

    Explain why each of the following {len(top_courses)} courses fits the student's interest.
    Use this exact format (no extra text):

    1. COURSECODE: Course Title
    Rationale: two sentences focused on fit
    Confidence: High/Medium/Low

    Course list:
    {course_text}
        """

        return await self.openai_client.generate_text(prompt, model=self.openai_client.rec_model)

    # -------------------------------------------------------------------
    # Step 4: Deterministic recommender
    # -------------------------------------------------------------------
    async def recommend_deterministic(self, query: str, top_k: int = 10, levels: Optional[List[int]] = None, 
                                      prefix: Optional[List[str]] = None, type: int = 0) -> str:
        """
        Stable two-stage recommender (FAISS + LLM explanation).
        """
        try:
            if self.courses_df is None:
                raise ValueError("Courses not loaded. Call load_courses() first.")
            
            filtered_df = self.courses_df
            if levels is not None:
                filtered_df = filtered_df[filtered_df["level"].isin(levels)]
            if prefix is not None:
                filtered_df = filtered_df[filtered_df["prefix"].isin(prefix)]
            filtered_df = filtered_df.reset_index(drop=True)

            if filtered_df.empty:
                return "No courses match the given prefix/level filter."

            if type == 0:
                top_k = 50
            if prefix is None and levels is None:
                top_courses = await self.retrieve_similar_courses(query, top_k)
            else:
                top_courses = await self.retrieve_similar_courses_from_df(query, filtered_df, top_k)

            top_courses = top_courses.sort_values(by="similarity", ascending=False).reset_index(drop=True)

            if type == 0:
                return "No recommendations", top_courses
            
            recommendations = await self.explain_courses(query, top_courses)
            return recommendations, top_courses

        except Exception as e:
            logger.exception("Error in recommend_deterministic")
            return f"Error: {str(e)}"

    # -------------------------------------------------------------------
    # Original (LLM-ranking) recommender
    # -------------------------------------------------------------------
    async def recommend(self, query: str, levels: Optional[List[int]] = None, prefix: Optional[List[str]] = None, top_k_rank: int = 10, type: int = 0, rationales: Optional[int] = 0) -> str:
        """
        Legacy recommender: LLM ranks courses directly by relevance and fit.
        """
        try:
            if self.courses_df is None:
                raise ValueError("Courses not loaded. Call load_courses() first.")

            # Generate example description + embedding
            example_description = await self.generate_example_description(query)
            example_embedding = await self.openai_client.generate_embedding(example_description)

            # Filter courses
            filtered_df = self.courses_df
            if levels is not None:
                filtered_df = filtered_df[filtered_df["level"].isin(levels)]
            if prefix is not None:
                filtered_df = filtered_df[filtered_df["prefix"].isin(prefix)]
            filtered_df = filtered_df.reset_index(drop=True)

            # Find top similar
            similar_results = self.find_similar_courses(filtered_df, example_embedding)
            
            similarities = [sim for sim, idx in similar_results]
            similar_indices = [idx for sim, idx in similar_results]

            filtered_df = filtered_df.iloc[similar_indices].copy()
            
            filtered_df["similarity"] = similarities

            filtered_df = filtered_df.sort_values(by="similarity", ascending=False).reset_index(drop=True)

            if type == 0:
                return "No recommendations", filtered_df

            course_string = "\n".join(
                f"{row['course']}: {row['title']}\n{row['description']}"
                for _, row in filtered_df.iterrows()
            )

            if rationales == 0:
                recommendations = await self.explain_courses(query, filtered_df.head(10))
                return recommendations, filtered_df
            

            system_rec_message = f"""You are an expert academic advisor specializing in personalized course recommendations. \
When evaluating matches between student profiles and courses, prioritize direct relevance and career trajectory fit.

Context: Student Profile ({query})
Course Options: 
{course_string}

REQUIREMENTS:
- Return exactly 10 courses, ranked by relevance and fit
- Recommend ONLY courses listed in Course Options
- If a course is cross-listed, write the course number as "COURSEXXX (Cross-listed as COURSEYYY)"
- For each recommendation include:
  1. Course number (include cross-listed courses)
  2. Course name
  2. Two-sentence explanation focused on student's specific profile/goals
  3. Confidence level (High/Medium/Low)

FORMAT (Markdown):
1. **COURSEXXX: COURSE_TITLE**
Rationale: [Two clear sentences explaining fit]
Confidence: [Level]

2. [Next course...]

CONSTRAINTS:
- NO general academic advice
- NO mentions of prerequisites unless explicitly stated in course description
- NO suggestions outside provided course list
- NO mention of being an AI or advisor
- No duplicates; each course must be unique"""

            messages = [{"role": "system", "content": system_rec_message}]

            prompt = system_rec_message
            recommendations = await self.openai_client.generate_text(prompt, model=self.openai_client.rec_model)
            return recommendations, filtered_df[:top_k_rank]

        except Exception as e:
            return f"Error: {str(e)}"
        
    
    # -------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------
    async def generate_example_description(self, query: str) -> str:
        """
        Generate a structured example course description for the query.
        """
        system_content = f"""You will be given a request from a student to provide quality course recommendations. \
Generate a course description that would be most applicable to their request. In the course description, provide a list of topics as well as a \
general description of the course. Limit the description to be less than 200 words.

Student Request:
{query}

Output:
A natural paragraph describing an ideal course that fits the student's request(with topics and summary).
"""
        return await self.openai_client.generate_text(system_content, model=self.openai_client.generator_model)

    def find_similar_courses(self, filtered_df: pd.DataFrame, example_embedding: List[float], top_n: int = 50) -> List[int]:
        """Fallback similarity calculation without FAISS."""
        heap = []
        for idx, row in filtered_df.iterrows():
            similarity = self.similarity_calculator.calculate(example_embedding, row["embedding"])
            if len(heap) < top_n:
                heapq.heappush(heap, (similarity, idx))
            elif similarity > heap[0][0]:
                heapq.heappushpop(heap, (similarity, idx))
        return sorted(heap, key=lambda x: x[0], reverse=True)
