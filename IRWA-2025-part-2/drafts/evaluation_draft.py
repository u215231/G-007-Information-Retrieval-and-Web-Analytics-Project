import pandas as pd
import numpy as np

class ChatEvaluation:
    def precision_at_k(df: pd.DataFrame, k: int = 10) -> float:
        return df.sort_values(by="score", ascending=False).head(k)["labels"].sum() / k

    def recall_at_k(df: pd.DataFrame, k: int = 10) -> float:
        r = df["labels"].sum()
        return df.sort_values(by="score", ascending=False).head(k)["labels"].sum() / r

    def average_precision_at_k(df: pd.DataFrame, k: int = 10) -> float:
        df_sorted = df.sort_values(by="score", ascending=False).head(k)
        average_precision = 0.0
        relevant_count = 0
        for i, (_, row) in enumerate(df_sorted.iterrows(), start=1):
            if row["labels"] == 1:
                relevant_count += 1
                average_precision += relevant_count / i
        total_relevant_at_k = df_sorted["labels"].sum()
        if total_relevant_at_k == 0:
            return 0.0
        return average_precision / total_relevant_at_k

    def f1_score_at_k(df: pd.DataFrame, k: int = 10) -> float:
        p = ChatEvaluation.precision_at_k(df, k)
        r = ChatEvaluation.recall_at_k(df, k)
        return 2 * p * r / (p + r) if p + r > 0 else 0.0

    def mean_average_precision(dfs: list[pd.DataFrame]) -> float:
        return sum(ChatEvaluation.average_precision_at_k(df) for df in dfs) / len(dfs)
        
    def reciprocal_rank(df: pd.DataFrame) -> float:
        df_sorted = df.sort_values(by="score", ascending=False)
        for i, (_, row) in enumerate(df_sorted.iterrows(), start=1):
            if row["labels"] == 1:
                return 1 / i
        return 0.0

    def mean_reciprocal_rank(dfs: list[pd.DataFrame]) -> float:
        return sum(ChatEvaluation.reciprocal_rank(df) for df in dfs) / len(dfs)

    def normalized_discounted_cumulative_gain(df: pd.DataFrame, k: int = None):
        if k is None:
            k = len(df)
        df_sorted = df.sort_values(by="score", ascending=False).head(k)
        gains = df_sorted["labels"].values
        dcg = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))
        ideal = np.sort(df["labels"].values)[::-1][:k]
        idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
        return dcg / idcg if idcg > 0 else 0.0