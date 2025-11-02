import numpy as np
import pandas as pd

class Evaluation:
    """
    Class of evaluation metrics for predicted documents versus labeled 
    relevant documents for queries.
    """

    search_results: pd.DataFrame
    """Search Results dataframe containing at least columns (query_id, labels, score)."""
    labels: np.ndarray
    """Ground truth relevance labels (1 = relevant, 0 = non-relevant)."""
    scores: np.ndarray
    """Predicted ranking scores for the documents."""
    query_id: int | str
    """Some identifier for the query category which is selected"""

    def __init__(self, search_results: pd.DataFrame, query_id) -> None:
        self.search_results = search_results
        self.scores = self._select_scores(query_id)
        self.labels = self._select_labels(query_id)
        self.query_id = query_id

    def _select_labels(self, query_id):
        return np.array(self.search_results[self.search_results["query_id"] == query_id]["labels"])
    
    def _select_scores(self, query_id):
        return np.array(self.search_results[self.search_results["query_id"] == query_id]["score"])

    def precision_at_k(self, k: int = 10) -> float:
        """Compute Precision@K."""
        order = np.argsort(self.scores)[::-1]  
        top_doc_score = self.labels[order[:k]]
        relevant = sum(top_doc_score == 1)
        return float(relevant) / k
    
    def recall_at_k(self, k: int = 10) -> float:
        """Compute Recall@K."""
        order = np.argsort(self.scores)[::-1]  
        top_doc_score = self.labels[order[:k]]
        relevant = sum(top_doc_score == 1)
        total_relevant = sum(self.labels == 1)
        return float(relevant) / total_relevant

    def avg_precision_at_k(self, k: int = 10) -> float:
        """Compute Average Precision@K."""
        order = np.argsort(self.scores)[::-1]  
        prec_at_i = 0
        prec_at_i_list = []
        number_of_relevant = 0
        number_to_iterate = min(k, len(order))
        for i in range(number_to_iterate):
            if self.labels[order[i]] == 1:
                number_of_relevant += 1
                prec_at_i = number_of_relevant / (i + 1)
                prec_at_i_list.append(prec_at_i)
        if number_of_relevant == 0:
            return 0
        else:
            return np.sum(prec_at_i_list) / number_of_relevant
    
    def f1_score_at_k(self, k: int = 10) -> float:
        """Compute F1 Scores@K."""
        p = self.precision_at_k(k)
        r = self.recall_at_k(k)
        return (2 * p * r) / (p + r) if p + r > 0 else 0.0
    
    def map_at_k(self, k: int = 10) -> float:
        """Compute Mean Average Precision@K."""
        avp = []
        for q in self.search_results["query_id"].unique(): 
            evaluation = Evaluation(self.search_results, q)
            average_precision = evaluation.avg_precision_at_k(k)
            avp.append(average_precision) 
        return np.sum(avp) / len(avp) 
    
    def rr_at_k(self, k: int = 10) -> float:
        """Compute Reciprocal Rank Scores@K."""
        order = np.argsort(self.scores)[::-1]  
        labels = self.labels[order[:k]]  
        if np.sum(labels) == 0:  
            return 0.0
        return (np.argmax(labels == 1) + 1)  
    
    def mrr_at_k(self, k: int = 10) -> float:
        """Compute Mean Reciprocal Rank Scores@K."""
        RRs = []
        for q in self.search_results["query_id"].unique():  
            evaluation = Evaluation(self.search_results, q)
            reciprocal_rank = evaluation.rr_at_k(k)
            RRs.append(reciprocal_rank)  
        return float(sum(RRs) / len(RRs))
    
    def dcg_at_k(self, labels: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
        """Compute Discount Cumulative Gain@K."""
        order = np.argsort(scores)[::-1] 
        top_doc_score = labels[order[:k]] 
        gain = 2 ** top_doc_score - 1  
        discounts = np.log2(np.arange(len(top_doc_score)) + 2)  
        return np.sum(gain / discounts)  

    def ndcg_at_k(self, k: int = 10) -> float:
        """Compute Normalized Discount Cumulative Gain@K."""
        dcg_max = self.dcg_at_k(self.labels, self.labels, k)
        if not dcg_max:
            return 0.0
        return self.dcg_at_k(self.labels, self.scores, k) / dcg_max
    
    def print_evaluation(self, k: int = 10) -> float:
        print(
            f"""
            query_id: {self.query_id}
            1: Precision at K: {self.precision_at_k(k):.3f}
            2: Recall at K: {self.recall_at_k(k):.3f}
            3: Average Precision at K: {self.avg_precision_at_k(k):.3f}
            4: F1 score at K: {self.f1_score_at_k(k):.3f}
            5: Mean Average Precision at K: {self.map_at_k(k):.3f}
            6: Mean Reciprocal Rank at K: {self.mrr_at_k(k):.3f}
            7: Normal Discount Cumulative Gain at K: {self.ndcg_at_k(k):.3f}
            """, 
            end=""
        )