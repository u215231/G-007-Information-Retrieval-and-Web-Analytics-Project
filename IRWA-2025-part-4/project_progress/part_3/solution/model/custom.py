import pandas as pd
from typing import override
from .tfidf import TFIDF


class Custom(TFIDF):
    """Class for custom ranking of documents."""

    static_scores: pd.Series
    """Static scores for documents sharing the same index as documents."""


    def __init__(
            self, 
            documents: pd.Series | None = None,
            static_scores: pd.Series | None = None,
            queries: pd.Series | None = None
        ) -> None:
        """
        Constructor for custom ranking object.
        """
        super().__init__(documents, queries)

        if documents is None and static_scores is None:
            static_scores = pd.Series(dtype=float)
        
        if documents is None and static_scores is not None:
            static_scores = self.apply_min_max(static_scores)
            
        if documents is not None and static_scores is None:
            static_scores = pd.Series(0.0, index=documents.index) 

        if documents is not None and static_scores is not None:
            self.check_common_keys(documents, static_scores)
        
        self.static_scores = static_scores
        self.default_options = {"alpha": 0.7}


    @override
    def set_documents(self, documents: pd.Series) -> None:
        """
        Set documents to the ranking object.
        """
        if self.static_scores is not None:
            self.check_common_keys(documents, self.static_scores)
        self.set_items(documents, "document")


    def set_static_scores(self, static_scores: pd.Series) -> None:
        """
        Set static scores to the ranking object.
        """
        if self.documents is not None:
            self.check_common_keys(self.documents, static_scores)
        self.static_scores = self.apply_min_max(static_scores)

    
    @override
    def get_query_scores(
            self, 
            query_id,
            topK: int = 10,
            boolean_filter: bool = True,
            **options
        ) -> dict:
        
        """
        Ranks the filtered subset of documents using a hybrid custom score of TF-IDF 
        Relevance plus document Metadata Score. In options, we should define ``alpha``
        value in order to give more weight to relevance or static score. It is default
        initialized to 0.70.
        """
        alpha = options["alpha"] if "alpha" in options else self.default_options["alpha"]
        options["alpha"] = alpha
        self.options = options

        query_vectors = self.get_tfidf("query")
        document_vectors = self.get_tfidf("document")
        matching_documents = self.get_matching_documents(self.queries.loc[query_id])
        documents = matching_documents if boolean_filter else self.documents

        scores = {}
        for doc_id in documents.index:
            rellevance_score = self.get_cosine_similarity(document_vectors[doc_id], query_vectors[query_id])
            static_score = self.static_scores[doc_id]
            scores[doc_id] = (alpha * rellevance_score) + ((1 - alpha) * static_score)
            
        return self.get_top_scores_dict(scores, topK)


    @staticmethod
    def apply_min_max(series: pd.Series) -> pd.Series:
        """
        Apply min max normalization to a given Pandas Series of numerical values.
        """
        min_score = series.min()
        max_score = series.max()
        series = series.fillna(min_score)
        return (series - min_score) / (max_score - min_score)
    
    @staticmethod
    def check_common_keys(documents: pd.Series, static_scores: pd.Series) -> None:
        if not all(documents.index == static_scores.index):
            raise Exception('Not common keys between "document" and "static_scores".')
        return