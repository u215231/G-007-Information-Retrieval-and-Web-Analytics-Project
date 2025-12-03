import pandas as pd
from typing import Literal
from ..indexing import Indexing


class Ranking:
    """
    Abstract class for ranking documents based on boolean ranking model.
    """

    documents: pd.Series 
    """Documents as series with columns [document identifier, document text]."""

    queries: pd.Series
    """Queries as series  with columns [query identifier, query text]."""

    indexer: Indexing
    """Index data structure for the documents."""

    scores: dict
    """Similarity scores as a dict {query, {document, score}}."""

    scores_df: pd.DataFrame
    """Pandas dataframe of similarity scores with columns [query, document, score]."""

    topK: int
    """Number of top K documents to retrieve from a query."""

    boolean_filter: bool
    """Apply conjunctive component filter for documents and queries when ranking."""

    options: dict
    """Options for special ranking algorithms."""

    default_options: dict
    """Default options for ranking object."""


    def __init__(
            self, 
            documents: pd.Series | None = None,
            queries: pd.Series | None = None
        ) -> None:
        """
        Constructor for ranking object.
        """
        self.documents = documents if documents is not None else pd.Series(dtype=object)
        self.queries = queries if queries is not None else pd.Series(dtype=object)
        self.indexer = None
        self.scores = None
        self.scores_df = None
        self.topK = None
        self.boolean_filter = None
        self.options = None
        self.default_options = {}


    def set_items(self, items: pd.Series, kind: Literal["document", "query"]) -> None:
        """
        Set documents or queries to the ranking object.
        """
        if kind == "document":
            self.documents = items
        else:
            self.queries = items
        self.scores = None
        self.topK = None
        self.boolean_filter = None
        self.options = None


    def set_documents(self, documents: pd.Series) -> None:
        """
        Set documents to the ranking object.
        """
        self.set_items(documents, "document")


    def set_queries(self, queries: pd.Series) -> None:
        """
        Set queries to the ranking object.
        """
        self.set_items(queries, "query")


    def get_indexer(self) -> Indexing:
        """
        Get the indexer object containing the inverted index of documents.
        """
        if self.indexer is not None:
            return self.indexer
        self.indexer = Indexing(self.documents)
        self.indexer.build_inverted_index()
        return self.indexer
    

    def get_matching_documents(self, query_text: str) -> pd.Series:
        """
        Gets a series of [document_id, document_text] cotaining the query words.
        """
        indexer = self.get_indexer()
        return indexer.search_by_conjunctive_queries(query_text)


    def get_query_scores(
            self, 
            query_id, 
            topK: int = 10, 
            boolean_filter: bool = True,
            **options
        ) -> dict:
        """
        Get binary similarities of query and documents as a dict 
        {document: score} where score is 1.0 if document has all words
        of query or 0.0 otherwise.
        """
        self.options = options

        query_text = self.queries[query_id]
        matching_documents = self.get_matching_documents(query_text)
        documents = matching_documents if boolean_filter else self.documents

        scores = {}
        for d in documents.index:
            scores[d] = float(d in documents.index)
        
        return self.get_top_scores_dict(scores, topK)


    def get_scores(
            self, 
            topK: int = 10, 
            boolean_filter: bool = True,
            **options
        ) -> dict:
        """ 
        Get topK similarity scores between queries and documents as a dictionary
        {query: {document: score}}. It is overriden by iheriting classes.

        Options
        -------
        - BM25: default initialized parameters ``b = 0.75`` and ``k1 = 1.2``.\n
        - Custom: default initialized parameter ``alpha = 0.70``.
        - Ranking, TF-IDF, and Word2Vec: no options needed.
        """
        if (
            self.scores is not None 
            and self.topK == topK 
            and self.boolean_filter == boolean_filter
            and self.options == options
        ):
            return self.scores
        
        scores = {}
        for q in self.queries.index:
            scores[q] = self.get_query_scores(q, topK, boolean_filter, **options) 

        self.scores = scores
        self.topK = topK
        self.boolean_filter = boolean_filter
        return scores
    

    def get_scores_df(
            self, 
            topK: int = 10, 
            boolean_filter: bool = True,
            **options
        ) -> pd.DataFrame:
        """
        Get topK similarity scores between queries and documents as a Pandas DataFrame
        with columns [query_id, query_text, document_id, document_text, score].
        
        Options
        -------
        - BM25: default initialized parameters ``b = 0.75`` and ``k1 = 1.2``.\n
        - Custom: default initialized parameter ``alpha = 0.70``.
        - Ranking, TF-IDF, and Word2Vec: no options needed.
        """
        scores = self.get_scores(topK=topK, boolean_filter=boolean_filter, **options)
        scores_df = self.score_to_dataframe(scores)
        self.scores_df = scores_df
        return scores_df


    def score_to_dataframe(self, scores_dict: dict) -> pd.DataFrame:
        """
        Gets the dataframe representation of a dict scores {query: {document, score}}.
        """
        scores_list = []
        for query_id, documents in scores_dict.items():
            for document_id, score in documents.items():
                scores_list.append({
                    "query_id": query_id,
                    "query_text": self.queries[query_id],
                    "document_id": document_id,
                    "document_text": self.documents[document_id],
                    "score": score
                })
        scores_df = pd.DataFrame.from_records(scores_list)
        return scores_df


    @staticmethod
    def get_top_scores_list(scores: dict, topK: int) -> list:
        """
        Get the top documents scores as a list [(document, score)] given an
        integer of the topK documents to be returned.
        """
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:topK]
    

    @staticmethod
    def get_top_scores_dict(scores: dict, topK: int) -> dict:
        """
        Get the top documents scores as a dictionary {document: score} given an
        integer of the topK documents to be returned.
        """
        return dict(Ranking.get_top_scores_list(scores, topK))