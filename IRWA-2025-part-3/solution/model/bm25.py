import pandas as pd
from typing import Literal, override
from .tfidf import TFIDF


class BM25(TFIDF):
    """
    Class for ranking documents for queries using Best Match 25.
    """

    doc_lengths: dict
    """Document lengths as a dict (document, length)."""
    
    avgdl: float
    """Average document length across the corpus."""


    def __init__(
            self, 
            documents: pd.Series | None = None,
            queries: pd.Series | None = None
        ) -> None:
        """
        Constructor for Best Match 25 ranking object.
        """
        super().__init__(documents, queries)
        self.default_options: dict = {"b": 0.75, "k1": 1.2}
        self.doc_lengths = None
        self.avgdl = None


    @override
    def set_items(self, items: pd.Series, type: Literal["document", "query"]) -> None:
        """
        Sets documents or queries to the BM25 ranking object.
        """
        super().set_items(items, type)
        if type == "document":
            self.doc_lengths = None
            self.avgdl = None

    
    def get_document_lenghts(self) -> dict:
        """
        Calculates and stores the length of each document as a dictionary (doc_id, doc_length).
        """
        if self.doc_lengths is not None:
            return self.doc_lengths
        
        doc_frequencies = self.get_f("document") 
        doc_lengths = {}
        
        for doc_id, terms_frequencies in doc_frequencies.items():
            doc_lengths[doc_id] = sum(terms_frequencies.values())

        self.doc_lengths = doc_lengths
        return doc_lengths


    def get_average_document_length(self) -> float:
        """
        Calculates the average document length.
        """
        if self.avgdl is not None:
            return self.avgdl
        doc_lengths = self.get_document_lenghts()

        avgdl = sum([length for length in doc_lengths.values()]) 
        avgdl /= len(doc_lengths)

        self.avgdl = avgdl
        return avgdl


    def get_bm25_term_score(self, term: str, doc_id: str, k1: float, b: float) -> float:
        """
        Calculates the Best Match 25 score contribution for a single term in a single document.
        """
        f_td = self.get_f("document").get(doc_id, {}).get(term, 0)        
        idf_t = self.get_idf().get(term, 0)
        
        if f_td == 0.0 or idf_t == 0.0:
            return 0.0
        
        length_d = self.get_document_lenghts().get(doc_id, 1) 
        avgdl = self.get_average_document_length()
        
        numerator = f_td * (k1 + 1)
        denominator = f_td + (k1 * (1 - b + b * (length_d / avgdl)))

        return idf_t * (numerator / denominator)
    

    def get_bm25_document_score(self, query_terms: list, doc_id: str, k1: float, b: float) -> float:
        """
        Computes de Best Match 25 score for each document and query terms.
        """
        return sum(self.get_bm25_term_score(t, doc_id, k1, b) for t in query_terms)


    @override
    def get_query_scores(
            self,
            query_id,
            topK: int = 10,
            boolean_filter: bool = True,
            **options
        ):
        """
        Ranks the documents given a query following the Best Match 25 algorithm. 
        The parameters in options should be ``b``, default initialized to 0.75,
        and ```k1```, default initialized to 1.2.
        """
        b = options["b"] = options["b"] if "b" in options else self.default_options["b"]
        k1 = options["k1"] = options["k1"] if "k1" in options else self.default_options["k1"]
        self.options = options

        query_text = self.queries[query_id]
        query_terms = query_text.split(" ")
        matching_documents = self.get_matching_documents(query_text)
        documents = matching_documents if boolean_filter else self.documents

        scores = {}
        for doc_id in documents.index:
            scores[doc_id] = self.get_bm25_document_score(query_terms, doc_id, k1, b)

        return super().get_top_scores_dict(scores, topK)    