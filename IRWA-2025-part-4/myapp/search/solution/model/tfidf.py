import pandas as pd
import math
from typing import Literal, override
from .ranking import Ranking


class TFIDF(Ranking):
    """
    Class for ranking documents for queries using Term Frequency
    Inverse Document Frequency model.
    """

    df: dict
    """Document frequencies as a dict (term, document frequency)."""
    
    idf: dict
    """Inverse document frequencies as a dict (term, inverse document frequency)."""
    
    f: dict
    """Frequencies of occurrence as a dict (kind, (document or query, (term, frequency))),
    where kind can only be "document" or "query"."""
    
    tf: dict
    """Term frequencies as a dict (kind, (document or query, (term, term frequency))),
    where kind can only be "document" or "query"."""
    
    tfidf: dict
    """TF-IDF weights as a dict (kind, (document or query, (term, TF-IDF frequency))),
    where kind can only be "document" or "query"."""


    def __init__(
            self, 
            documents: pd.Series | None = None,
            queries: pd.Series | None = None
        ) -> None:
        """
        Constructor for Term Frequency Inverse Document Frequency ranking object.
        """
        super().__init__(documents, queries)
        self.df = None
        self.idf = None
        self.f = {"document": None, "query": None}
        self.tf = {"document": None, "query": None}
        self.tfidf = {"document": None, "query": None}

    @override
    def set_items(self, items: pd.Series, kind: Literal["document", "query"]) -> None:
        """
        Set documents or queries to the TFIDF ranking object.
        """
        super().set_items(items, kind)
        if kind == "document":
            self.df = None
            self.idf = None
        self.f = {"document": None, "query": None}
        self.tf = {"document": None, "query": None}
        self.tfidf = {"document": None, "query": None}


    def get_df(self) -> dict:
        """
        Get document frequencies for all terms in documents as a 
        dictitonary (term, number of documents).
        """
        if self.df is not None:
            return self.df
        df = {} 
        for text in self.documents:
            terms = set(text.split())
            for term in terms:
                df[term] = df.get(term, 0) + 1
        self.df = df
        return df


    def get_idf(self) -> dict: 
        """
        Get inverse document frequencies for all terms in documents as 
        a dictionary (term, inverse DF).
        """
        if self.idf is not None:
            return self.idf
        df = self.get_df()
        N = len(self.documents)  
        idf = {term: math.log(N / df[term], 2) for term in df}
        self.idf = idf
        return idf 


    def get_f(self, kind: Literal["document", "query"]) -> dict:
        """
        Get frequencies of occurrence of term in document for all documents 
        and terms as a dictionary (document, (term, frequency)). The same procedure
        can be applied to queries.
        """
        if self.f[kind] is not None:
            return self.f[kind]
        items = self.documents if kind == "document" else self.queries
        f = {id: self.get_frequencies_from_text(text) for id, text in items.items()}
        self.f[kind] = f
        return f


    def get_tf(self, kind: Literal["document", "query"]) -> dict:
        """
        Get term frequencies for all documents or queries and terms as a dictionary 
        (document or query, (term, frequency)).
        """
        if self.tf[kind] is not None:
            return self.tf[kind]
        f = self.get_f(kind)
        tf = {document: self.get_tf_from_frequency(f[document]) for document in f}
        self.tf[kind] = tf
        return tf


    def get_tfidf(self, kind: Literal["document", "query"]) -> dict:
        """
        Get term weights for documents or queries as dict (document or query, (term, wieght)).
        """
        if self.tfidf[kind] is not None:
            return self.tfidf[kind]
        tf = self.get_tf(kind)
        idf = self.get_idf()
        tfidf = {query: TFIDF.get_weights_from_tfidf(tf[query], idf) for query in tf}
        self.tfidf[kind] = tfidf
        return tfidf
    

    @override
    def get_query_scores(
            self, 
            query_id, 
            topK: int = 10, 
            boolean_filter: bool = True,
            **options
        ) -> dict:
        """ 
        Get topK TFIDF similarity scores between a query and filtered documents by conjunctive 
        components as a dictionary (document, score). No options needed.
        """
        self.options = options

        query_vectors = self.get_tfidf("query")
        document_vectors = self.get_tfidf("document")
        matching_documents = self.get_matching_documents(self.queries[query_id])
        documents = matching_documents if boolean_filter else self.documents

        scores = {}
        for d in documents.index:
            scores[d] = self.get_cosine_similarity(document_vectors[d], query_vectors[query_id])
        return self.get_top_scores_dict(scores, topK)


    @staticmethod
    def get_frequencies_from_text(text: str) -> dict:
        """
        Gets the frequency of occurrence of each term in text as a dict 
        (term, frequency).
        """
        count = {}
        terms = text.split()
        for term in terms:
            count[term] = count.get(term, 0) + 1
        return count
    
    @staticmethod
    def get_tf_from_frequency(frequencies: dict) -> dict:
        """
        Gets the term frequencies from a frequencies dict (term, frequency)
        as another dict (term, term frequency).
        """
        return {term: 1 + math.log(frequencies[term]) for term in frequencies}
    
    @staticmethod
    def get_weights_from_tfidf(tf: dict, idf: dict) -> dict: 
        """
        Gets the TF-IDF weights from a term frequency dictionary (term, term frequency) 
        and inverse document frequency dictionary (term, inverse DF) as another dictionary
        (term, TF-IDF weight).
        """
        return {term: tf[term] * idf.get(term, 0) for term in tf}
    
    @staticmethod
    def get_cosine_similarity(document_vector: dict, query_vector: dict) -> float: 
        """
        Gets the cosine similarity between a document and a query both expressed 
        as dictionaries (term, TF-IDF weight).
        """
        sim = 0.0
        for term in query_vector:
            wd = document_vector.get(term, 0)
            wq = query_vector[term]
            sim += wd * wq
        sim /= TFIDF.get_norm(document_vector)
        sim /= TFIDF.get_norm(query_vector)
        return sim

    @staticmethod
    def get_norm(vector: dict) -> float:
        """ 
        Gets the norm of a vector expressed as a dictionary (term, TF-IDF weight).
        """
        norm = 0
        for weight in vector.values():
            norm += weight**2
        norm = math.sqrt(norm)
        return norm