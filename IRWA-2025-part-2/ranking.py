import pandas as pd
import math
from typing import Literal

class Ranking:
    """
    Class for ranking documents for queries.
    """

    documents: pd.Series 
    """Documents as series with columns (document identifier, document text)."""
    queries: pd.Series
    """Queries as series  with columns (query identifier, query text)."""
    df: dict
    """Document frequencies as a dict (term, document frequency)."""
    idf: dict
    """Inverse document frequencies as a dict (term, inverse document frequency)."""
    f: dict
    """Frequencies of occurrence as a dict (type, (document or query, (term, frequency))),
    where type can only be "document" or "query"."""
    tf: dict
    """Term frequencies as a dict (type, (document or query, (term, term frequency))),
    where type can only be "document" or "query"."""
    w: dict
    """TF-IDF weights as a dict (type, (document or query, (term, TF-IDF frequency))),
    where type can only be "document" or "query"."""
    scores: dict
    """Cosine similarity scores as a dict (query, (document, score))."""
    scores_df: pd.DataFrame 
    """Cosine similarity scores as a dataframe with columns (query, document, score)."""
    topK: int
    """Number of top K documents to retrieve from a query."""

    def __init__(
            self, 
            documents: pd.Series | None = None,
            queries: pd.Series | None = None
        ) -> None:
        self.documents = documents if documents is not None else pd.Series(dtype=object)
        self.queries = queries if queries is not None else pd.Series(dtype=object)
        self.df = None
        self.idf = None
        self.f = {"document": None, "query": None}
        self.tf = {"document": None, "query": None}
        self.w = {"document": None, "query": None}
        self.scores = None
        self.scores_df = None
        self.topK = None

    def set_items(self, items: pd.Series, type: Literal["document", "query"]) -> None:
        """Sets documents or queries to the ranking object."""
        if type == "document":
            self.documents = items
            self.df = None
            self.idf = None
        else:
            self.queries = items
        self.f[type] = None
        self.tf[type] = None
        self.w[type] = None
        self.scores = None
        self.scores_df = None
        self.topK = None
    
    def set_documents(self, documents: pd.Series) -> None:
        """Sets documents to the ranking object."""
        self.set_items(documents, "document")

    def set_queries(self, queries: pd.Series) -> None:
        """Sets queries to the ranking object."""
        self.set_items(queries, "query")

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
    
    def get_f(self, type: Literal["document", "query"]) -> dict:
        """
        Get frequencies of occurrence of term in document or for all documents 
        and terms as a dictionary (document, (term, frequency)). It can also
        be computed for queries as a dictionary (query, (term, frequency)).
        """
        if self.f[type] is not None:
            return self.f[type]
        d = self.documents if type == "document" else self.queries
        f = {document: Ranking.get_frequencies_from_text(text) for document, text in d.items()}
        self.f[type] = f
        return f

    def get_tf(self, type: Literal["document", "query"]) -> dict:
        """
        Get term frequencies the terms of documents or queries as a dictionary 
        (document or query, (term, frequency)).
        """
        if self.tf[type] is not None:
            return self.tf[type]
        f = self.get_f(type)
        tf = {document: Ranking.get_tf_from_frequency(f[document]) for document in f}
        self.tf[type] = tf
        return tf
    
    def get_tfidf(self, type: Literal["document", "query"]) -> dict:
        """
        Get term weights for documents or queries as dict 
        (document or query, (term, wieght)).
        """
        if self.w[type] is not None:
            return self.w[type]
        tf = self.get_tf(type)
        idf = self.get_idf()
        w = {query: Ranking.get_weights_from_tf_idf(tf[query], idf) for query in tf}
        self.w[type] = w
        return w

    def rank_tfidf_dict(self, topK: int = 10) -> dict:
        """ 
        Get topK similarity scores between queries and documents as a dictionary 
        (query, (document, score)).
        """
        if self.scores is not None and self.topK == topK:
            return self.scores
        w = {}
        w["query"] = self.get_tfidf("query")
        w["document"] = self.get_tfidf("document")
        scores = {}
        for q, query_weigths in w["query"].items():
            scores[q] = {}
            for d, document_weights in w["document"].items():
                scores[q][d] = Ranking.get_cosine_similarity(document_weights, query_weigths)
            scores[q] = Ranking.get_top_scores(scores[q], topK)
        self.scores = scores
        self.topK = topK
        return scores
    
    def rank_tfidf_dataframe(
            self, 
            use_query_text: bool = False,                
            use_document_text: bool = False,            
            topK: int = 10                                      
        ) -> pd.DataFrame:
        """ 
        Get topK similarity scores between queries and documents as a Pandas 
        DataFrame (query, document, score). It is possible to choose if the content
        of the columns is the identifier or the text of the document or query.
        """
        scores = {}
        if self.scores is not None and self.topK == topK:
            scores = self.scores
        else: 
            scores = self.rank_tfidf_dict(topK)
        scores_df = pd.DataFrame(columns=["query", "document", "score"])
        records = []
        for query, documents in scores.items():
            for document, score in documents.items():
                records.append({
                    "query": self.queries.loc[query] if use_query_text else query, 
                    "document": self.documents.loc[document] if use_document_text else document,
                    "score": score
                })
        self.topK = topK
        scores_df = pd.DataFrame.from_records(records)
        self.scores_df = scores_df
        return scores_df

    def print_rankings(self, topK: int = 10) -> None:
        """
        Print rankings of topK documents for all queries one by one.
        """
        scores_merged = {}
        queries = self.queries
        documents = self.documents
        rank = Ranking(self.documents)
        for query in queries.index:
            if self.scores is not None and self.topK == topK:
                scores = {}
                scores[query] = self.scores[query]
            else:
                rank.set_queries(pd.Series({query: queries.loc[query]}, name="query"))
                scores = rank.rank_tfidf_dict(topK)
                scores_merged[query] = scores[query]
            print(f"\nquery=%s, text=%s" 
                  % (query, queries.loc[query]))
            for j, document in enumerate(scores[query].keys(), 1):
                print(f"%02d: score=%1.4f, document=%s, text=%s" 
                      % (j, scores[query][document], document, documents.loc[document]))
        if self.scores is None:
            self.scores = scores_merged
        self.topK = topK

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
    def get_weights_from_tf_idf(tf: dict, idf: dict) -> dict: 
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
        sim /= Ranking.get_norm(document_vector)
        sim /= Ranking.get_norm(query_vector)
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
    
    @staticmethod
    def get_top_scores(document_scores: dict, topK: int) -> dict:
        """
        Get the top documents scores as a dictionary (document, score) given an
        integer of the topK documents to be returned.
        """
        s = sorted(document_scores.items(), key=lambda item: item[1], reverse=True)
        return dict(s[:topK])