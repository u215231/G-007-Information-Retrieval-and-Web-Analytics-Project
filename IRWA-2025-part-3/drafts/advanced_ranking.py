"""
Author: Marc Bosch Manzano and Àlex Roger Moya with assistance 
of ChatGPT in BM25 and Custom scores.
"""

import pandas as pd
import math
from typing import Literal
import numpy as np
from advanced_indexing import Indexing

class Ranking:
    """
    Class for ranking documents for queries.
    """
    
    indexer: Indexing
    """Index data structure for the documents."""
    
    documents: pd.Series 
    """Documents as series with columns (document identifier, document text)."""
    
    queries: pd.Series
    """Queries as series  with columns (query identifier, query text)."""

    static_scores: pd.Series
    """Static scores for documents."""
    
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
    
    doc_lengths: dict
    """Document lengths as a dict (document, length)."""
    
    avgdl: float
    """Average document length across the corpus."""

    all_scores: dict


    def __init__(
            self, 
            documents: pd.Series | None = None,
            static_scores: pd.Series | None = None,
            queries: pd.Series | None = None
        ) -> None:
        """Constructor for ranker object."""
        self.documents = documents if documents is not None else pd.Series(dtype=object)
        self.queries = queries if queries is not None else pd.Series(dtype=object)
        self.static_scores = static_scores if static_scores is not None else pd.Series(dtype=float)
        self.df = None
        self.idf = None
        self.f = {"document": None, "query": None}
        self.tf = {"document": None, "query": None}
        self.w = {"document": None, "query": None}
        self.scores = None
        self.scores_df = None
        self.topK = None
        self.doc_lengths = None
        self.avgdl = None
        self.indexer = None
        self.query_rank_methods = {
            "tfidf": self.query_rank_tfidf, 
            "tfidf_filtered": self.query_rank_tfidf_filtered,
            "bm25": self.query_rank_bm25,
            "custom": self.query_rank_custom_score,
            "word2vec": None
        }

    def _set_items(self, items: pd.Series, type: Literal["document", "query"]) -> None:
        """Sets documents or queries to the ranking object."""
        if type == "document":
            self.documents = items
            self.df = None
            self.idf = None
            self.doc_lengths = None
            self.avgdl = None
            self.indexer = None
        else:
            self.queries = items
        self.f = {"document": None, "query": None}
        self.tf = {"document": None, "query": None}
        self.w = {"document": None, "query": None}
        self.scores = None
        self.scores_df = None
        self.topK = None


    def set_documents(self, documents: pd.Series) -> None:
        """Sets documents to the ranking object."""
        self._set_items(documents, "document")


    def set_queries(self, queries: pd.Series) -> None:
        """Sets queries to the ranking object."""
        self._set_items(queries, "query")


    def get_indexer(self) -> Indexing:
        """
        Gets the indexer object containing the inverted index of documents.
        """
        if self.indexer:
            return self.indexer
        self.indexer = Indexing(self.documents)
        self.indexer.build_inverted_index()
        return self.indexer


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
        Get frequencies of occurrence of term in document for all documents 
        and terms as a dictionary (document, (term, frequency)).
        """
        if self.f[type] is not None:
            return self.f[type]
        d = self.documents if type == "document" else self.queries
        f = {document: Ranking.get_frequencies_from_text(text) for document, text in d.items()}
        self.f[type] = f
        return f


    def get_tf(self, type: Literal["document", "query"]) -> dict:
        """
        Get term frequencies for all documents and terms as a dictionary 
        (document, (term, frequency)).
        """
        if self.tf[type] is not None:
            return self.tf[type]
        f = self.get_f(type)
        tf = {document: Ranking.get_tf_from_frequency(f[document]) for document in f}
        self.tf[type] = tf
        return tf


    def get_tfidf(self, type: Literal["document", "query"]) -> dict:
        """
        Get term weights for a queries as dict (query, (term, wieght)).
        """
        if self.w[type] is not None:
            return self.w[type]
        tf = self.get_tf(type)
        idf = self.get_idf()
        w = {query: Ranking.get_weights_from_tfidf(tf[query], idf) for query in tf}
        self.w[type] = w
        return w


    def rank_tfidf_dict(self, topK: int = 10) -> dict:
        """ 
        Get topK similarity scores between queries and documents as a dictionary 
        (query, (document, score)).
        """
        if self.scores is not None and self.topK == topK:
            return self.scores
        queires_weights = self.get_tfidf("query")
        documents_weights = self.get_tfidf("document")
        scores = {}
        for q, query_weigths in queires_weights.items():
            scores[q] = {}
            for d, document_weights in documents_weights.items():
                scores[q][d] = Ranking.get_cosine_similarity(document_weights, query_weigths)
            scores[q] = Ranking.get_top_scores_dict(scores[q], topK)
        self.scores = scores
        self.topK = topK
        return scores
    
    def query_rank_tfidf(self, query_id, topK: int = 10) -> dict:
        """ 
        Get topK similarity scores between a query and documents as a dictionary 
        (query, (document, score)).
        """
        scores = {}
        query_vectors = self.get_tfidf("query")
        document_vectors = self.get_tfidf("document")
        for d in self.documents.index:
            scores[d] = Ranking.get_cosine_similarity(document_vectors[d], query_vectors[query_id])
        scores = Ranking.get_top_scores_dict(scores, topK)
        return scores
    

    def query_rank_tfidf_filtered(self, query_id, topK: int = 10) -> dict:
        """ 
        Get topK similarity scores between a query and filtered documents by conjunctive 
        components as a dictionary (query, (document, score)).
        """
        scores = {}
        query_vectors = self.get_tfidf("query")
        document_vectors = self.get_tfidf("document")
        indexer = self.get_indexer()
        matching_documents = indexer.search_by_conjunctive_queries(self.queries.loc[query_id])
        for d in matching_documents.index:
            scores[d] = Ranking.get_cosine_similarity(document_vectors[d], query_vectors[query_id])
        scores = Ranking.get_top_scores_dict(scores, topK)
        return scores
    

    def rank_dict(
            self, 
            kind: Literal["tfidf", "tfidf_filtered", "bm25", "custom", "word2vec"], 
            topK: int = 10
        ) -> dict:
        """ 
        Get topK similarity scores between queries and documents as a dictionary
        (query, (document, score)) using a proposed model.
        """
        return {query: self.query_rank_methods[kind](query, topK=topK) for query in self.queries.index}

    
    def score_to_dataframe(
            self, 
            scores_dict: dict,
            use_query_text: bool = False,                
            use_document_text: bool = False,            
        ) -> pd.DataFrame:
        """
        Gets the dataframe representation of a dict scores (query, (document, score)).
        """
        scores_list = []
        for query, documents in scores_dict.items():
            for document, score in documents.items():
                scores_list.append({
                    "query": self.queries.loc[query] if use_query_text else query, 
                    "document": self.documents.loc[document] if use_document_text else document,
                    "score": score
                })
        scores_df = pd.DataFrame.from_records(scores_list)
        return scores_df
    

    def rank_dataframe(
            self, 
            kind: Literal["tfidf", "tfidf_filtered", "bm25", "custom", "word2vec"], 
            topK: int = 10
        ) -> dict:
        """ 
        Get topK similarity scores between queries and documents as a Pandas 
        DataFrame (query, document, score). It is possible to choose if the content
        of the columns is the identifier or the text of the document or query.
        """
        scores = self.rank_dict(kind, topK)
        scores_df = self.score_to_dataframe(scores)
        return scores_df


    def print_rankings(self, topK: int = 10) -> None:
        """
        Print rankings of topK documents for all queries one by one.
        """
        scores_merged = {}
        queries = self.queries
        documents = self.documents
        ranker = Ranking(self.documents)
        for query in queries.index:
            if self.scores is not None and self.topK == topK:
                scores = {}
                scores[query] = self.scores[query]
            else:
                ranker.set_queries(pd.Series({query: queries.loc[query]}, name="query"))
                scores = ranker.rank_tfidf_dict(topK)
                scores_merged[query] = scores[query]
            print(f"\nquery=%s, text=%s" 
                  % (query, queries.loc[query]))
            for j, document in enumerate(scores[query].keys(), 1):
                print(f"%02d: score=%1.4f, document=%s, text=%s" 
                      % (j, scores[query][document], document, documents.loc[document]))
        if self.scores is None:
            self.scores = scores_merged
        self.topK = topK


    def get_document_lenghts(self) -> dict:
        """
        Calculates and stores the length of each document as a dictionary (doc_id, doc_length).
        """
        if self.doc_lengths:
            return self.doc_lengths
        
        doc_frequencies = self.get_f("document") 
        doc_lengths = {}
        
        for doc_id, terms_freqs in doc_frequencies.items():
            length = sum(terms_freqs.values())
            doc_lengths[doc_id] = length

        self.doc_lengths = doc_lengths
        return doc_lengths


    def get_average_document_length(self) -> float:
        """
        Calculates the average document length.
        """
        if self.avgdl:
            return self.avgdl
        doc_lengths = self.get_document_lenghts()
        avgdl = sum([length for length in doc_lengths.values()]) 
        avgdl /= len(doc_lengths)
        self.avgdl = avgdl
        return avgdl


    def get_score_bm25_term(self, term: str, doc_id: str, k1: float, b: float) -> float:
        """
        Calculates the Best Match 25 score contribution for a single term in a single document.
        """

        doc_frequencies = self.get_f("document")
        f_td = doc_frequencies.get(doc_id, {}).get(term, 0)
        
        idf_t = self.get_idf().get(term, 0)
        
        length_d = self.get_document_lenghts().get(doc_id, 1) 
        avgdl = self.get_average_document_length()
        
        if f_td == 0 or idf_t == 0:
            return 0.0
        
        numerator = f_td * (k1 + 1)
        
        length_normalization = k1 * (1 - b + b * (length_d / avgdl))
        denominator = f_td + length_normalization

        return idf_t * (numerator / denominator)
    

    def get_score_bm25(self, query_terms: list[str], doc_id: str, k1: float, b: float) -> float:
        """
        Computes de Best Match 25 score for each document and query terms.
        """
        doc_bm25_score = 0.0
        for term in query_terms:
            term_score = self.get_score_bm25_term(term, doc_id, k1, b)
            doc_bm25_score += term_score
        return doc_bm25_score

    def query_rank_bm25(
            self,
            query_id,
            b: float = 0.75,
            k1: float = 1.2,
            topK: int = 10
        ):
        """
        Ranks the documents given a query following the Best Match 25 algorithm. 
        """
        indexer = self.get_indexer()
        matching_documents = indexer.search_by_conjunctive_queries(self.queries.loc[query_id])
        query_text = self.queries.loc[query_id]
        query_terms = query_text.split(" ")

        scores = {}
        for doc_id in matching_documents.index:
            print(query_terms, doc_id, k1, b)
            scores[doc_id] = self.get_score_bm25(query_terms, doc_id, k1, b)

        scores = Ranking.get_top_scores_dict(scores, topK)
        return scores      


    def get_static_scores(self) -> pd.Series:
        static_scores = self.static_scores
        min_score = static_scores.min()
        max_score = static_scores.max()
        static_scores = static_scores.fillna(min_score)
        min_max_scores = (static_scores - min_score) / (max_score - min_score)
        return min_max_scores
        

    def query_rank_custom_score(
            self, 
            query_id,
            alpha: float = 0.7,
            topK: int = 10
        ) -> list[tuple[str, float]]:
        
        """
        Ranks the filtered subset of documents using a hybrid Custom Score 
        (TF-IDF Relevance + Metadata Score).
        
        Arguments:
            alpha: Controls the weight of Relevance Score (0.7 means 70% relevance, 30% metadata).
            documents_df: A dataframe with columns (identifier, text, average_rating), where the last\
                column must have this name mandatory.
        
        Returns: 
            A tuple
        """
        
        query_vectors = self.get_tfidf("query")
        document_vectors = self.get_tfidf("document")
        document_scores = self.get_static_scores()

        indexer = self.get_indexer()
        matching_documents = indexer.search_by_conjunctive_queries(self.queries.loc[query_id])

        scores = {}
        for doc_id in matching_documents.index:
            rellevance_score = Ranking.get_cosine_similarity(document_vectors[doc_id], query_vectors[query_id])
            static_score = document_scores.loc[doc_id]
            scores[doc_id] = (alpha * rellevance_score) + ((1 - alpha) * static_score)
            
        scores = Ranking.get_top_scores_dict(scores, topK)
        return scores
    
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
    def get_top_scores_list(scores: dict, topK: int) -> list:
        """
        Get the top documents scores as a list (document, score) given an
        integer of the topK documents to be returned.
        """
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:topK]
    
    @staticmethod
    def get_top_scores_dict(scores: dict, topK: int) -> dict:
        """
        Get the top documents scores as a dictionary (document, score) given an
        integer of the topK documents to be returned.
        """
        return dict(Ranking.get_top_scores_list(scores, topK))
    

    