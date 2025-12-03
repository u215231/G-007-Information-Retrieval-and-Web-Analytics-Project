from .ranking import Ranking
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from typing import Literal, override

class WordVector(Ranking):
    """Word to vector ranking model based on gensim.models Word2Vec."""

    word2vec: Word2Vec
    """Word to vector converter. Dimensionality of the word vectors is in 
    attribute ```vector_size```."""

    update: True
    """A boolean that controls wether word to vector conveter was updated."""

    document_vectors: dict
    """Representation of documents as vectors in Word to Vector model."""


    def __init__(
            self, 
            documents: pd.Series | None = None,
            queries: pd.Series | None = None,
            word2vec: Word2Vec | None = None
        ) -> None:
        """
        Constructor for word to vector ranking model.
        """
        super().__init__(documents, queries)
        self.word2vec = word2vec if word2vec else Word2Vec()
        self.update = True
        self.document_vectors = None


    @override
    def set_items(self, items: pd.Series, type: Literal["document", "query"]) -> None:
        """
        Set documents or queries to the ranking object.
        """
        super().set_items(items, type)
        if type == "document":
            self.update = True
            self.document_vectors = None


    def get_document_sentences(self) -> list:
        """
        Returns a list of documents represented as sentences,
        where each sentence is a list of docuemnt words.
        """
        return self.get_sentences(self.documents)


    def set_word2vec_model(self, word2vec: Word2Vec) -> None:
        """
        Sets a new word to vector object with parameters.
        """
        self.word2vec = word2vec
        self.update = True


    def get_word2vec_model(self) -> Word2Vec:
        """
        Get word to vector object for the given documents in the corpus.
        """
        if not self.update:
            return self.word2vec 
        sentences = self.get_document_sentences()
        self.word2vec.build_vocab(sentences)
        self.word2vec.train(
            sentences, 
            total_examples=len(sentences), 
            epochs=self.word2vec.epochs
        )
        self.update = False
        return self.word2vec
    

    def get_word2vec_dimensions(self) -> int:
        """
        Gets the dimensionality of the vectors of the word2vec model.
        """
        return self.word2vec.vector_size


    def get_word2vecs(self, text: str) -> list:
        """
        Get the vector representation for text words.
        """
        words = str(text).split()
        word_vectors = self.get_word2vec_model().wv

        vectors = []
        for word in words:
            if word in word_vectors:
                vectors.append(word_vectors[word])

        if not vectors:
            return np.zeros(self.get_word2vec_dimensions())
        return vectors
   

    def get_average_word2vec(self, text: str) -> np.ndarray:
        """
        Get average word2vec as unit norm vector of a text. 
        """
        vectors = self.get_word2vecs(text)

        mean_vector = np.mean(np.array(vectors), axis=0)
        norm = np.linalg.norm(mean_vector)

        if norm == 0.0:
            return mean_vector
        return mean_vector / norm


    def get_document_vectors(self) -> dict:
        """
        Gets the vector representation of documents using Word2Vec. 
        Returns a dictionary {document_id: document_vector}, where
        document_vector is a np.ndarray.
        """
        if self.document_vectors is not None:
            return self.document_vectors
        
        document_vectors = {}
        for d, text in self.documents.items():
            document_vectors[d] = self.get_average_word2vec(text)

        self.document_vectors = document_vectors
        return document_vectors


    @override
    def get_query_scores(
            self, 
            query_id, 
            topK: int = 10, 
            boolean_filter: bool = True,
            **options
        ) -> dict:
        """
        Get query scores for Word to Vector model.
        """
        self.options = options

        query_text = self.queries[query_id]
        matching_documents = self.get_matching_documents(query_text)
        query_vector = self.get_average_word2vec(query_text)
        document_vectors = self.get_document_vectors()
        documents = matching_documents if boolean_filter else self.documents

        scores = {}
        for d in documents.index:
            scores[d] = np.dot(document_vectors[d], query_vector)
        
        return self.get_top_scores_dict(scores, topK)
    

    @staticmethod
    def get_sentences(items: pd.Series) -> list:
        return [str(text).split() for text in items]
