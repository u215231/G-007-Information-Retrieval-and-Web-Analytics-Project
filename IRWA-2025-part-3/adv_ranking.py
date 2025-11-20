import pandas as pd
import numpy as np

from ranking import Ranking
from indexing import Indexing

class AdvancedRanking(Ranking):
    """
    Part 3: TF-IDF + AND, BM25, OwnScore, Word2Vec.
    """
    def __init__(
        self,
        documents: pd.Series | None = None,
        queries: pd.Series | None = None,
        indexer: Indexing | None = None) -> None:
        
        super().__init__(documents, queries)
        self.indexer = indexer # inverted index + AND search
        self._bm25_state = None # cache for BM25 params
        self.metadata = None # product info (rating, discount, etc.)
        self.scores_bm25 = None # BM25 scores
        self.scores_own = None # own score

    def set_indexer(self, indexer: Indexing) -> None:
        self.indexer = indexer

    # 1a. TF-IDF + AND
    def rank_tfidf_and(self, topK: int = 10) -> dict:
        """
        TF-IDF ranking over documents filtered by AND queries.
        returns {query_id: {doc_id: score}} with topK documents.
        """
        if self.indexer is None:
            raise ValueError("No hay indexer. Usa set_indexer() primero.")

        # calculated TF-IDF weights
        w_query = self.get_tfidf("query") # q: {term: weight}
        w_doc   = self.get_tfidf("document") # d: {term: weight}
        scores = {}

        for qid, q_vec in w_query.items():
            q_text = self.queries.loc[qid]

            # candidates by AND
            cand_docs = self.indexer.search_by_conjunctive_queries(q_text)
            doc_scores = {}

            # cosine only in those candidates
            for d in cand_docs:
                doc_vec = w_doc.get(d)
                if doc_vec is None:
                    continue
                sim = Ranking.get_cosine_similarity(doc_vec, q_vec)
                if sim > 0:
                    doc_scores[d] = sim

            # keep topK
            scores[qid] = Ranking.get_top_scores(doc_scores, topK)

        return scores
    
    # 1b. bm25 helpers
    def get_score_bm25_term(self, term, doc_id, doc_lengths, avgdl, k1, b):
        """
        BM25 score for one term in one document.
        """
        
        f_doc = self.get_f("document") # term frequencies in docs
        tfs = f_doc.get(doc_id, {})
        f_td = tfs.get(term, 0) # freq of term in doc

        idf = self.get_idf() # idf(t) = log(N/df)
        IDF_t = idf.get(term, 0.0)

        if f_td == 0 or IDF_t == 0:
            return 0.0

        L_d = doc_lengths.get(doc_id, 1)

        # bm25 formula
        num = f_td * (k1 + 1)
        denom = f_td + k1 * (1 - b + b * (L_d / avgdl))

        return IDF_t * (num / denom)

    # bm25 main
    def rank_bm25_and(self, topK=10, k1=1.2, b=0.75):
        """
        BM25 ranking over AND candidates.
        returns {query_id: {doc_id: score}}
        """
        if self.indexer is None:
            raise ValueError("no indexer set")

        # doc lengths and average length
        f_doc = self.get_f("document")      
        doc_lengths = {d: sum(freqs.values()) for d, freqs in f_doc.items()}
        avgdl = sum(doc_lengths.values()) / (len(doc_lengths) or 1)

        f_q = self.get_f("query") # term frequencies in queries
        scores = {}

        for qid, q_freqs in f_q.items():
            q_text = self.queries.loc[qid]
            terms = list(q_freqs.keys())

            # candidate docs by AND
            cand_docs = self.indexer.search_by_conjunctive_queries(q_text)
            doc_scores = {}

            # bm25 score for each candidate doc
            for d in cand_docs:
                s = 0.0
                for t in terms:
                    s += self.get_score_bm25_term(t, d,
                                                doc_lengths, avgdl,
                                                k1, b)
                if s > 0:
                    doc_scores[d] = round(s, 4)

            # keep topK
            scores[qid] = Ranking.get_top_scores(doc_scores, topK)

        self.scores_bm25 = scores
        return scores

    # 1c. own score
    def set_metadata(self, metadata_df, id_column="pid"):
        """
        Store metadata (rating, discount) indexed by pid.
        """
        meta = metadata_df.copy()
        meta[id_column] = meta[id_column].astype(str)
        self.metadata = meta.set_index(id_column)

    def rank_own_from_tfidf_and(self, topK=10, w_text=0.7, w_rating=0.2, w_discount=0.1):
        """
        Own score = TF-IDF score + rating + discount.
        returns {query_id: {doc_id: score}}.
        """
        if self.metadata is None:
            raise ValueError("no metadata set")

        # base: tf-idf + AND
        base = self.rank_tfidf_and(topK=topK)
        meta = self.metadata
        scores = {}

        for qid, ds in base.items():
            scores[qid] = {}
            for pid, s_text in ds.items():

                if pid in meta.index:
                    row = meta.loc[pid]

                    # rating [0,5] -> [0,1]
                    r = float(row["average_rating"]) / 5.0 if "average_rating" in row and not pd.isna(row["average_rating"]) else 0.0

                    # discount [0,100] -> [0,1]
                    d = float(row["discount"]) / 100.0 if "discount" in row and not pd.isna(row["discount"]) else 0.0

                    # weighted score
                    s = w_text * s_text + w_rating * r + w_discount * d
                else:
                    # no metadata -> only text score
                    s = s_text

                scores[qid][pid] = round(s, 4)

            # keep topK
            scores[qid] = Ranking.get_top_scores(scores[qid], topK)

        self.scores_own = scores
        return scores

    # 2. word2vec
    def text_to_vec_w2v(self, text, w2v):
        """
        text -> average word2vec vector (normalized).
        """
        words = str(text).split()
        vecs = []

        # collect word vectors
        for w in words:
            if w in w2v:
                vecs.append(w2v[w])

        if not vecs:
            return None

        v = np.mean(np.array(vecs), axis=0)
        norm = np.linalg.norm(v)
        if norm == 0:
            return None

        # return unit vector for cosine
        return v / norm

    def rank_w2v_and(self, w2v, topK=20):
        """
        word2vec + cosine ranking over AND candidates.
        returns {query_id: {doc_id: score}}.
        """
        if self.indexer is None:
            raise ValueError("no indexer set")

        scores = {}

        # precompute document vectors
        doc_vecs = {}
        for pid, text in self.documents.items():
            v = self.text_to_vec_w2v(text, w2v)
            if v is not None:
                doc_vecs[pid] = v

        # ranking per query
        for qid, q_text in self.queries.items():
            # candidate docs by AND
            cand_docs = self.indexer.search_by_conjunctive_queries(q_text)

            q_vec = self.text_to_vec_w2v(q_text, w2v)
            if q_vec is None:
                scores[qid] = {}
                continue

            result = {}

            # cosine similarity
            for pid in cand_docs:
                d_vec = doc_vecs.get(pid)
                if d_vec is None:
                    continue
                score = float(np.dot(d_vec, q_vec)) # both vectors normalized
                if score > 0:
                    result[pid] = round(score, 4)

            # keep topK
            sorted_docs = sorted(result.items(), key=lambda x: x[1], reverse=True)
            scores[qid] = dict(sorted_docs[:topK])

        return scores