import jieba
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
import pickle
import os


class TagSuggester:
    def __init__(self, stop_words: Optional[List[str]] = None, max_features: int = 5000):
        self.max_features = max_features
        self.stop_words = stop_words or []
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            tokenizer=self._tokenize,
            token_pattern=None
        )
        self.mlb = MultiLabelBinarizer()
        self.classifier = MultiOutputClassifier(LogisticRegression(max_iter=1000))
        self.is_trained = False
        self.tag_descriptions = {}
        self.documents = []
        self.document_tags = []
        self._idf_ = None

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r'[^\w\s]', ' ', text)
        words = jieba.lcut(text)
        return [w for w in words if len(w) > 1 and w not in self.stop_words]

    def add_tag_description(self, tag: str, description: str):
        self.tag_descriptions[tag] = description

    def fit(self, documents: List[str], tags_list: List[List[str]]):
        self.documents = list(documents)
        self.document_tags = [list(tags) for tags in tags_list]

        X = self.vectorizer.fit_transform(documents)
        y = self.mlb.fit_transform(tags_list)
        self.classifier.fit(X, y)
        self._build_tag_tfidf()
        self.is_trained = True

    def _build_tag_tfidf(self):
        self.tag_vectors = {}
        for tag in self.mlb.classes_:
            tag_docs = [doc for doc, tags in zip(self.documents, self.document_tags) if tag in tags]
            if tag_docs:
                vec = self.vectorizer.transform(tag_docs).mean(axis=0)
                self.tag_vectors[tag] = np.asarray(vec).flatten()
            elif tag in self.tag_descriptions:
                vec = self.vectorizer.transform([self.tag_descriptions[tag]])
                self.tag_vectors[tag] = np.asarray(vec).flatten()
            else:
                self.tag_vectors[tag] = np.zeros(self.vectorizer.vocabulary_.__len__())

    def suggest(self, document: str, top_k: int = 5, threshold: float = 0.1) -> Tuple[List[str], List[float]]:
        if not self.is_trained:
            return self._suggest_by_similarity(document, top_k)
        return self._suggest_by_classifier(document, top_k, threshold)

    def _suggest_by_classifier(self, document: str, top_k: int, threshold: float) -> Tuple[List[str], List[float]]:
        X = self.vectorizer.transform([document])
        proba_list = self.classifier.predict_proba(X)
        scores = {}
        for i, tag in enumerate(self.mlb.classes_):
            scores[tag] = proba_list[i][0][1]
        sorted_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result_tags = [t for t, s in sorted_tags if s >= threshold][:top_k]
        result_scores = [s for t, s in sorted_tags if s >= threshold][:top_k]
        return result_tags, result_scores

    def _suggest_by_similarity(self, document: str, top_k: int) -> Tuple[List[str], List[float]]:
        if not self.tag_descriptions:
            return [], []
        doc_vec = self.vectorizer.transform([document])
        scores = {}
        for tag, desc in self.tag_descriptions.items():
            desc_vec = self.vectorizer.transform([desc])
            sim = cosine_similarity(doc_vec, desc_vec)[0][0]
            scores[tag] = sim
        sorted_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [t for t, s in sorted_tags[:top_k]], [s for t, s in sorted_tags[:top_k]]

    def partial_fit(self, documents: List[str], tags_list: List[List[str]]):
        if not self.is_trained:
            self.fit(documents, tags_list)
            return
        self.documents.extend(documents)
        self.document_tags.extend(tags_list)
        self.fit(self.documents, self.document_tags)

    def get_feature_names(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()

    def get_tfidf_vector(self, document: str):
        return self.vectorizer.transform([document])

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'mlb': self.mlb,
                'classifier': self.classifier,
                'is_trained': self.is_trained,
                'tag_descriptions': self.tag_descriptions,
                'documents': self.documents,
                'document_tags': self.document_tags,
            }, f)

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.mlb = data['mlb']
            self.classifier = data['classifier']
            self.is_trained = data['is_trained']
            self.tag_descriptions = data['tag_descriptions']
            self.documents = data['documents']
            self.document_tags = data['document_tags']
            if self.is_trained:
                self._build_tag_tfidf()
