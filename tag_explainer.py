import numpy as np
import jieba
import re
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer


class TagExplainer:
    def __init__(self, suggester):
        self.suggester = suggester

    def explain_tag(self, document: str, tag: str, top_k_words: int = 5) -> Dict:
        if not self.suggester.is_trained or tag not in self.suggester.tag_vectors:
            return self._explain_by_description(document, tag, top_k_words)

        doc_vec = self.suggester.vectorizer.transform([document])
        tag_vec = self.suggester.tag_vectors[tag]

        doc_array = doc_vec.toarray()[0]
        feature_names = self.suggester.vectorizer.get_feature_names_out()

        contribution = doc_array * tag_vec

        top_indices = np.argsort(contribution)[::-1][:top_k_words]
        top_words = [(feature_names[i], float(contribution[i])) for i in top_indices if contribution[i] > 0]

        doc_words = set(jieba.lcut(document))
        matched_words = [(w, s) for w, s in top_words if w in doc_words]

        sim_score = float(np.dot(doc_array, tag_vec) / (np.linalg.norm(doc_array) * np.linalg.norm(tag_vec) + 1e-10))

        return {
            "tag": tag,
            "similarity_score": round(sim_score, 4),
            "top_contributing_words": matched_words[:top_k_words],
            "explanation": self._generate_explanation(tag, matched_words[:top_k_words], sim_score)
        }

    def _explain_by_description(self, document: str, tag: str, top_k_words: int) -> Dict:
        if tag not in self.suggester.tag_descriptions:
            return {
                "tag": tag,
                "similarity_score": 0.0,
                "top_contributing_words": [],
                "explanation": f"标签 '{tag}' 暂无描述信息，无法提供解释。"
            }

        description = self.suggester.tag_descriptions[tag]
        doc_words = set(self._tokenize(document))
        desc_words = set(self._tokenize(description))
        common_words = doc_words & desc_words

        common_with_score = [(w, 1.0) for w in list(common_words)[:top_k_words]]

        return {
            "tag": tag,
            "similarity_score": len(common_words) / max(len(desc_words), 1),
            "top_contributing_words": common_with_score,
            "explanation": self._generate_explanation(tag, common_with_score, len(common_words) / max(len(desc_words), 1))
        }

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r'[^\w\s]', ' ', text)
        words = jieba.lcut(text)
        return [w for w in words if len(w) > 1]

    def _generate_explanation(self, tag: str, top_words: List[Tuple[str, float]], score: float) -> str:
        if not top_words:
            return f"标签 '{tag}' 的推荐依据不足。"

        words_str = "、".join([f"'{w}'" for w, s in top_words])

        if score > 0.7:
            strength = "高度相关"
        elif score > 0.4:
            strength = "较为相关"
        else:
            strength = "有一定关联"

        return f"标签 '{tag}' 与文档{strength}，主要匹配关键词：{words_str}。"

    def explain_all(self, document: str, tags: List[str], top_k_words: int = 5) -> List[Dict]:
        return [self.explain_tag(document, tag, top_k_words) for tag in tags]

    def get_document_keywords(self, document: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.suggester.is_trained:
            words = self._tokenize(document)
            word_count = {}
            for w in words:
                word_count[w] = word_count.get(w, 0) + 1
            sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
            return [(w, float(c) / len(words)) for w, c in sorted_words[:top_k]]

        doc_vec = self.suggester.vectorizer.transform([document])
        feature_names = self.suggester.vectorizer.get_feature_names_out()
        doc_array = doc_vec.toarray()[0]

        top_indices = np.argsort(doc_array)[::-1][:top_k]
        return [(feature_names[i], float(doc_array[i])) for i in top_indices if doc_array[i] > 0]
