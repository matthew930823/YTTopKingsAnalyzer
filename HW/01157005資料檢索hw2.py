import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import nltk
from nltk.tokenize import word_tokenize


JSON_PATH = Path(__file__).with_name("ReutersCorn-train.json")
NLTK_DATA_DIR = Path(__file__).with_name("nltk_data")

QUERY_DOC_IDS = [
    "RTC_TR0159",
    "RTC_TR0197",
    "RTC_TR0346",
    "RTC_TR0371",
    "RTC_TR0781",
]


def load_documents(json_path: Path) -> list[dict]:
	with json_path.open("r", encoding="utf-8") as file:
		return json.load(file)


def ensure_nltk_data() -> None:
	NLTK_DATA_DIR.mkdir(exist_ok=True)
	nltk.data.path = [str(NLTK_DATA_DIR)]

	for package in ("punkt", "punkt_tab"):
		nltk.download(package, download_dir=str(NLTK_DATA_DIR), quiet=True)


def tokenize(text: str) -> list[str]:
	return [token.lower() for token in word_tokenize(text)]


def build_tf_and_df(docs: list[dict]) -> tuple[dict[str, Counter], dict[str, int]]:
	doc_term_freqs: dict[str, Counter] = {}
	document_frequencies: dict[str, int] = defaultdict(int)

	for doc in docs:
		doc_id = doc["docID"]
		tokens = tokenize(doc.get("text", ""))
		term_freqs = Counter(tokens)
		doc_term_freqs[doc_id] = term_freqs

		for token in term_freqs:
			document_frequencies[token] += 1

	return doc_term_freqs, dict(document_frequencies)


def build_tfidf_vectors(
	doc_term_freqs: dict[str, Counter],
	document_frequencies: dict[str, int],
	total_documents: int,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
	doc_vectors: dict[str, dict[str, float]] = {}
	doc_norms: dict[str, float] = {}

	for doc_id, term_freqs in doc_term_freqs.items():
		vector: dict[str, float] = {}
		squared_sum = 0.0

		for token, tf in term_freqs.items():
			df = document_frequencies[token]
			idf = math.log10(total_documents / df)
			term_weight = 1 + math.log10(tf) if tf > 0 else 0
			weight = term_weight * idf
			if weight:
				vector[token] = weight
				squared_sum += weight * weight

		doc_vectors[doc_id] = vector
		doc_norms[doc_id] = math.sqrt(squared_sum)

	return doc_vectors, doc_norms


def cosine_similarity(
	vector_a: dict[str, float],
	norm_a: float,
	vector_b: dict[str, float],
	norm_b: float,
) -> float:
	if not norm_a or not norm_b:
		return 0.0

	if len(vector_a) > len(vector_b):
		vector_a, vector_b = vector_b, vector_a

	dot_product = 0.0
	for token, weight_a in vector_a.items():
		weight_b = vector_b.get(token)
		if weight_b is not None:
			dot_product += weight_a * weight_b

	return dot_product / (norm_a * norm_b)


def top_similar_documents(
	query_doc_id: str,
	doc_vectors: dict[str, dict[str, float]],
	doc_norms: dict[str, float],
	top_k: int = 5,
) -> list[dict[str, float]]:
	query_vector = doc_vectors.get(query_doc_id)
	query_norm = doc_norms.get(query_doc_id, 0.0)
	if query_vector is None:
		raise KeyError(f"Document not found: {query_doc_id}")

	scores: list[dict[str, float]] = []
	for doc_id, vector in doc_vectors.items():
		if doc_id == query_doc_id:
			continue
		score = cosine_similarity(query_vector, query_norm, vector, doc_norms[doc_id])
		if score > 0:
			scores.append({"docid": doc_id, "cosine": score})

	scores.sort(key=lambda item: (-item["cosine"], item["docid"]))
	return scores[:top_k]


def main() -> None:
	ensure_nltk_data()
	documents = load_documents(JSON_PATH)
	doc_term_freqs, document_frequencies = build_tf_and_df(documents)
	doc_vectors, doc_norms = build_tfidf_vectors(doc_term_freqs, document_frequencies, len(documents))

	for query_doc_id in QUERY_DOC_IDS:
		top_docs = top_similar_documents(query_doc_id, doc_vectors, doc_norms)
		print(f"{query_doc_id} 相似前 5 名是：" + "/".join(item["docid"] for item in top_docs))
		print(top_docs)
		print()


if __name__ == "__main__":
	main()