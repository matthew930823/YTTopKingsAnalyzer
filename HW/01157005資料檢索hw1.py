import json
from collections import Counter, defaultdict
from pathlib import Path

import nltk
from nltk.tokenize import word_tokenize


JSON_PATH = Path(__file__).with_name("ReutersCorn-train.json")
NLTK_DATA_DIR = Path(__file__).with_name("nltk_data")

DF_ONLY_WORDS = ["some", "into", "here", "control", "service", "what", "sea"]
DF_TF_WORDS = ["weekend", "heart", "turn", "field", "design", "check", "broken"]

# Reference terms provided by the assignment prompt for correctness checks.
REFERENCE_DF_ONLY_WORDS = ["this", "support", "city"]
REFERENCE_DF_TF_WORDS = ["ships", "hotel", "hotels"]


def load_documents(json_path: Path) -> list[dict]:
	with json_path.open("r", encoding="utf-8") as file:
		return json.load(file)


def ensure_nltk_data() -> None:
	NLTK_DATA_DIR.mkdir(exist_ok=True)
	nltk.data.path = [str(NLTK_DATA_DIR)]

	for package in ("punkt", "punkt_tab"):
		nltk.download(package, download_dir=str(NLTK_DATA_DIR), quiet=True)


def build_inverted_index(docs: list[dict]) -> dict[str, dict[str, int]]:
	inverted_index: dict[str, dict[str, int]] = defaultdict(dict)

	for doc in docs:
		doc_id = doc["docID"]
		text = doc.get("text", "")

		tokens = [token.lower() for token in word_tokenize(text)]
		tf_counter = Counter(tokens)

		for token, tf in tf_counter.items():
			inverted_index[token][doc_id] = tf

	return dict(inverted_index)


def format_df_line(token: str, index: dict[str, dict[str, int]]) -> str:
	postings = index.get(token, {})
	return f"{token}, df={len(postings)}"


def format_df_tf_line(token: str, index: dict[str, dict[str, int]]) -> str:
	postings = index.get(token, {})
	sorted_doc_ids = sorted(postings)
	inv_list = "".join(f"({doc_id},{postings[doc_id]})" for doc_id in sorted_doc_ids)
	return f"{token}, df={len(postings)}, inv_list={inv_list}"


def print_section(title: str, tokens: list[str], index: dict[str, dict[str, int]], with_tf: bool) -> None:
	print(title)
	formatter = format_df_tf_line if with_tf else format_df_line
	for token in tokens:
		print(formatter(token, index))
	print()


def main() -> None:
	ensure_nltk_data()
	documents = load_documents(JSON_PATH)
	index = build_inverted_index(documents)

	print_section("Quiz#1 - 各詞的 df 值", DF_ONLY_WORDS, index, with_tf=False)
	print_section("Quiz#1 - 各詞的 df 值以及在各文件中的 tf 值", DF_TF_WORDS, index, with_tf=True)

	print_section("Reference check - 各詞的 df 值", REFERENCE_DF_ONLY_WORDS, index, with_tf=False)
	print_section(
		"Reference check - 各詞的 df 值以及在各文件中的 tf 值",
		REFERENCE_DF_TF_WORDS,
		index,
		with_tf=True,
	)


if __name__ == "__main__":
	main()
