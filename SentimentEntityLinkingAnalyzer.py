"""YouTube comment sentiment-entity linking analysis.

Workflow:
1. Ask the user for a comments CSV filename.
2. Load Comment_Text with pandas.
3. Detect language with langid and translate non-Chinese text to Traditional Chinese.
4. Segment the unified text with CKIP (albert-base WS/POS).
5. Count entity-sentiment co-occurrences within a 5-token window in each sentence.
6. Save the report as {video_id}_analysis_report.csv next to the input file.
"""

from __future__ import annotations

import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

try:
    import langid
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Missing dependency: langid. Install it before running this script."
    ) from exc

try:
    from deep_translator import GoogleTranslator
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Missing dependency: deep-translator. Install it before running this script."
    ) from exc

try:
    from ckip_transformers.nlp import CkipPosTagger, CkipWordSegmenter
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Missing dependency: ckip-transformers. Install it before running this script."
    ) from exc


SENTIMENT_POS_TAGS = {"VH", "VHC", "VK"}
ENTITY_POS_TAGS = {"Na", "Nb"}
TRANSLATION_TARGET = "zh-TW"
CKIP_MODEL_NAME = "albert-base"
WINDOW_SIZE = 5
SIMPLIFIED_HINT_CHARS = set("汉话车门风云发后为见万与东乌乐乔体网国里广台这来从会汉龙")


def prompt_for_csv_path() -> Path:
    """Prompt the user for a CSV filename and resolve it on disk."""
    while True:
        user_input = input("Enter comments CSV filename (e.g. BMBhxRmyQ-o_comments.csv): ").strip().strip('"')
        if not user_input:
            print("Please enter a filename.")
            continue

        candidate = Path(user_input)
        if candidate.is_file():
            return candidate.resolve()

        cwd = Path.cwd()
        matches = list(cwd.rglob(user_input))
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            print("Multiple files matched that name. Please enter a more specific relative path.")
            for match in matches[:10]:
                print(f"  - {match}")
            continue

        print(f"File not found: {user_input}")


def load_comments(csv_path: Path) -> pd.DataFrame:
    """Load comment text from the input CSV."""
    df = pd.read_csv(csv_path)
    if "Comment_Text" not in df.columns:
        raise ValueError("Input CSV must contain a Comment_Text column.")

    comments = (
        df["Comment_Text"]
        .fillna("")
        .astype(str)
        .map(str.strip)
    )
    comments = comments[comments.ne("")].reset_index(drop=True)

    if comments.empty:
        raise ValueError("No non-empty comments found in Comment_Text.")

    return pd.DataFrame({"Comment_Text": comments})


def detect_language(text: str) -> Tuple[str, float]:
    """Return the langid language code and confidence."""
    if not text.strip():
        return "unknown", 0.0
    return langid.classify(text)


def looks_traditional_chinese(text: str) -> bool:
    """Best-effort check for whether a Chinese string is already Traditional Chinese."""
    if not text.strip():
        return False

    if not re.search(r"[\u4e00-\u9fff]", text):
        return False

    return not any(char in SIMPLIFIED_HINT_CHARS for char in text)


def translate_to_traditional_chinese(text: str) -> str:
    """Translate text to Traditional Chinese using GoogleTranslator when needed."""
    if not text.strip():
        return ""

    language, _confidence = detect_language(text)
    if language.lower().startswith("zh") and looks_traditional_chinese(text):
        return text

    translator = GoogleTranslator(source="auto", target=TRANSLATION_TARGET)
    translated = translator.translate(text)
    time.sleep(0.5)
    return translated if translated else text


def translate_comments(texts: Sequence[str]) -> List[str]:
    """Translate each comment as needed and show progress."""
    translated_texts: List[str] = []
    for text in tqdm(texts, desc="Translating", unit="comment"):
        try:
            translated_texts.append(translate_to_traditional_chinese(text))
        except Exception:
            translated_texts.append(text)
    return translated_texts


def split_sentences(text: str) -> List[str]:
    """Split text into sentence-like chunks for sentence-local co-occurrence counting."""
    normalized = re.sub(r"[\r\t]+", " ", text).strip()
    if not normalized:
        return []

    pieces = re.split(r"(?<=[。！？!?；;\n])", normalized)
    sentences = [piece.strip(" \n。！？!?；;") for piece in pieces]
    return [sentence for sentence in sentences if sentence]


def initialize_ckip_models() -> Tuple[CkipWordSegmenter, CkipPosTagger]:
    """Create CKIP WS/POS models using the ALBERT base checkpoint."""
    ws_driver = CkipWordSegmenter(model=CKIP_MODEL_NAME, device=-1)
    pos_driver = CkipPosTagger(model=CKIP_MODEL_NAME, device=-1)
    return ws_driver, pos_driver


def ckip_process_comments(
    texts: Sequence[str],
    ws_driver: CkipWordSegmenter,
    pos_driver: CkipPosTagger,
) -> List[List[Tuple[List[str], List[str]]]]:
    """Return per-comment sentence token/POS outputs."""
    all_results: List[List[Tuple[List[str], List[str]]]] = []

    for text in tqdm(texts, desc="CKIP", unit="comment"):
        sentences = split_sentences(text)
        if not sentences:
            all_results.append([])
            continue

        ws_results = ws_driver(sentences)
        pos_results = pos_driver(ws_results)
        all_results.append(list(zip(ws_results, pos_results)))

    return all_results


def extract_unique_pairs_from_sentence(words: Sequence[str], pos_tags: Sequence[str]) -> set[Tuple[str, str]]:
    """Collect unique entity-sentiment pairs from one sentence."""
    entity_tokens = [(idx, word) for idx, (word, tag) in enumerate(zip(words, pos_tags)) if tag in ENTITY_POS_TAGS]
    sentiment_tokens = [
        (idx, word)
        for idx, (word, tag) in enumerate(zip(words, pos_tags))
        if tag in SENTIMENT_POS_TAGS
    ]

    pairs: set[Tuple[str, str]] = set()
    for entity_idx, entity_word in entity_tokens:
        for sentiment_idx, sentiment_word in sentiment_tokens:
            if abs(entity_idx - sentiment_idx) <= WINDOW_SIZE:
                pairs.add((entity_word, sentiment_word))

    return pairs


def infer_reason(entity: str, sentiment_word: str) -> str:
    """Generate a simple explanation for the linked pair."""
    return (
        f"'{entity}' is a noun entity and '{sentiment_word}' is a sentiment-bearing word; "
        f"they appear within a {WINDOW_SIZE}-token window in the same sentence."
    )


def build_report(ckip_results: Sequence[Sequence[Tuple[Sequence[str], Sequence[str]]]]) -> pd.DataFrame:
    """Aggregate co-occurrence counts across all comments."""
    pair_counter: Counter[Tuple[str, str]] = Counter()

    for comment_result in ckip_results:
        for words, pos_tags in comment_result:
            unique_pairs = extract_unique_pairs_from_sentence(words, pos_tags)
            pair_counter.update(unique_pairs)

    rows = [
        {
            "Entity": entity,
            "Sentiment_Word": sentiment_word,
            "Co_occurrence_Count": count,
            "Inferred_Reason": infer_reason(entity, sentiment_word),
        }
        for (entity, sentiment_word), count in sorted(
            pair_counter.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )
    ]

    return pd.DataFrame(rows, columns=["Entity", "Sentiment_Word", "Co_occurrence_Count", "Inferred_Reason"])


def build_comment_detail_report(
    comments_df: pd.DataFrame,
    translated_comments: Sequence[str],
    ckip_results: Sequence[Sequence[Tuple[Sequence[str], Sequence[str]]]],
) -> pd.DataFrame:
    """Create a per-comment detail table with matched pairs and matched sentences."""
    detail_rows: list[dict[str, object]] = []

    for row_index, original_row in comments_df.iterrows():
        comment_result = ckip_results[row_index] if row_index < len(ckip_results) else []
        analysis_text = translated_comments[row_index] if row_index < len(translated_comments) else str(original_row["Comment_Text"])

        comment_pairs: set[Tuple[str, str]] = set()
        matched_sentences: list[str] = []

        for words, pos_tags in comment_result:
            sentence_pairs = extract_unique_pairs_from_sentence(words, pos_tags)
            if not sentence_pairs:
                continue

            comment_pairs.update(sentence_pairs)
            matched_sentences.append(" ".join(words))

        detail_row = original_row.to_dict()
        detail_row["Source_Row_Index"] = row_index
        detail_row["CKIP_Analysis_Text"] = analysis_text
        detail_row["CKIP_Matched_Pairs"] = "; ".join(sorted(f"{entity}|{sentiment_word}" for entity, sentiment_word in comment_pairs))
        detail_row["CKIP_Matched_Pair_Count"] = len(comment_pairs)
        detail_row["CKIP_Matched_Sentences"] = " || ".join(matched_sentences)
        detail_rows.append(detail_row)

    return pd.DataFrame(detail_rows)


def resolve_output_path(input_csv: Path) -> Path:
    """Create the output file path from the video id embedded in the filename."""
    stem = input_csv.stem
    video_id = stem[:-9] if stem.endswith("_comments") else stem
    return input_csv.with_name(f"{video_id}_analysis_report.csv")


def resolve_detail_output_path(input_csv: Path) -> Path:
    """Create the comment-level detail output path from the video id embedded in the filename."""
    stem = input_csv.stem
    video_id = stem[:-9] if stem.endswith("_comments") else stem
    return input_csv.with_name(f"{video_id}_analysis_detail.csv")


def main() -> None:
    """Run the end-to-end sentiment-entity linking analysis."""
    try:
        csv_path = prompt_for_csv_path()
        comments_df = load_comments(csv_path)

        print(f"Loaded {len(comments_df)} comments from {csv_path.name}")
        translated_comments = translate_comments(comments_df["Comment_Text"].tolist())

        ws_driver, pos_driver = initialize_ckip_models()
        ckip_results = ckip_process_comments(translated_comments, ws_driver, pos_driver)

        report_df = build_report(ckip_results)
        detail_df = build_comment_detail_report(comments_df, translated_comments, ckip_results)
        output_path = resolve_output_path(csv_path)
        detail_output_path = resolve_detail_output_path(csv_path)
        report_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        detail_df.to_csv(detail_output_path, index=False, encoding="utf-8-sig")

        print(f"Saved {len(report_df)} linked pairs to {output_path}")
        print(f"Saved {len(detail_df)} comment-level rows to {detail_output_path}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\nAnalysis failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()