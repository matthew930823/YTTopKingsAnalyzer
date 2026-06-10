"""Two-stage YouTube comment insights pipeline.

Stage 1: CKIP (ALBERT base) filter
- Load comments from a CSV file.
- Segment with CKIP Word Segmentation and POS tagging.
- Keep sentence-level entity/sentiment co-occurrences where an entity (Na, Nb)
  and a sentiment word (VH, VHC, VK) appear within a 5-token window.
- Rank the top co-occurrence pairs and keep original comments that contain those
  pairs.

Stage 2: Google Gemini sentiment analysis
- Send only the Stage 1 comments, capped at top 50 by relevance.
- Ask Gemini 3.1 Flash Lite to classify the true emotion and return a JSON object.
- Merge the Gemini result back into the extracted comments.

Stage 3: Gemini meta-analysis report
- Aggregate the final_insights output into emotion/reason statistics.
- Ask Gemini 3.5 Flash to write an executive report in Markdown.
- Save the report as {video_id}_final_executive_report.md.

The final output is saved as {video_id}_final_insights.csv next to the input CSV.
"""

from __future__ import annotations

import argparse
import json
import importlib
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm


SENTIMENT_POS_TAGS = {"VH", "VHC", "VK"}
ENTITY_POS_TAGS = {"Na", "Nb"}
WINDOW_SIZE = 5
DEFAULT_TOP_PAIR_LIMIT = 20
DEFAULT_STAGE2_LIMIT = 50
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
DEFAULT_META_GEMINI_MODEL = "gemini-3.1-flash-lite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CKIP + Gemini insights from a YouTube comments CSV.")
    parser.add_argument("csv_path", nargs="?", help="Path to a comments CSV file containing Comment_Text.")
    parser.add_argument("--top-pairs", type=int, default=DEFAULT_TOP_PAIR_LIMIT, help="How many CKIP pairs to keep.")
    parser.add_argument("--limit", type=int, default=DEFAULT_STAGE2_LIMIT, help="Max comments sent to Gemini.")
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL, help="Gemini model name.")
    parser.add_argument("--meta-model", default=DEFAULT_META_GEMINI_MODEL, help="Gemini model name for the executive report.")
    return parser.parse_args()


def prompt_for_csv_path() -> Path:
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


def resolve_input_path(args: argparse.Namespace) -> Path:
    if args.csv_path:
        candidate = Path(args.csv_path)
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"Input CSV not found: {candidate}")
    return prompt_for_csv_path()


def extract_video_id(csv_path: Path) -> str:
    stem = csv_path.stem
    return stem[:-9] if stem.endswith("_comments") else stem


def resolve_output_path(input_csv: Path) -> Path:
    video_id = extract_video_id(input_csv)
    return input_csv.with_name(f"{video_id}_final_insights.csv")


def resolve_meta_report_path(input_csv: Path) -> Path:
    video_id = extract_video_id(input_csv)
    return input_csv.with_name(f"{video_id}_final_executive_report.md")


def resolve_analysis_report_path(input_csv: Path) -> Path:
    video_id = extract_video_id(input_csv)
    return input_csv.with_name(f"{video_id}_analysis_report.csv")


def resolve_analysis_detail_path(input_csv: Path) -> Path:
    video_id = extract_video_id(input_csv)
    return input_csv.with_name(f"{video_id}_analysis_detail.csv")


def extract_pairs_from_sentence(words: Iterable[str], pos_tags: Iterable[str]) -> set[tuple[str, str]]:
    words_list = list(words)
    pos_list = list(pos_tags)

    entity_tokens = [
        (idx, word)
        for idx, (word, tag) in enumerate(zip(words_list, pos_list))
        if tag in ENTITY_POS_TAGS
    ]
    sentiment_tokens = [
        (idx, word)
        for idx, (word, tag) in enumerate(zip(words_list, pos_list))
        if tag in SENTIMENT_POS_TAGS
    ]

    pairs: set[tuple[str, str]] = set()
    for entity_idx, entity_word in entity_tokens:
        for sentiment_idx, sentiment_word in sentiment_tokens:
            if abs(entity_idx - sentiment_idx) <= WINDOW_SIZE:
                pairs.add((entity_word, sentiment_word))

    return pairs


def build_pair_label(pair: tuple[str, str]) -> str:
    return f"{pair[0]}|{pair[1]}"


def load_analysis_artifacts(input_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    report_path = resolve_analysis_report_path(input_csv)
    detail_path = resolve_analysis_detail_path(input_csv)

    if not report_path.is_file():
        raise FileNotFoundError(f"Missing analyzer pair report: {report_path}")
    if not detail_path.is_file():
        raise FileNotFoundError(f"Missing analyzer comment-level detail file: {detail_path}")

    pair_df = pd.read_csv(report_path)
    detail_df = pd.read_csv(detail_path)
    if detail_df.empty:
        raise ValueError(f"Comment-level detail file is empty: {detail_path}")

    required_columns = {"Comment_Text", "CKIP_Matched_Pairs"}
    missing = required_columns.difference(detail_df.columns)
    if missing:
        raise ValueError(f"Comment-level detail file is missing columns: {sorted(missing)}")

    return pair_df, detail_df


def select_top_pairs(pair_df: pd.DataFrame, top_n: int) -> list[tuple[str, str]]:
    if pair_df.empty:
        return []

    top_rows = pair_df.head(top_n)
    return [(str(row["Entity"]), str(row["Sentiment_Word"])) for _, row in top_rows.iterrows()]


def filter_comments_by_top_pairs(analysis_df: pd.DataFrame, top_pairs: list[tuple[str, str]]) -> pd.DataFrame:
    if analysis_df.empty or not top_pairs:
        return analysis_df.iloc[0:0].copy()

    top_pair_labels = {build_pair_label(pair) for pair in top_pairs}
    filtered = analysis_df[analysis_df["CKIP_Matched_Pairs"].fillna("").map(
        lambda value: any(label in value.split("; ") for label in top_pair_labels)
    )].copy()

    if filtered.empty:
        return filtered

    filtered["Stage1_Relevance_Score"] = filtered["CKIP_Matched_Pairs"].fillna("").map(
        lambda value: sum(1 for label in value.split("; ") if label and label in top_pair_labels)
    )

    sort_columns = ["Stage1_Relevance_Score"]
    ascending = [False]
    if "Like_Count" in filtered.columns:
        sort_columns.append("Like_Count")
        ascending.append(False)
    else:
        sort_columns.append("Source_Row_Index")
        ascending.append(True)

    return filtered.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)


def get_gemini_api_key() -> str:
    env_candidates = [
        os.getenv("GEMINI_API_KEY", ""),
        os.getenv("GOOGLE_API_KEY", ""),
        os.getenv("GOOGLE_GENAI_API_KEY", ""),
    ]
    for candidate in env_candidates:
        key = str(candidate).strip()
        if key:
            return key

    try:
        import config  # type: ignore

        for attr in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY", "YOUTUBE_API_KEY", "API_KEY"):
            key = str(getattr(config, attr, "")).strip()
            if key:
                return key
    except Exception:
        pass

    raise ValueError(
        "Missing Gemini API key. Set GEMINI_API_KEY, GOOGLE_API_KEY, or GOOGLE_GENAI_API_KEY, or add it to config.py."
    )


def init_gemini_client(api_key: str):
    try:
        genai = importlib.import_module("google.genai")
        types = importlib.import_module("google.genai.types")

        return {
            "mode": "new",
            "client": genai.Client(api_key=api_key),
            "types": types,
        }
    except ImportError:
        pass

    try:
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        return {"mode": "legacy", "client": genai, "types": None}
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "Missing Gemini SDK. Install google-genai (preferred) or google-generativeai."
        ) from exc


def build_gemini_prompt(comment_text: str) -> str:
    return (
        "請分析這則 YouTube 留言的真實情感（注意辨識反諷或迷因）。請用一個 JSON 格式回傳兩個欄位："
        "'emotion': [興奮/感動/憤怒/反諷/期待/中立], 'reason': (10字以內繁體中文原因)。\n\n"
        f"留言內容：\n{comment_text.strip()}"
    )


def build_batch_gemini_prompt(batch_rows: list[dict[str, Any]]) -> str:
    items_text: list[str] = []
    for item in batch_rows:
        items_text.append(f"[{item['item_index']}] {item['Comment_Text']}")

    joined_items = "\n".join(items_text)
    return (
        "請分析以下多則 YouTube 留言的真實情感（注意辨識反諷或迷因）。"
        "請只回傳 JSON 陣列，不要加任何說明文字。每個元素都要包含三個欄位："
        "'index': 留言編號, 'emotion': [興奮/感動/憤怒/反諷/期待/中立], 'reason': (10字以內繁體中文原因)。\n\n"
        f"留言清單：\n{joined_items}"
    )


def build_meta_data_summary(final_df: pd.DataFrame, top_reason_limit: int = 10) -> str:
    if final_df.empty:
        return "情緒分布：\n- 無資料\n\n常見原因：\n- 無資料"

    emotion_counts = final_df["Gemini_Emotion"].fillna("未知").astype(str).value_counts()
    reason_counts = final_df["Gemini_Reason"].fillna("未知").astype(str).value_counts()

    emotion_lines = ["情緒分布："]
    for emotion, count in emotion_counts.items():
        emotion_lines.append(f"- {emotion}: {int(count)}")

    reason_lines = ["常見原因："]
    for reason, count in reason_counts.head(top_reason_limit).items():
        reason_lines.append(f"- {reason}: {int(count)}")

    return "\n".join(emotion_lines + [""] + reason_lines)


def build_meta_analysis_prompt(data_summary: str) -> str:
    return (
        "你現在是一位懂大眾語言的 YouTube 內容分析師。請把下面的統計資料寫成「白話簡潔版」報告，讓一般人一看就懂。\n"
        "[數據摘要]\n"
        f"{data_summary}\n\n"
        "請直接輸出 Markdown，並且只保留以下四段，內容要短、白話、好懂：\n"
        "1. 【一句話總結】：用 1 到 2 句話講完這支影片為什麼會紅。\n"
        "2. 【為什麼會紅】：列出 3 個最重要的原因，每點 1 到 2 句。\n"
        "3. 【大家為什麼一直看】：用白話說明哪些小原因會讓觀眾一直重看、一直分享。\n"
        "4. 【會不會繼續熱】：用一句話預測它是「長效型」還是「爆發型」。\n\n"
        "規則：\n"
        "- 全部使用繁體中文。\n"
        "- 少用專業術語、少用比喻、少寫空話。\n"
        "- 每段盡量 2 到 4 句。\n"
        "- 直接講結論，不要先鋪陳太多。"
    )


def generate_text_with_gemini(prompt: str, model_name: str) -> str:
    api_key = get_gemini_api_key()
    gemini = init_gemini_client(api_key)
    legacy_model = gemini["client"].GenerativeModel(model_name) if gemini["mode"] == "legacy" else None

    if gemini["mode"] == "new":
        response = gemini["client"].models.generate_content(
            model=model_name,
            contents=prompt,
            config=gemini["types"].GenerateContentConfig(
                temperature=0,
            ),
        )
        return (getattr(response, "text", "") or "").strip()

    response = legacy_model.generate_content(  # type: ignore[union-attr]
        prompt,
        generation_config={"temperature": 0},
    )
    return (getattr(response, "text", "") or "").strip()


def parse_gemini_json(text: str) -> dict[str, str]:
    cleaned = text.strip()
    if not cleaned:
        return {"emotion": "中立", "reason": "空回應"}

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    if not cleaned.startswith("{"):
        object_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if object_match:
            cleaned = object_match.group(0).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"emotion": "中立", "reason": "解析失敗"}

    emotion = str(parsed.get("emotion", "中立")).strip() or "中立"
    reason = str(parsed.get("reason", "解析失敗")).strip() or "解析失敗"
    return {"emotion": emotion, "reason": reason[:10]}


def parse_gemini_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    if not cleaned:
        return []

    fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    if not cleaned.startswith("["):
        array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if array_match:
            cleaned = array_match.group(0).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "index": item.get("index"),
                "emotion": str(item.get("emotion", "中立")).strip() or "中立",
                "reason": str(item.get("reason", "解析失敗")).strip()[:10] or "解析失敗",
            }
        )
    return normalized


def is_retryable_gemini_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "503" in message or "unavailable" in message or "high demand" in message or "temporarily" in message


def run_gemini_batch(batch_rows: list[dict[str, Any]], model_name: str) -> list[dict[str, Any]]:
    prompt = build_batch_gemini_prompt(batch_rows)
    raw_text = generate_text_with_gemini(prompt, model_name)
    return parse_gemini_json_array(raw_text)


def analyze_batch_with_fallback(
    batch_rows: list[dict[str, Any]],
    model_name: str,
    max_retries: int = 3,
    depth: int = 0,
) -> list[dict[str, Any]]:
    try:
        return run_gemini_batch(batch_rows, model_name)
    except Exception as exc:
        if not is_retryable_gemini_error(exc):
            raise

        if max_retries > 0:
            wait_seconds = 2 ** depth
            print(f"Gemini batch hit 503/UNAVAILABLE; retrying in {wait_seconds}s ({max_retries} retries left).")
            time.sleep(wait_seconds)
            return analyze_batch_with_fallback(batch_rows, model_name, max_retries=max_retries - 1, depth=depth + 1)

        raise


def analyze_with_gemini(filtered_df: pd.DataFrame, model_name: str, stage2_limit: int) -> pd.DataFrame:
    if filtered_df.empty:
        return filtered_df.copy()

    work_df = filtered_df.head(stage2_limit).copy().reset_index(drop=True)
    work_df["Batch_Row_Index"] = work_df.index
    work_df["Gemini_Emotion"] = "中立"
    work_df["Gemini_Reason"] = "API失敗"

    batch_rows = work_df[["Batch_Row_Index", "Comment_Text"]].rename(columns={"Batch_Row_Index": "item_index"}).to_dict(orient="records")
    try:
        parsed_items = analyze_batch_with_fallback(batch_rows, model_name)
        parsed_by_index = {int(item["index"]): item for item in parsed_items if item.get("index") is not None}

        for local_index, row in work_df.iterrows():
            parsed = parsed_by_index.get(int(row["Batch_Row_Index"]))
            if parsed is None:
                continue
            work_df.loc[local_index, "Gemini_Emotion"] = parsed["emotion"]
            work_df.loc[local_index, "Gemini_Reason"] = parsed["reason"]
    except Exception as exc:
        print(f"Gemini batch analysis failed: {exc}")

    work_df["Gemini_Model"] = model_name
    work_df = work_df.drop(columns=["Batch_Row_Index"])
    return work_df


def save_dataframe_with_fallback(df: pd.DataFrame, preferred_path: Path) -> Path:
    try:
        df.to_csv(preferred_path, index=False, encoding="utf-8-sig")
        return preferred_path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = preferred_path.with_name(f"{preferred_path.stem}_{timestamp}{preferred_path.suffix}")
        df.to_csv(fallback_path, index=False, encoding="utf-8-sig")
        print(f"Warning: could not overwrite {preferred_path.name}; saved to {fallback_path.name} instead.")
        return fallback_path


def save_text_with_fallback(text: str, preferred_path: Path) -> Path:
    try:
        preferred_path.write_text(text, encoding="utf-8")
        return preferred_path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = preferred_path.with_name(f"{preferred_path.stem}_{timestamp}{preferred_path.suffix}")
        fallback_path.write_text(text, encoding="utf-8")
        print(f"Warning: could not overwrite {preferred_path.name}; saved to {fallback_path.name} instead.")
        return fallback_path


def generate_executive_report(final_df: pd.DataFrame, model_name: str) -> str:
    data_summary = build_meta_data_summary(final_df)
    prompt = build_meta_analysis_prompt(data_summary)
    report_text = generate_text_with_gemini(prompt, model_name)
    return report_text or "# 霸榜原因深度解讀報告\n\n（Gemini 未返回內容）"


def summarize_stage1(filtered_df: pd.DataFrame, pair_df: pd.DataFrame, top_pairs: list[tuple[str, str]]) -> None:
    print(f"Loaded {len(filtered_df)} CKIP-qualified comments")
    print(f"Found {len(pair_df)} unique entity-sentiment pairs")
    if top_pairs:
        print("Top pairs:")
        for rank, pair in enumerate(top_pairs[:10], start=1):
            count = pair_df[(pair_df["Entity"] == pair[0]) & (pair_df["Sentiment_Word"] == pair[1])]["Co_occurrence_Count"].iloc[0]
            print(f"  {rank:>2}. {pair[0]} | {pair[1]} ({int(count)})")


def main() -> None:
    args = parse_args()

    try:
        input_csv = resolve_input_path(args)
        video_id = extract_video_id(input_csv)

        pair_df, detail_df = load_analysis_artifacts(input_csv)
        print(f"Loaded {len(detail_df)} comment-level rows from analyzer outputs for {input_csv.name}")

        top_pairs = select_top_pairs(pair_df, args.top_pairs)
        filtered_df = filter_comments_by_top_pairs(detail_df, top_pairs)

        summarize_stage1(filtered_df, pair_df, top_pairs)

        if filtered_df.empty:
            output_path = resolve_output_path(input_csv)
            saved_output_path = save_dataframe_with_fallback(filtered_df, output_path)
            print(f"No comments matched the top pairs. Saved empty output to {saved_output_path}")
            return

        gemini_df = analyze_with_gemini(filtered_df, args.model, args.limit)
        output_path = resolve_output_path(input_csv)
        saved_output_path = save_dataframe_with_fallback(gemini_df, output_path)

        final_insights_df = pd.read_csv(saved_output_path)
        report_text = generate_executive_report(final_insights_df, args.meta_model)
        report_path = resolve_meta_report_path(input_csv)
        saved_report_path = save_text_with_fallback(report_text, report_path)

        print(f"Saved final insights for {video_id} -> {saved_output_path}")
        print(f"Saved executive report -> {saved_report_path}")
        print(f"Gemini analyzed {len(gemini_df)} comments (limit={args.limit}).")
        print("\n" + report_text + "\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\nPipeline failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()