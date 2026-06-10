"""
Fetch YouTube comments for Top 5 Persistent Kings videos.

Workflow:
1. Ask user for start/end date in MM-DD (year assumed to be 2026).
2. Locate {start}-{end}霸榜王/{start}-{end}霸榜王.csv.
3. Read Top 5 videos by Total_Heat_Score.
4. Fetch up to 500 comments per video (order='relevance').
5. Save each video's comments to {Video_ID}_comments.csv in the same folder.

API key loading (secure):
- Preferred: environment variable YOUTUBE_API_KEY
- Alternative: config.py with YOUTUBE_API_KEY or API_KEY
"""

from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import os
import sys

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def parse_user_input() -> Tuple[str, str]:
    """Read and validate date input in MM-DD format."""
    while True:
        try:
            print("\n" + "=" * 64)
            print("YouTube Top Kings Comments Fetcher")
            print("=" * 64)
            print("Please enter date range in MM-DD format (year is assumed 2026).")
            print()

            start_mmdd = input("Enter Start Date (MM-DD): ").strip()
            end_mmdd = input("Enter End Date (MM-DD): ").strip()

            if len(start_mmdd) != 5 or start_mmdd[2] != "-":
                print("❌ Invalid Start Date format. Use MM-DD.")
                continue
            if len(end_mmdd) != 5 or end_mmdd[2] != "-":
                print("❌ Invalid End Date format. Use MM-DD.")
                continue

            start_full = f"2026-{start_mmdd}"
            end_full = f"2026-{end_mmdd}"

            try:
                start_date = datetime.strptime(start_full, "%Y-%m-%d")
                end_date = datetime.strptime(end_full, "%Y-%m-%d")
            except ValueError:
                print("❌ Invalid date value. Please ensure month/day are valid.")
                continue

            if start_date > end_date:
                print("❌ Start date must be before or equal to end date.")
                continue

            return start_mmdd, end_mmdd
        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user.")
            sys.exit(0)


def get_api_key() -> str:
    """Load API key from environment variable or config.py."""
    env_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        # Optional local config file (not committed to source control).
        import config  # type: ignore

        key = getattr(config, "YOUTUBE_API_KEY", "") or getattr(config, "API_KEY", "")
        key = str(key).strip()
        if key:
            return key
    except Exception:
        pass

    raise ValueError(
        "Missing API key. Set environment variable YOUTUBE_API_KEY "
        "or create config.py with YOUTUBE_API_KEY = '...'."
    )


def locate_source_csv(base_dir: Path, start_mmdd: str, end_mmdd: str) -> Tuple[Path, Path]:
    """Locate target folder and source Top Kings CSV."""
    folder_name = f"{start_mmdd}-{end_mmdd}霸榜王"
    folder_path = base_dir / folder_name

    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Target folder not found: {folder_path}")

    csv_path = folder_path / f"{folder_name}.csv"
    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f"Source CSV not found: {csv_path}")

    return folder_path, csv_path


def get_top5_video_ids(source_csv: Path) -> pd.DataFrame:
    """Read source CSV and select Top 5 by Total_Heat_Score."""
    df = pd.read_csv(source_csv)

    required_cols = {"Video_ID", "Total_Heat_Score"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Source CSV missing required columns: {sorted(missing)}")

    top5 = (
        df.sort_values("Total_Heat_Score", ascending=False)
        .head(5)
        .copy()
    )

    top5["Video_ID"] = top5["Video_ID"].astype(str)
    return top5


def fetch_comments_for_video(youtube, video_id: str, max_comments: int = 500) -> List[Dict[str, object]]:
    """Fetch up to max_comments comments for one video using relevance order."""
    comments: List[Dict[str, object]] = []
    page_token = None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            order="relevance",
            textFormat="plainText",
            pageToken=page_token,
        )

        response = request.execute()
        items = response.get("items", [])

        for item in items:
            snippet = item.get("snippet", {})
            top_level = snippet.get("topLevelComment", {}).get("snippet", {})

            comments.append(
                {
                    "Comment_Author": top_level.get("authorDisplayName", ""),
                    "Comment_Text": top_level.get("textDisplay", ""),
                    "Comment_Date": top_level.get("publishedAt", ""),
                    "Like_Count": int(top_level.get("likeCount", 0) or 0),
                }
            )

            if len(comments) >= max_comments:
                break

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return comments


def save_comments_csv(output_dir: Path, video_id: str, comments: List[Dict[str, object]]) -> Path:
    """Save comments into {Video_ID}_comments.csv in output_dir."""
    out_path = output_dir / f"{video_id}_comments.csv"
    comments_df = pd.DataFrame(comments, columns=["Comment_Author", "Comment_Text", "Comment_Date", "Like_Count"])
    comments_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def main() -> None:
    """Main execution flow."""
    try:
        base_dir = Path.cwd()
        start_mmdd, end_mmdd = parse_user_input()

        target_dir, source_csv = locate_source_csv(base_dir, start_mmdd, end_mmdd)
        top5_df = get_top5_video_ids(source_csv)

        api_key = get_api_key()
        youtube = build("youtube", "v3", developerKey=api_key)

        print(f"\n📂 Source CSV: {source_csv}")
        print(f"🎯 Fetching comments for {len(top5_df)} videos...\n")

        for idx, row in top5_df.iterrows():
            video_id = str(row["Video_ID"])
            rank_info = idx + 1
            print(f"[{rank_info}/5] Video_ID: {video_id}")

            try:
                comments = fetch_comments_for_video(youtube, video_id=video_id, max_comments=500)

                if not comments:
                    # Gracefully skip videos with no comments.
                    print("  ↳ No comments found. Skipped.")
                    continue

                out_path = save_comments_csv(target_dir, video_id, comments)
                print(f"  ↳ Saved {len(comments)} comments -> {out_path.name}")

            except HttpError as http_err:
                print(f"  ↳ API error for {video_id}: {http_err}")
                print("  ↳ Skipped this video and continue.")
                continue
            except Exception as e:
                print(f"  ↳ Unexpected error for {video_id}: {e}")
                print("  ↳ Skipped this video and continue.")
                continue

        print("\n✅ Comments fetching completed.")
        print(f"📁 Output folder: {target_dir}")

    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
