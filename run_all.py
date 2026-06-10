"""Orchestrator: run TopKings analysis, fetch comments, then run sentiment-entity linking.

Usage: python run_all.py

It prompts for start/end dates in MM-DD (year 2026), locates the Top Kings CSV,
fetches comments for the Top-5 videos, saves each {Video_ID}_comments.csv, then
runs the SentimentEntityLinkingAnalyzer pipeline and writes both
{Video_ID}_analysis_report.csv and {Video_ID}_analysis_detail.csv.
"""

from pathlib import Path
import sys

import TopKingsCommentsFetcher as tkf
import SentimentEntityLinkingAnalyzer as sela


def parse_mmdd_range() -> tuple[str, str]:
    while True:
        try:
            start = input("Enter Start Date (MM-DD): ").strip()
            end = input("Enter End Date (MM-DD): ").strip()
            if len(start) != 5 or start[2] != '-' or len(end) != 5 or end[2] != '-':
                print("Invalid format. Use MM-DD.")
                continue
            return start, end
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            sys.exit(0)


def main() -> None:
    try:
        base_dir = Path.cwd()
        start_mmdd, end_mmdd = parse_mmdd_range()

        # Locate folder and source CSV
        target_dir, source_csv = tkf.locate_source_csv(base_dir, start_mmdd, end_mmdd)
        print(f"Found source CSV: {source_csv}")

        # Read top5
        top5_df = tkf.get_top5_video_ids(source_csv)
        if top5_df.empty:
            print("No top videos found in source CSV.")
            return

        # Prepare YouTube client
        api_key = tkf.get_api_key()
        from googleapiclient.discovery import build

        youtube = build("youtube", "v3", developerKey=api_key)

        # Initialize CKIP models once
        ws_driver, pos_driver = sela.initialize_ckip_models()

        for idx, row in top5_df.iterrows():
            video_id = str(row["Video_ID"])
            print(f"\nProcessing video {idx+1}/{len(top5_df)}: {video_id}")

            try:
                comments = tkf.fetch_comments_for_video(youtube, video_id=video_id, max_comments=500)
                if not comments:
                    print("  ↳ No comments found. Skipping.")
                    continue

                comments_path = tkf.save_comments_csv(target_dir, video_id, comments)
                print(f"  ↳ Saved {len(comments)} comments -> {comments_path.name}")

                # Run sentiment-entity linking on saved comments
                comments_df = sela.load_comments(comments_path)
                texts = comments_df["Comment_Text"].tolist()

                translated = sela.translate_comments(texts)
                ckip_results = sela.ckip_process_comments(translated, ws_driver, pos_driver)
                report_df = sela.build_report(ckip_results)
                detail_df = sela.build_comment_detail_report(comments_df, translated, ckip_results)
                output_path = sela.resolve_output_path(comments_path)
                detail_output_path = sela.resolve_detail_output_path(comments_path)
                report_df.to_csv(output_path, index=False, encoding="utf-8-sig")
                detail_df.to_csv(detail_output_path, index=False, encoding="utf-8-sig")
                print(f"  ↳ Saved analysis -> {output_path.name}")
                print(f"  ↳ Saved detail -> {detail_output_path.name}")

            except Exception as e:
                print(f"  ↳ Error processing {video_id}: {e}")
                continue

        print("\nAll done. Outputs are in:")
        print(f"  - {target_dir}")

    except Exception as exc:
        print(f"Fatal error: {exc}")
        sys.exit(1)


if __name__ == '__main__':
    main()
