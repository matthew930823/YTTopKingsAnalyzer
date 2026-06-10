#streamlit run top_kings_dashboard.py --server.headless true
from __future__ import annotations

from pathlib import Path
import html
import re

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


BASE_DIR = Path(__file__).resolve().parent
RANGE_PATTERN = re.compile(r"^(\d{2}-\d{2})-(\d{2}-\d{2})霸榜王$")


if __name__ == "__main__" and get_script_run_ctx() is None:
    print("Please run this app with: streamlit run top_kings_dashboard.py")
    raise SystemExit(0)


st.set_page_config(
    page_title="霸榜王儀表板",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #17181c;
        --muted: #5f6573;
        --bg: #f3efe6;
        --panel: rgba(255, 255, 255, 0.72);
        --panel-border: rgba(23, 24, 28, 0.08);
        --accent: #1d5c96;
        --accent-soft: rgba(29, 92, 150, 0.12);
        --accent-warm: #b96f2b;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(29, 92, 150, 0.12), transparent 35%),
            radial-gradient(circle at top right, rgba(185, 111, 43, 0.10), transparent 28%),
            linear-gradient(180deg, #faf7f0 0%, #f3efe6 100%);
        color: var(--ink);
    }

    .hero {
        padding: 1.4rem 1.5rem;
        border: 1px solid var(--panel-border);
        border-radius: 24px;
        background: var(--panel);
        box-shadow: 0 18px 48px rgba(23, 24, 28, 0.08);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }

    .hero h1, .hero h2, .hero h3 {
        margin-bottom: 0.2rem;
    }

    .hero p {
        color: var(--muted);
        margin-bottom: 0;
    }

    .section-title {
        margin: 1.2rem 0 0.75rem;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--ink);
    }

    .metric-card {
        border: 1px solid var(--panel-border);
        background: rgba(255, 255, 255, 0.80);
        border-radius: 18px;
        padding: 1rem 1rem 0.9rem;
        box-shadow: 0 10px 24px rgba(23, 24, 28, 0.06);
        height: 100%;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-size: 1.2rem;
        font-weight: 750;
        color: var(--ink);
        line-height: 1.2;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.84rem;
        margin-top: 0.35rem;
    }

    .video-card {
        border: 1px solid var(--panel-border);
        background: rgba(255, 255, 255, 0.84);
        border-radius: 22px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 28px rgba(23, 24, 28, 0.05);
        margin-bottom: 1rem;
    }

    .video-rank {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 0.55rem;
    }

    .video-title {
        font-size: 1.18rem;
        font-weight: 800;
        margin: 0.1rem 0 0.35rem;
    }

    .video-meta {
        color: var(--muted);
        font-size: 0.92rem;
        margin-bottom: 0.5rem;
    }

    .highlight-box {
        border-left: 4px solid var(--accent-warm);
        background: rgba(185, 111, 43, 0.08);
        padding: 0.75rem 0.9rem;
        border-radius: 14px;
        margin: 0.8rem 0 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def discover_range_files(base_dir: Path) -> list[tuple[str, Path]]:
    range_files: list[tuple[str, Path]] = []
    for folder_path in base_dir.iterdir():
        if not folder_path.is_dir():
            continue
        match = RANGE_PATTERN.match(folder_path.name)
        if not match:
            continue
        csv_path = folder_path / f"{folder_path.name}.csv"
        if csv_path.exists():
            range_files.append((folder_path.name, csv_path))
    return sorted(range_files, key=lambda item: item[1].name)


def load_table(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    for column in [
        "Rank",
        "View_Count",
        "Like_Count",
        "Comment_Count",
        "Total_Heat_Score",
        "Appearance_Count",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def format_int(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    try:
        return f"{int(float(value)):,}"
    except Exception:
        return str(value)


def format_text(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    text = str(value).strip()
    return text or "N/A"


def format_upload_time(value: object) -> str:
    text = format_text(value)
    if text == "N/A":
        return text
    try:
        return pd.to_datetime(text).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return text


def render_page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric(label: str, value: str, note: str = "") -> None:
    note_html = f'<div class="metric-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_numeric_sum(frame: pd.DataFrame, column_name: str) -> int | None:
    if column_name not in frame.columns:
        return None
    numeric = pd.to_numeric(frame[column_name], errors="coerce").fillna(0)
    return int(numeric.sum())


def safe_numeric_max(frame: pd.DataFrame, column_name: str) -> int | None:
    if column_name not in frame.columns:
        return None
    numeric = pd.to_numeric(frame[column_name], errors="coerce").fillna(0)
    return int(numeric.max())


def render_key_metrics(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("這個資料集沒有內容。")
        return

    top_row = frame.iloc[0]
    total_videos = int(frame.shape[0])
    total_heat = safe_numeric_sum(frame, "Total_Heat_Score")
    max_appearances = safe_numeric_max(frame, "Appearance_Count")
    top_title = format_text(top_row.get("Title", ""))

    metric_columns = st.columns(4)
    with metric_columns[0]:
        render_metric("霸榜王", top_title, f"#1 / {format_text(top_row.get('Channel', ''))}")
    with metric_columns[1]:
        render_metric("影片數量", f"{total_videos}", "這個檢視中的影片數")
    with metric_columns[2]:
        if total_heat is None:
            render_metric("總熱度", "N/A", "每日 Top 50 沒有這個欄位")
        else:
            render_metric("總熱度", f"{total_heat:,}", "Total Heat Score 加總")
    with metric_columns[3]:
        if max_appearances is None:
            render_metric("最高上榜天數", "N/A", "區間霸榜王才會出現這個欄位")
        else:
            render_metric("最高上榜天數", f"{max_appearances}", "Appearance Count 最大值")


def build_summary_block(row: pd.Series) -> str:
    return (
        f"<div class='video-card'>"
        f"<div class='video-rank'>Rank #{format_text(row.get('Rank', ''))}</div>"
        f"<div class='video-title'>{html.escape(format_text(row.get('Title', '')))}</div>"
        f"<div class='video-meta'>"
        f"Channel: {html.escape(format_text(row.get('Channel', '')))} · "
        f"Upload: {html.escape(format_upload_time(row.get('Upload_Time', '')))} · "
        f"Duration: {html.escape(format_text(row.get('Duration', '')))}"
        f"</div>"
        f"<div class='highlight-box'>"
        f"Views {format_int(row.get('View_Count', 0))} · "
        f"Likes {format_int(row.get('Like_Count', 0))} · "
        f"Comments {format_int(row.get('Comment_Count', 0))} · "
        f"Heat {format_int(row.get('Total_Heat_Score', 0))} · "
        f"Appearances {format_int(row.get('Appearance_Count', 0))}"
        f"</div>"
        f"</div>"
    )


def find_video_file(folder: Path, video_id: str, suffix: str) -> Path | None:
    candidate = folder / f"{video_id}{suffix}"
    return candidate if candidate.exists() else None


def render_comments_preview(comments_path: Path | None) -> None:
    st.markdown("<div class='section-title'>留言預覽</div>", unsafe_allow_html=True)
    if not comments_path:
        st.caption("這支影片目前沒有留言 CSV。")
        return

    comments_frame = pd.read_csv(comments_path)
    if comments_frame.empty:
        st.caption("留言檔案是空的。")
        return

    preview_frame = comments_frame.head(12).copy()
    if "Like_Count" in preview_frame.columns:
        preview_frame["Like_Count"] = pd.to_numeric(preview_frame["Like_Count"], errors="coerce")
        preview_frame["Like_Count"] = preview_frame["Like_Count"].fillna(0).astype(int)

    st.dataframe(
        preview_frame,
        use_container_width=True,
        hide_index=True,
    )


def render_analysis_preview(report_path: Path | None) -> None:
    st.markdown("<div class='section-title'>情緒 / 實體關聯</div>", unsafe_allow_html=True)
    if not report_path:
        st.caption("這支影片目前沒有分析報告。")
        return

    report_frame = pd.read_csv(report_path)
    if report_frame.empty:
        st.caption("分析報告是空的。")
        return

    st.dataframe(
        report_frame.head(20),
        use_container_width=True,
        hide_index=True,
    )


def render_executive_report(report_path: Path | None) -> None:
    st.markdown("<div class='section-title'>執行摘要（白話簡潔版）</div>", unsafe_allow_html=True)
    if not report_path:
        st.caption("目前無 final_executive_report.md 檔案。")
        return

    try:
        text = report_path.read_text(encoding="utf-8")
        # Render Markdown report; allow GitHub-style Markdown rendering
        st.markdown(text)
    except Exception:
        st.caption("無法讀取執行報告檔案。")


def render_video_detail(row: pd.Series, folder: Path) -> None:
    st.markdown(build_summary_block(row), unsafe_allow_html=True)

    detail_columns = st.columns(3)
    with detail_columns[0]:
        st.markdown("**基本資料**")
        st.write(
            {
                "Video_ID": format_text(row.get("Video_ID", "")),
                "Channel": format_text(row.get("Channel", "")),
                "Upload_Time": format_upload_time(row.get("Upload_Time", "")),
                "Duration": format_text(row.get("Duration", "")),
            }
        )
    with detail_columns[1]:
        st.markdown("**流量資料**")
        st.write(
            {
                "View_Count": format_int(row.get("View_Count", 0)),
                "Like_Count": format_int(row.get("Like_Count", 0)),
                "Comment_Count": format_int(row.get("Comment_Count", 0)),
            }
        )
    with detail_columns[2]:
        st.markdown("**霸榜指標**")
        st.write(
            {
                "Total_Heat_Score": format_int(row.get("Total_Heat_Score", 0)),
                "Appearance_Count": format_int(row.get("Appearance_Count", 0)),
            }
        )

    video_id = format_text(row.get("Video_ID", ""))
    comments_path = find_video_file(folder, video_id, "_comments.csv")
    report_path = find_video_file(folder, video_id, "_analysis_report.csv")

    preview_columns = st.columns(2)
    with preview_columns[0]:
        render_comments_preview(comments_path)
        render_executive_report(find_video_file(folder, video_id, "_final_executive_report.md"))
    with preview_columns[1]:
        render_analysis_preview(report_path)
        


def render_range_view(base_dir: Path) -> None:
    range_files = discover_range_files(base_dir)
    if not range_files:
        st.warning("找不到任何霸榜王區間資料夾。先執行 `run_all.py` 產生資料，再回來看儀表板。")
        return

    labels = [label for label, _ in range_files]
    selected_label = st.sidebar.selectbox("選擇霸榜王區間", labels, index=len(labels) - 1)
    selected_path = dict(range_files)[selected_label]
    folder = selected_path.parent
    frame = load_table(selected_path)
    frame = frame.sort_values("Rank", ascending=True).reset_index(drop=True)

    title = f"霸榜王區間：{folder.name}"
    subtitle = "展示時間區間內的持續霸榜王，並附上留言與情緒分析補充。"
    render_page_header(title, subtitle)
    render_key_metrics(frame)

    st.markdown("<div class='section-title'>前五名持續霸榜王</div>", unsafe_allow_html=True)
    for _, row in frame.head(5).iterrows():
        with st.expander(f"#{format_text(row.get('Rank', ''))} {format_text(row.get('Title', ''))}", expanded=int(row.get("Rank", 0)) == 1):
            render_video_detail(row, folder)

    st.markdown("<div class='section-title'>完整名單</div>", unsafe_allow_html=True)
    st.dataframe(frame, use_container_width=True, hide_index=True)


def sidebar_overview(base_dir: Path) -> None:
    range_files = discover_range_files(base_dir)

    st.sidebar.title("霸榜王儀表板")
    st.sidebar.caption("從既有 CSV 直接生成前端介面")
    st.sidebar.markdown("---")
    st.sidebar.metric("區間資料檔", len(range_files))
    st.sidebar.markdown("---")
    st.sidebar.markdown("**使用方式**")
    st.sidebar.write("1. 先選擇要看的時間區間。")
    st.sidebar.write("2. 在側邊欄切換 mm-dd-mm-dd 霸榜王。")
    st.sidebar.write("3. 點開卡片看更完整的影片資訊、留言與分析報告。")


def main() -> None:
    sidebar_overview(BASE_DIR)

    if not discover_range_files(BASE_DIR):
        render_page_header("霸榜王儀表板", "目前還沒有可視化資料。先執行抓取與分析流程。")
        st.info("資料夾中需要有 `mm-dd-mm-dd霸榜王/*.csv`。")
        return

    render_range_view(BASE_DIR)


if __name__ == "__main__":
    main()