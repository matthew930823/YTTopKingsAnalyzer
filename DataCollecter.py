from googleapiclient.discovery import build
import pandas as pd
from datetime import datetime
import re

# 1. 設定你的 API Key
API_KEY = 'AIzaSyDDOND-4Q0seRDzjbLSCDt00lmtFwsWIwM'
youtube = build('youtube', 'v3', developerKey=API_KEY)

# 常見 YouTube 影片分類備援對照
FALLBACK_CATEGORY_MAP = {
    '1': 'Film & Animation',
    '2': 'Autos & Vehicles',
    '10': 'Music',
    '15': 'Pets & Animals',
    '17': 'Sports',
    '18': 'Short Movies',
    '19': 'Travel & Events',
    '20': 'Gaming',
    '21': 'Videoblogging',
    '22': 'People & Blogs',
    '23': 'Comedy',
    '24': 'Entertainment',
    '25': 'News & Politics',
    '26': 'Howto & Style',
    '27': 'Education',
    '28': 'Science & Technology',
    '29': 'Nonprofits & Activism',
    '30': 'Movies',
    '31': 'Anime/Animation',
    '32': 'Action/Adventure',
    '33': 'Classics',
    '34': 'Comedy',
    '35': 'Documentary',
    '36': 'Drama',
    '37': 'Family',
    '38': 'Foreign',
    '39': 'Horror',
    '40': 'Sci-Fi/Fantasy',
    '41': 'Thriller',
    '42': 'Shorts',
    '43': 'Shows',
    '44': 'Trailers'
}


def get_category_mapping(region_code='TW'):
    request = youtube.videoCategories().list(
        part='snippet',
        regionCode=region_code
    )
    response = request.execute()

    category_map = FALLBACK_CATEGORY_MAP.copy()
    for item in response.get('items', []):
        category_map[item['id']] = item['snippet']['title']

    return category_map


def format_duration(iso_duration):
    """將 ISO 8601 duration 格式轉換成易讀文字 (HH:MM:SS)"""
    if not iso_duration:
        return 'Unknown'
    
    # 匹配 ISO 8601 duration 格式 (PT1H30M45S)
    pattern = r'P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, iso_duration)
    
    if not match:
        return iso_duration
    
    days, hours, minutes, seconds = match.groups()
    hours = int(hours or 0)
    minutes = int(minutes or 0)
    seconds = int(seconds or 0)
    
    # 如果有天數，加到小時
    if days:
        hours += int(days) * 24
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def get_trending_videos():
    category_map = get_category_mapping('TW')

    # 2. 呼叫 videos().list 接口
    
    request = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        chart="mostPopular",      # 抓取發燒榜
        regionCode="TW",          # 台灣地區
        maxResults=50            # 只拿前 50 名
    )
    response = request.execute()

    video_list = []
    for item in response['items']:
        category_id = item['snippet'].get('categoryId', 'Unknown')
        published_at = item['snippet'].get('publishedAt', 'Unknown')
        # 如果需要只顯示日期，可以用: published_at.split('T')[0]
        
        # 讚數可能因隱私設定無法取得（返回 N/A）
        like_count = item['statistics'].get('likeCount')
        like_count_display = like_count if like_count else 'N/A'
        
        # 影片長度格式轉換
        duration = item['contentDetails'].get('duration', 'Unknown')
        duration_formatted = format_duration(duration)
        
        video_data = {
            'Rank': len(video_list) + 1,
            'Video_ID': item['id'],
            'Title': item['snippet']['title'],
            'Channel': item['snippet']['channelTitle'],
            'Category': category_map.get(category_id, f'Unknown ({category_id})'),
            'Upload_Time': published_at,
            'Duration': duration_formatted,
            'View_Count': item['statistics'].get('viewCount', 0),
            'Like_Count': like_count_display,
            'Comment_Count': item['statistics'].get('commentCount', 0)
        }
        video_list.append(video_data)
    
    return video_list

# 3. 執行並存成 DataFrame 方便查看或匯出
top_50_videos = get_trending_videos()
df = pd.DataFrame(top_50_videos)

# 匯出成 CSV 檔案
today = datetime.now().strftime('%Y-%m-%d')
filename = f'daily_top50_videos_{today}.csv'
df.to_csv(filename, index=False, encoding='utf-8-sig')
print("今日前 50 名影片抓取成功！")
print(df[['Rank', 'Title', 'Duration', 'Upload_Time', 'Like_Count', 'View_Count']])