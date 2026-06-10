"""
YouTube Trending Video Analysis - Top 5 Persistent Kings (霸榜王) Analyzer
This script identifies the videos that consistently rank high on YouTube trending lists.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, List
import sys


def parse_user_input() -> Tuple[str, str]:
    """
    Parse user input for start and end dates in MM-DD format.
    Assumes year is 2026.
    
    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format
    """
    while True:
        try:
            print("\n" + "="*60)
            print("YouTube Trending Video Analysis - Top 5 Persistent Kings")
            print("="*60)
            print("Please enter date range in MM-DD format (assuming year 2026)")
            print()
            
            start_input = input("Enter Start Date (MM-DD): ").strip()
            end_input = input("Enter End Date (MM-DD): ").strip()
            
            # Validate format
            if len(start_input) != 5 or start_input[2] != '-':
                print("❌ Invalid format. Please use MM-DD format.")
                continue
            if len(end_input) != 5 or end_input[2] != '-':
                print("❌ Invalid format. Please use MM-DD format.")
                continue
            
            # Convert to full dates
            start_date = f"2026-{start_input}"
            end_date = f"2026-{end_input}"
            
            # Validate dates
            try:
                start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
                end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
                
                if start_datetime > end_datetime:
                    print("❌ Start date must be before or equal to end date.")
                    continue
                
                return start_date, end_date
            except ValueError:
                print("❌ Invalid date. Please ensure MM and DD are valid.")
                continue
                
        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user.")
            sys.exit(0)


def get_date_range_files(start_date: str, end_date: str, directory: Path) -> List[Path]:
    """
    Find all CSV files within the specified date range.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        directory: Directory to search for files
    
    Returns:
        List of file paths that fall within the date range
    """
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
    
    matching_files = []
    
    # Scan for daily_top50_videos_YYYY-MM-DD.csv files
    pattern = "daily_top50_videos_*.csv"
    for file in directory.glob(pattern):
        try:
            # Extract date from filename
            date_str = file.stem.replace("daily_top50_videos_", "")
            file_datetime = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Check if within range
            if start_datetime <= file_datetime <= end_datetime:
                matching_files.append(file)
        except ValueError:
            # Skip files that don't match the naming convention
            continue
    
    return sorted(matching_files)


def calculate_heat_score(rank: int) -> int:
    """
    Calculate Heat Score for a video based on its rank.
    Heat Score = 51 - Rank (Top 10, so Rank 1 = 50, Rank 2 = 49, etc.)
    
    Args:
        rank: The rank of the video (1-10)
    
    Returns:
        Heat score (integer)
    """
    return 51 - rank


def analyze_videos(file_list: List[Path]) -> pd.DataFrame:
    """
    Analyze video data from multiple CSV files and calculate aggregated metrics.
    
    Args:
        file_list: List of CSV file paths to analyze
    
    Returns:
        DataFrame with aggregated video statistics
    """
    all_data = []
    
    print(f"\n📂 Found {len(file_list)} files in the date range")
    print("Processing files...")
    
    for file in file_list:
        try:
            df = pd.read_csv(file)
            
            # Calculate heat score for each row
            df['Heat_Score'] = df['Rank'].apply(calculate_heat_score)
            
            # Add file date for reference
            date_str = file.stem.replace("daily_top50_videos_", "")
            df['Date'] = date_str
            
            all_data.append(df)
            print(f"  ✓ {file.name}")
        except Exception as e:
            print(f"  ⚠ Error reading {file.name}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("❌ No valid CSV files found or could be read in the date range.")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Group by Video_ID and Title, aggregate metrics
    aggregated = combined_df.groupby(['Video_ID', 'Title']).agg({
        'Heat_Score': 'sum',
        'Rank': ['count', 'mean'],
        'Channel': 'first',  # Get the channel name (should be same for same video)
        'Upload_Time': 'first',  # Get the upload time
        'Duration': 'first',  # Get the duration
        'View_Count': 'max',  # Get the maximum view count (most recent/highest)
        'Like_Count': 'max',  # Get the maximum like count
        'Comment_Count': 'max',  # Get the maximum comment count
    }).reset_index()
    
    # Flatten column names
    aggregated.columns = ['Video_ID', 'Title', 'Total_Heat_Score', 'Appearance_Count', 'Avg_Rank', 
                         'Channel', 'Upload_Time', 'Duration', 'View_Count', 'Like_Count', 'Comment_Count']
    
    # Sort by total heat score in descending order
    aggregated = aggregated.sort_values('Total_Heat_Score', ascending=False).reset_index(drop=True)
    
    return aggregated


def create_output(aggregated_df: pd.DataFrame, start_date: str, end_date: str, 
                 current_dir: Path) -> None:
    """
    Create output directory and save the Top 5 videos to a CSV file.
    
    Args:
        aggregated_df: DataFrame with aggregated video data
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        current_dir: Current working directory
    """
    # Extract MM-DD from dates
    start_mmdd = start_date[5:10]  # From YYYY-MM-DD, get MM-DD
    end_mmdd = end_date[5:10]
    
    # Create output directory name
    output_dir_name = f"{start_mmdd}-{end_mmdd}霸榜王"
    output_dir = current_dir / output_dir_name
    
    # Create directory
    output_dir.mkdir(exist_ok=True)
    
    # Get top 5 videos
    top_5 = aggregated_df.head(5).copy()
    
    # Add rank column (1-5)
    top_5.insert(0, 'Rank', range(1, len(top_5) + 1))
    
    # Reorder columns for output
    output_df = top_5[['Rank', 'Video_ID', 'Title', 'Channel', 'Upload_Time', 'Duration', 
                       'View_Count', 'Like_Count', 'Comment_Count', 'Total_Heat_Score', 'Appearance_Count']].copy()
    
    # Format columns
    output_df.loc[:, 'Appearance_Count'] = output_df['Appearance_Count'].astype(int)
    output_df.loc[:, 'Total_Heat_Score'] = output_df['Total_Heat_Score'].astype(int)
    output_df.loc[:, 'View_Count'] = output_df['View_Count'].astype(int)
    output_df.loc[:, 'Like_Count'] = output_df['Like_Count'].astype(int)
    output_df.loc[:, 'Comment_Count'] = output_df['Comment_Count'].astype(int)
    
    # Save to CSV
    output_filename = f"{start_mmdd}-{end_mmdd}霸榜王.csv"
    output_path = output_dir / output_filename
    
    output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # Display results
    print(f"\n✅ Analysis Complete!")
    print(f"\n{'='*80}")
    print(f"📊 Top 5 Persistent Kings ({start_mmdd} to {end_mmdd})")
    print(f"{'='*80}\n")
    print(output_df.to_string(index=False))
    print(f"\n{'='*80}")
    print(f"📁 Output saved to: {output_path}")
    print(f"{'='*80}\n")


def main():
    """Main execution function."""
    try:
        # Get current directory
        current_dir = Path.cwd()
        
        # Get user input
        start_date, end_date = parse_user_input()
        
        # Get files in date range
        files = get_date_range_files(start_date, end_date, current_dir)
        
        if not files:
            print(f"\n❌ No CSV files found between {start_date} and {end_date}.")
            print(f"Available files in current directory:")
            for f in sorted(current_dir.glob("daily_top50_videos_*.csv")):
                print(f"  - {f.name}")
            return
        
        # Analyze videos
        aggregated_data = analyze_videos(files)
        
        # Create output
        create_output(aggregated_data, start_date, end_date, current_dir)
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
