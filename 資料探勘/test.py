import pandas as pd

# CSV 介面網址
url = "https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL?response=open_data"

# 讀取資料
try:
    df = pd.read_csv(url)
    print(df.head()) # 顯示前五筆資料
    
    # 儲存到本地
    df.to_csv("stock_data.csv", index=False, encoding="utf-8-sig")
except Exception as e:
    print(f"抓取失敗: {e}")