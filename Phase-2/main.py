import glob
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

progress_lock = threading.Lock()
PROGRESS_FILE = "progress.json"
RAW_DATA_DIR = "raw_data"
DATA_DIR = "data"
BOARDS = [
    "Baseball",
    "Boy-Girl",
    "c_chat",
    "hatepolitics",
    "Lifeismoney",
    "Military",
    "pc_shopping",
    "stock",
    "Tech_Job",
]


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def get_prev_page_url(soup):
    paging = soup.select("div.btn-group-paging a")
    if len(paging) >= 2:
        prev_btn = paging[1]
        href = prev_btn.get("href")
        if href:
            return "https://www.ptt.cc" + href
    return None


def crawl_board(board, max_articles=200000):
    base_url = f"https://www.ptt.cc/bbs/{board}/index.html"
    cookies = {"over18": "1"}

    session = requests.Session()
    session.cookies.update(cookies)

    current_url = base_url
    all_titles = []

    pbar = tqdm(total=max_articles, desc=board, unit="篇", position=0, leave=True)

    while len(all_titles) < max_articles:
        try:
            response = session.get(current_url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            articles = soup.select("div.r-ent")

            for article in articles:
                title_elem = article.select_one("div.title a")
                if title_elem:
                    all_titles.append(
                        {"title": title_elem.text.strip(), "board": board}
                    )
                    pbar.update(1)

                    if len(all_titles) >= max_articles:
                        break

            current_url = get_prev_page_url(soup)
            time.sleep(0.5)

            if not current_url:
                break

        except requests.RequestException as e:
            print(f"請求失敗: {e}")
            continue

    pbar.close()
    return all_titles


def crawl_all_boards_parallel(boards, max_articles=200000, max_workers=3):
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    progress = load_progress()
    boards_to_crawl = [b for b in boards if progress.get(b) != "completed"]

    if not boards_to_crawl:
        print("所有看板都已完成")
        return

    print(f"待爬取看板: {boards_to_crawl}")

    def crawl_single_board(board):
        with progress_lock:
            progress = load_progress()
            progress[board] = "in_progress"
            save_progress(progress)

        data = crawl_board(board, max_articles=max_articles)
        filename = f"{RAW_DATA_DIR}/raw_{board}.csv"
        save_to_csv(data, filename)

        with progress_lock:
            progress = load_progress()
            progress[board] = "completed"
            save_progress(progress)

        return board, len(data)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(crawl_single_board, board): board
            for board in boards_to_crawl
        }

        for future in as_completed(futures):
            board, count = future.result()
            print(f"{board} 完成，共 {count} 筆")

    print(f"全部看板爬取完成")


def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding="utf-8-sig")


def merge_all_csv(input_dir, output_file):
    csv_files = glob.glob(f"{input_dir}/raw_*.csv")

    if not csv_files:
        print(f"沒有 raw csv")
        return None

    all_dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        all_dfs.append(df)

    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    return merged_df


def clean_data(input_file, output_file):
    df = pd.read_csv(input_file)
    df = df.dropna(subset=["title"])
    df = df[~df["title"].str.lower().str.startswith(("re:", "fw:", "r: ["))]
    df = df[~df["title"].str.lower().str.contains(r"^\[.*?\] ?r:", regex=True)]
    df["title"] = df["title"].str.strip()
    df["title"] = df["title"].str.lower()
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    crawl_all_boards_parallel(BOARDS, max_articles=200000, max_workers=3)
    merge_all_csv(RAW_DATA_DIR, f"{DATA_DIR}/all_boards_raw.csv")
    clean_data(f"{DATA_DIR}/all_boards_raw.csv", f"{DATA_DIR}/cleaned_titles.csv")
