import glob
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import jieba
import jieba.posseg as pseg
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm

progress_lock = threading.Lock()
PROGRESS_FILE = "progress.json"
RAW_DATA_DIR = "raw_data"
DATA_DIR = "data"
MODEL_DIR = "model"
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

DICT_FILE = "dict.txt"
STOPWORDS = {"嗎", "呀", "啦", "哩", "囉", "嘍", "喔", "吗", "呢", "啊", "哦", "吧"}


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


def preprocess_title(title):
    # remove [xxx]
    title = re.sub(r"\[.*?\]", "", title)
    title = title.strip()
    return title


def init_jieba():
    if os.path.exists(DICT_FILE):
        jieba.set_dictionary(DICT_FILE)

    custom_dict = "custom_dict.txt"
    if os.path.exists(custom_dict):
        jieba.load_userdict(custom_dict)

    return pseg


def tokenize_text(text, pseg):
    words_with_flags = pseg.cut(text)
    punctuation_pattern = re.compile(
        r'^[。，、；：！？～…．·「」『』（）《》〈〉【】〔〕""'
        "﹁﹂—／/\\"
        r"?.,;:!@#$%^&*()\[\]{}+=_|<>\-]+$"
    )

    filtered_words = []
    for word, flag in words_with_flags:
        word = word.strip()
        if not word:
            continue
        if flag in {"u", "uj", "ul", "p", "c", "e", "zg", "y"}:
            continue
        if word in STOPWORDS:
            continue
        if punctuation_pattern.match(word):
            continue
        filtered_words.append(word)

    return filtered_words


def tokenize_dataset(input_file, output_file):
    pseg = init_jieba()
    df = pd.read_csv(input_file)
    total_rows = len(df)
    tokens_list = []

    for _, row in tqdm(df.iterrows(), total=total_rows, desc="分詞進度", unit="篇"):
        title = row["title"]
        cleaned_title = preprocess_title(title)
        tokens = tokenize_text(cleaned_title, pseg)
        tokens_str = ",".join(tokens)
        tokens_list.append(tokens_str)

    df["tokens"] = tokens_list
    output_df = df[["board", "title", "tokens"]]
    output_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    return output_df


def load_titles(input_file):
    df = pd.read_csv(input_file)
    df["tokens_list"] = df["tokens"].apply(
        lambda x: x.split(",") if pd.notna(x) and x.strip() else []
    )
    df = df[df["tokens_list"].apply(len) > 0]
    print("Titles Ready")
    return df


def prepare_tagged_documents(df):
    tagged_docs = []
    for idx, row in df.iterrows():
        tag = str(row["doc_id"]) if "doc_id" in df.columns else str(idx)
        tagged_docs.append(TaggedDocument(words=row["tokens_list"], tags=[tag]))
    print("Tagged Documents Ready")
    return tagged_docs


def train_doc2vec_model(
    tagged_docs,
    vector_size=100,
    epochs=20,
    window=5,
    min_count=2,
    dm=0,
    seed=42,
    workers=8,
):
    print("Start Training")

    model = Doc2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        dm=dm,
        seed=seed,
    )

    model.build_vocab(tagged_docs)

    for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
        model.train(tagged_docs, total_examples=model.corpus_count, epochs=1)

    return model


def evaluate_model(model, tagged_docs, test_size=1000, seed=42):
    print("Test Similarity")
    random.seed(seed)

    test_sample = random.sample(tagged_docs, test_size)

    self_similarity_count = 0
    second_self_similarity_count = 0

    for doc in tqdm(test_sample, desc="Evaluating", unit="doc"):
        inferred_vector = model.infer_vector(doc.words)
        similar_docs = model.dv.most_similar([inferred_vector], topn=2)

        # similar_docs：[('tag', similarity_score), ...]
        top1_tag = similar_docs[0][0]
        top2_tags = [similar_docs[0][0], similar_docs[1][0]]

        my_tag = doc.tags[0]
        if top1_tag == my_tag:
            self_similarity_count += 1
        if my_tag in top2_tags:
            second_self_similarity_count += 1

    self_similarity = self_similarity_count / len(test_sample)
    second_self_similarity = second_self_similarity_count / len(test_sample)

    print(f"Self Similarity {self_similarity:.3f}")
    print(f"Second Self Similarity {second_self_similarity:.3f}")

    return self_similarity, second_self_similarity, test_sample


def save_model(model, filename="doc2vec_model.bin"):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(f"{MODEL_DIR}/{filename}")
    print("Model saved")


if __name__ == "__main__":
    # os.makedirs(DATA_DIR, exist_ok=True)
    # crawl_all_boards_parallel(BOARDS, max_articles=200000, max_workers=3)
    # merge_all_csv(RAW_DATA_DIR, f"{DATA_DIR}/all_boards_raw.csv")
    # clean_data(f"{DATA_DIR}/all_boards_raw.csv", f"{DATA_DIR}/cleaned_titles.csv")

    # tokenize_dataset(
    #     input_file=f"{DATA_DIR}/cleaned_titles.csv",
    #     output_file=f"{DATA_DIR}/tokenized_titles.csv",
    # )

    # ===== week 3 =====
    np.random.seed(42)
    df = load_titles(f"{DATA_DIR}/tokenized_titles.csv")
    tagged_docs = prepare_tagged_documents(df)
    model = train_doc2vec_model(
        tagged_docs,
        vector_size=300,
        epochs=100,
        window=8,
        min_count=1,
        dm=0,
        seed=42,
        workers=8,
    )

    self_sim, second_self_sim, test_sample = evaluate_model(
        model, tagged_docs, test_size=1000, seed=42
    )

    if second_self_sim >= 0.80:
        save_model(model, "doc2vec_model.bin")
        print(f"Model passed. Second Self-Similarity = {second_self_sim:.3f}")

        sample_data = []
        for doc in test_sample:
            doc_id = int(doc.tags[0])
            original_row = df.loc[doc_id]
            sample_data.append(
                {
                    "doc_id": doc_id,
                    "board": original_row["board"],
                    "title": original_row["title"],
                    "tokens": original_row["tokens"],
                }
            )

        sample_df = pd.DataFrame(sample_data)
        sample_df.to_csv(
            "./tokenized_titles_sample.csv", index=False, encoding="utf-8-sig"
        )
        print(f"Sample saved: ./tokenized_titles_sample.csv")
    else:
        print(f"Model failed. Second Self-Similarity = {second_self_sim:.3f}")

    # df = load_titles("./tokenized_titles_sample.csv")
    # tagged_docs = prepare_tagged_documents(df)
    # model = Doc2Vec.load(f"{MODEL_DIR}/doc2vec_model.bin")
    # self_sim, second_self_sim, _ = evaluate_model(
    #     model, tagged_docs, test_size=1000, seed=42
    # )
    # print(f"Second Self-Similarity = {second_self_sim:.3f}")
