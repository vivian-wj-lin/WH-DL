import json
import os
import re
from typing import List, Dict, Tuple

import jieba.posseg as pseg
import ollama
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm


TRAFFIC_LAW_FILE = "traffic-law.json"
MODEL_DIR = "rag_models"
DOC2VEC_MODEL = f"{MODEL_DIR}/traffic_law_doc2vec.model"
ARTICLES_JSON = f"{MODEL_DIR}/articles.json"
STOPWORDS = {"嗎", "呀", "啦", "哩", "囉", "嘍", "喔", "吗", "呢", "啊", "哦", "吧"}


def load_traffic_law_data(json_file: str) -> List[Dict]:
    with open(json_file, "r", encoding="utf-8") as f:
        law_data = json.load(f)

    articles = []
    for article in law_data["LawArticles"]:
        if article["ArticleType"] == "A":
            article_no = article["ArticleNo"]
            content = article["ArticleContent"]

            content_cleaned = content.replace("\r\n", " ").replace("\n", " ")

            articles.append(
                {
                    "article_no": article_no,
                    "content": content_cleaned,
                    "full_text": f"{article_no} {content_cleaned}",
                }
            )

    return articles


def save_articles(articles: List[Dict], output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)


def load_articles(json_file: str) -> List[Dict]:
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


def init_jieba():
    return pseg


def tokenize_text(text: str, pseg) -> List[str]:
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


def tokenize_articles(articles: List[Dict]) -> List[Dict]:
    pseg_obj = init_jieba()

    for article in tqdm(articles, desc="分詞進度"):
        tokens = tokenize_text(article["full_text"], pseg_obj)
        article["tokens"] = tokens

    return articles


def prepare_tagged_documents(articles: List[Dict]) -> List[TaggedDocument]:
    tagged_docs = []
    for idx, article in enumerate(articles):
        if "tokens" in article and len(article["tokens"]) > 0:
            tagged_docs.append(TaggedDocument(words=article["tokens"], tags=[str(idx)]))
    return tagged_docs


def train_doc2vec_model(
    tagged_docs: List[TaggedDocument],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 40,
    dm: int = 1,
    workers: int = 4,
    seed: int = 42,
) -> Doc2Vec:
    print("\n開始訓練 Doc2Vec 模型")
    print(f"參數: vector_size={vector_size}, epochs={epochs}, dm={dm}")

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

    for epoch in tqdm(range(epochs), desc="訓練進度"):
        model.train(tagged_docs, total_examples=model.corpus_count, epochs=1)

    return model


def save_doc2vec_model(model: Doc2Vec, model_path: str):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)


def load_doc2vec_model(model_path: str) -> Doc2Vec:
    model = Doc2Vec.load(model_path)
    return model


def retrieve_relevant_articles(
    question: str, model: Doc2Vec, articles: List[Dict], top_k: int = 3
) -> List[Tuple[Dict, float]]:
    pseg_obj = init_jieba()
    question_tokens = tokenize_text(question, pseg_obj)

    if not question_tokens:
        print("問題分詞後為空")
        return []

    question_vector = model.infer_vector(question_tokens)
    similar_docs = model.dv.most_similar([question_vector], topn=top_k)
    results = []
    for doc_id, similarity in similar_docs:
        idx = int(doc_id)
        if idx < len(articles):
            results.append((articles[idx], similarity))

    return results


def generate_rag_response(
    question: str,
    relevant_articles: List[Tuple[Dict, float]],
    model_name: str = "gemma3:4b",
) -> str:

    context = "\n\n".join(
        [
            f"【{article['article_no']}】\n{article['content']}"
            for article, _ in relevant_articles
        ]
    )

    prompt = f"""請根據以下台灣道路交通管理處罰條例的法規，用繁體中文簡潔地回答問題。

相關法規：
{context}

問題：{question}"""

    try:
        response = ollama.chat(
            model=model_name, messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"調用 Ollama 失敗: {e}"


def rag_query(
    question: str,
    doc2vec_model: Doc2Vec,
    articles: List[Dict],
    top_k: int = 3,
    ollama_model: str = "gemma3:4b",
    show_sources: bool = True,
) -> str:
    relevant_articles = retrieve_relevant_articles(
        question, doc2vec_model, articles, top_k
    )

    if not relevant_articles:
        return "沒有找到相關的法規條文。"

    # if show_sources:
    #     print(f"\n找到 {len(relevant_articles)} 條相關法規：")
    #     for article, similarity in relevant_articles:
    #         print(f"  - {article['article_no']} (相似度: {similarity:.4f})")
    #         print(f"    {article['content'][:100]}...")

    answer = generate_rag_response(question, relevant_articles, ollama_model)

    return answer


def build_rag_system():

    articles = load_traffic_law_data(TRAFFIC_LAW_FILE)
    save_articles(articles, ARTICLES_JSON)

    articles = tokenize_articles(articles)
    tagged_docs = prepare_tagged_documents(articles)
    model = train_doc2vec_model(tagged_docs)

    save_doc2vec_model(model, DOC2VEC_MODEL)
    save_articles(articles, ARTICLES_JSON)

    return model, articles


def test_rag_system(model=None, articles=None):
    if model is None:
        model = load_doc2vec_model(DOC2VEC_MODEL)
    if articles is None:
        articles = load_articles(ARTICLES_JSON)

    test_questions = [
        "闖紅燈會被罰多少錢？",
        "酒駕的處罰是什麼？",
        "機車沒戴安全帽會怎樣？",
    ]

    for question in test_questions:
        answer = rag_query(question, model, articles, top_k=3)
        print(f"\n問題:\n{question}")
        print(f"\n答案:\n{answer}")
        print(f"=" * 20 + "分隔線" + f"=" * 20)


if __name__ == "__main__":
    # model, articles = build_rag_system()
    # test_rag_system(model, articles)

    test_rag_system()
