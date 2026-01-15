import os
import re

import jieba
import jieba.posseg as pseg
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models.doc2vec import Doc2Vec

STOPWORDS = {"嗎", "呀", "啦", "哩", "囉", "嘍", "喔", "吗", "呢", "啊", "哦", "吧"}

DICT_FILE = "./dict/dict.txt"
CUSTOM_DICT_FILE = "./dict/custom_dict.txt"

MODEL_DIR = "model"
DOC2VEC_MODEL_PATH = os.path.join(MODEL_DIR, "doc2vec_model.bin")
CLASSIFIER_MODEL_PATH = os.path.join(MODEL_DIR, "best_classifier.pth")

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

BOARD_DISPLAY_NAMES = {
    "Baseball": "Baseball",
    "Boy-Girl": "Boy-Girl",
    "c_chat": "C_Chat",
    "hatepolitics": "HatePolitics",
    "Lifeismoney": "Lifeismoney",
    "Military": "Military",
    "pc_shopping": "PC_Shopping",
    "stock": "Stock",
    "Tech_Job": "Tech_Job",
}


def init_jieba():
    if os.path.exists(DICT_FILE):
        jieba.set_dictionary(DICT_FILE)
        print(f"Loaded main dictionary: {DICT_FILE}")
    if os.path.exists(CUSTOM_DICT_FILE):
        jieba.load_userdict(CUSTOM_DICT_FILE)
        print(f"Loaded custom dictionary: {CUSTOM_DICT_FILE}")
    return pseg


def preprocess_title(title):
    title = re.sub(r"\[.*?\]", "", title)
    title = title.strip()
    return title


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


class ArticleClassifier(nn.Module):
    def __init__(
        self, input_size=300, hidden1=128, hidden2=64, num_classes=9, dropout=0.2
    ):
        super(ArticleClassifier, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden1)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x


def load_models():
    if not os.path.exists(DOC2VEC_MODEL_PATH):
        raise FileNotFoundError(f"Doc2Vec model not found: {DOC2VEC_MODEL_PATH}")
    if not os.path.exists(CLASSIFIER_MODEL_PATH):
        raise FileNotFoundError(f"Classifier model not found: {CLASSIFIER_MODEL_PATH}")

    print(f"Loading Doc2Vec model: {DOC2VEC_MODEL_PATH}")
    doc2vec_model = Doc2Vec.load(DOC2VEC_MODEL_PATH)
    print(f"Doc2Vec model loaded (vector size: {doc2vec_model.vector_size})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    classifier = ArticleClassifier(
        input_size=doc2vec_model.vector_size,
        hidden1=128,
        hidden2=64,
        num_classes=len(BOARDS),
        dropout=0.2,
    )

    print(f"Loading classifier model: {CLASSIFIER_MODEL_PATH}")
    classifier.load_state_dict(
        torch.load(CLASSIFIER_MODEL_PATH, map_location=device, weights_only=True)
    )
    classifier.to(device)
    classifier.eval()
    print("Classifier model loaded")
    return doc2vec_model, classifier, device


def predict_title(title, doc2vec_model, classifier, device, pseg_module):
    cleaned_title = preprocess_title(title)
    tokens = tokenize_text(cleaned_title, pseg_module)
    vector = doc2vec_model.infer_vector(tokens)
    X_tensor = torch.FloatTensor(vector).unsqueeze(0)
    X_tensor = X_tensor.to(device)
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(X_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_idx = predicted_idx.item()  # tensor → int
        confidence = confidence.item()  # tensor → float

    predicted_label = BOARDS[predicted_idx]
    display_name = BOARD_DISPLAY_NAMES[predicted_label]

    return {
        "label": predicted_label,
        "display_name": display_name,
        "confidence": round(confidence, 4),
    }
