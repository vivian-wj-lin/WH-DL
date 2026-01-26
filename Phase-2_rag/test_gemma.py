import ollama

test_questions = [
    "闖紅燈會被罰多少錢？",
    "酒駕的處罰是什麼？",
    "機車沒戴安全帽會怎樣？",
]

for question in test_questions:
    response = ollama.chat(
        model="gemma3:4b",
        messages=[
            {
                "role": "user",
                "content": f"請用繁體中文，簡潔地回答台灣交通法規的問題：{question}",
            }
        ],
    )

    print(f"\n問題:\n{question}")
    print(f"\n答案:\n{response['message']['content']}")
    print(f"=" * 20 + "分隔線" + f"=" * 20)
