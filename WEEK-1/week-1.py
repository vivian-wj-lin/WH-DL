import urllib.request
import re
import time


def fetch_html(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            html = response.read().decode("utf-8")
        return html
    except Exception as e:
        print(f"{e}")
        return None


def parse_products(html):
    products = {}
    link_blocks = re.split(r'(?=<a class="c-prodInfoV2__link)', html)

    for block in link_blocks:
        if 'href="/prod/' not in block:
            continue

        id_match = re.search(r'href="/prod/([^"]+)"', block)
        if not id_match:
            continue
        product_id = id_match.group(1)

        title_match = re.search(
            r'class="c-prodInfoV2__title"[^>]*>(.*?)</div>', block, re.DOTALL
        )
        title = title_match.group(1).strip() if title_match else ""

        price_match = re.search(
            r"c-prodInfoV2__priceValue[^>]*>\$([0-9,]+)</div>", block
        )
        if price_match:
            price_str = price_match.group(1).replace(",", "")
            price = int(price_str)
        else:
            price = 0

        review_match = re.search(
            r'c-prodInfoV2__text--xs500GrayDark">\((\d+)\)</div>', block
        )
        review_count = int(review_match.group(1)) if review_match else 0

        products[product_id] = {
            "title": title,
            "price": price,
            "review_count": review_count,
        }

    return products


def fetch_precise_rating(product_id):
    url = f"https://24h.pchome.com.tw/prod/{product_id}"

    html = fetch_html(url)
    if html is None:
        return None

    rating_match = re.search(r"c-ratingIcon__textNumber[^>]*>(\d+\.?\d*)</div>", html)
    if rating_match:
        return float(rating_match.group(1))
    return None


def main():
    all_products = {}
    base_url = "https://24h.pchome.com.tw/store/DSAA31"
    page = 1

    while True:
        if page == 1:
            url = base_url
        else:
            url = f"{base_url}?p={page}"

        html = fetch_html(url)
        if html is None:
            break

        products = parse_products(html)
        if len(products) == 0:
            break

        all_products.update(products)
        page += 1
        time.sleep(1)

    with open("products.txt", "w", encoding="utf-8") as f:
        for product_id in all_products.keys():
            f.write(product_id + "\n")

    best_products = []
    products_with_reviews = [
        (pid, info) for pid, info in all_products.items() if info["review_count"] >= 1
    ]
    # print(f" {len(products_with_reviews)} 個有評論的產品")

    for i, (product_id, info) in enumerate(products_with_reviews, 1):
        precise_rating = fetch_precise_rating(product_id)

        if precise_rating is not None:
            if precise_rating > 4.9:
                best_products.append(product_id)
        time.sleep(0.5)

    with open("best-products.txt", "w", encoding="utf-8") as f:
        for product_id in best_products:
            f.write(product_id + "\n")

    i5_products = []
    for product_id, info in all_products.items():
        title = info["title"].lower()
        price = info["price"]

        if "i5" in title and price > 0:
            i5_products.append(
                {"id": product_id, "title": info["title"], "price": price}
            )

    if len(i5_products) > 0:
        total_price = sum(p["price"] for p in i5_products)
        avg_price = total_price / len(i5_products)
        print(f"平均價格：${avg_price:,.2f}")

    # # 檢查特定商品
    # if 'DSAA31-A900IS8SK' in all_products:
    #     info = all_products['DSAA31-A900IS8SK']
    #     print(f"找到商品")
    #     print(f"  標題: {info['title']}")
    #     print(f"  價格: {info['price']}")
    #     print(f"  評論數: {info['review_count']}")
    # else:
    #     print("找不到商品")

    # # 檢查價格為 0 的商品
    # zero_price_products = [pid for pid, info in all_products.items() if info['price'] == 0]
    # print(f"\n價格為 0 的商品數量：{len(zero_price_products)}")
    # if len(zero_price_products) > 0:
    #     for pid in zero_price_products[:5]:
    #         print(f"  - {pid}")

    valid_products = [
        (pid, info) for pid, info in all_products.items() if info["price"] > 0
    ]

    prices = [info["price"] for pid, info in valid_products]

    n = len(prices)
    mean_price = sum(prices) / n  # 平均值 μ

    # 標準差：σ = sqrt(Σ(x - μ)² / N)
    variance = sum((price - mean_price) ** 2 for price in prices) / n
    std_dev = variance**0.5  # 開根號

    with open("standardization.csv", "w", encoding="utf-8") as f:
        f.write("ProductID,Price,PriceZScore\n")

        for product_id, info in valid_products:
            price = info["price"]

            # Z-score = (x - μ) / σ
            if std_dev > 0:
                z_score = (price - mean_price) / std_dev
            else:
                z_score = 0.0

            f.write(f"{product_id},{price},{z_score:.6f}\n")


if __name__ == "__main__":
    main()
