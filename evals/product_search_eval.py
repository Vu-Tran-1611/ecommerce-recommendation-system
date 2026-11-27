"""Evaluate product search through the chat API.

Run from the project root while the FastAPI server is running:
    python -m evals.product_search_eval
"""
import argparse
import re
import time

import requests

CHAT_URL = "http://127.0.0.1:8000/api/chat"
# Multi-turn tests go through Laravel so the cookie, Redis memory and history are used
SHOP_URL = "http://127.0.0.1:8001"

APPLE_SAMSUNG_PHONES_UNDER_900 = [43, 44, 186, 190, 191, 194, 195, 198, 200]
TOYS = [278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 290, 291, 292, 293]
GAMING = [11, 59, 60, 61, 62, 65, 66, 67, 68, 69, 71, 74, 75, 76, 241, 242, 243, 244, 248, 249, 250, 252, 253, 254, 255]
PHONE_CASES = [49, 50, 51]

TEST_CASES = [
    {
        "id": "1.1 typos",
        "prompt": "I'm looking for a gift for my niece. Do you have the Disny Lotzo Strawbery Bear plush, "
                  "and maybe the Pop Mrt Lotso keychain too? She loves pink stuff.",
        "must_include": [278, 281],
    },
    {
        "id": "1.2 word order",
        "prompt": "Show me tablet ipad pro 13 inch apple and the galaxy tab ultra s8 from samsung",
        "must_include": [205, 210],
    },
    {
        "id": "2.1 meaning",
        "prompt": "My 6-year-old son is obsessed with Toy Story and his birthday is next week. I want something "
                  "soft he can cuddle at night, not a hard figure, and under $40 if possible.",
        "any_of": [278, 279, 280],
        "min_any": 2,
        "must_exclude": [290, 291, 292],
    },
    {
        "id": "2.2 synonyms",
        "prompt": "I need a cheap wristwatch for my teenage daughter for school, something digital and "
                  "water resistant, nothing fancy or expensive.",
        # Cheap digital watches: Casio kids' digital watches or the $39 Xiaomi Y68 smart watch
        "any_of": [149, 155, 10],
        "must_exclude": [126, 128, 268],
    },
    {
        "id": "6.2 chatty",
        "prompt": "Ugh my phone screen just cracked again lol, I think it's time to finally get something new, "
                  "maybe an iPhone or a Galaxy, nothing crazy expensive though, what do you have?",
        "any_of": APPLE_SAMSUNG_PHONES_UNDER_900,
    },
    {
        "id": "7.1 need, not product",
        "prompt": "My 5 year old is bored at home on rainy days, what could keep them busy?",
        "any_of": TOYS,
    },
    {
        "id": "7.2 gift by interest",
        "prompt": "I want a birthday gift for my brother who spends all his free time playing video games.",
        "any_of": GAMING,
    },
    {
        "id": "7.3 described need",
        "prompt": "I keep dropping my phone, I need something to protect it.",
        "any_of": PHONE_CASES,
    },
    {
        "id": "7.4 not sold",
        "prompt": "Do you sell car tires?",
        "expect_empty": True,
    },
]


# Conversations: each turn is sent in order; the last turn is checked by product card names
MULTI_TURN_CASES = [
    {
        "id": "M1 cheapest of shown",
        "turns": ["Show me Casio watches", "Which one is the cheapest?"],
        "first_card": "Casio Junior Sports Kid Watch Red",
    },
    {
        "id": "M2 narrow by gender",
        "turns": ["Do you have hoodies?", "Any for women?"],
        "first_card": "Nike Sportswear Phoenix Fleece Cropped Hoodie",
    },
    {
        "id": "M3 switch brand",
        "turns": ["Show me Samsung tablets", "What about Apple ones?"],
        "cards_contain": "iPad",
    },
]


# Check returned product IDs against a test case
def check(case: dict, products: list[int]) -> list[str]:
    problems = []
    missing = [pid for pid in case.get("must_include", []) if pid not in products]
    if missing:
        problems.append(f"missing {missing}")

    if "any_of" in case:
        found = [pid for pid in case["any_of"] if pid in products]
        if len(found) < case.get("min_any", 1):
            problems.append(f"found {len(found)} of any_of, need {case.get('min_any', 1)}")

    if case.get("expect_empty") and products:
        problems.append(f"expected no products, got {products}")

    unwanted = [pid for pid in case.get("must_exclude", []) if pid in products]
    if unwanted:
        problems.append(f"unwanted {unwanted}")
    return problems


# Send one prompt to the chat API
def ask(prompt: str) -> tuple[list[int], str, float]:
    start = time.monotonic()
    response = requests.post(CHAT_URL, json={"message": prompt, "history": []}, timeout=180)
    response.raise_for_status()
    data = response.json()
    return data.get("products", []), data.get("response", ""), time.monotonic() - start


# Run one conversation through Laravel like a browser and return the last card names
def ask_conversation(turns: list[str]) -> tuple[list[str], str, float]:
    session = requests.Session()
    home = session.get(SHOP_URL + "/", timeout=60).text
    token = re.search(r'name="csrf-token" content="([^"]+)"', home).group(1)
    headers = {"X-CSRF-TOKEN": token, "Accept": "application/json"}

    start = time.monotonic()
    data = {}
    for turn in turns:
        response = session.post(SHOP_URL + "/bot-chat", data={"message": turn}, headers=headers, timeout=180)
        response.raise_for_status()
        data = response.json()
    cards = re.findall(r'alt="([^"]+)"', data.get("products_html", ""))
    return cards, data.get("answer", ""), time.monotonic() - start


# Check the last turn's product cards against a multi-turn case
def check_conversation(case: dict, cards: list[str]) -> list[str]:
    if "first_card" in case and (not cards or cards[0] != case["first_card"]):
        return [f"first card {cards[:1]}, expected {case['first_card']!r}"]
    if "cards_contain" in case and not any(case["cards_contain"] in card for card in cards):
        return [f"no card containing {case['cards_contain']!r}"]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="Runs per test case")
    args = parser.parse_args()

    total_passed = 0
    for case in TEST_CASES:
        passed = 0
        print(f"\n=== {case['id']}")
        for run in range(1, args.runs + 1):
            try:
                products, text, seconds = ask(case["prompt"])
                problems = check(case, products)
            except requests.RequestException as error:
                products, text, seconds, problems = [], "", 0.0, [f"request failed: {error}"]

            passed += not problems
            status = "PASS" if not problems else "FAIL " + "; ".join(problems)
            print(f"  run {run}: {status} | {seconds:.1f}s | products {products}")
            if problems:
                print(f"         reply: {text[:140]!r}")

        total_passed += passed
        print(f"  -> {passed}/{args.runs} passed")

    for case in MULTI_TURN_CASES:
        passed = 0
        print(f"\n=== {case['id']}")
        for run in range(1, args.runs + 1):
            try:
                cards, text, seconds = ask_conversation(case["turns"])
                problems = check_conversation(case, cards)
            except (requests.RequestException, AttributeError) as error:
                cards, text, seconds, problems = [], "", 0.0, [f"request failed: {error}"]

            passed += not problems
            status = "PASS" if not problems else "FAIL " + "; ".join(problems)
            print(f"  run {run}: {status} | {seconds:.1f}s | cards {[card[:28] for card in cards[:3]]}")
            if problems:
                print(f"         reply: {text[:140]!r}")

        total_passed += passed
        print(f"  -> {passed}/{args.runs} passed")

    total_runs = (len(TEST_CASES) + len(MULTI_TURN_CASES)) * args.runs
    print(f"\nTOTAL: {total_passed}/{total_runs} runs passed")


if __name__ == "__main__":
    main()
