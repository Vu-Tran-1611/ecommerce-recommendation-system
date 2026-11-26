"""Evaluate product search through the chat API.

Run from the project root while the FastAPI server is running:
    python -m evals.product_search_eval
"""
import argparse
import time

import requests

CHAT_URL = "http://127.0.0.1:8000/api/chat"

APPLE_SAMSUNG_PHONES_UNDER_900 = [43, 44, 186, 190, 191, 194, 195, 198, 200]

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
        "any_of": [149, 155],
        "must_exclude": [126, 128, 268],
    },
    {
        "id": "6.2 chatty",
        "prompt": "Ugh my phone screen just cracked again lol, I think it's time to finally get something new, "
                  "maybe an iPhone or a Galaxy, nothing crazy expensive though, what do you have?",
        "any_of": APPLE_SAMSUNG_PHONES_UNDER_900,
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

    unwanted = [pid for pid in case.get("must_exclude", []) if pid in products]
    if unwanted:
        problems.append(f"unwanted {unwanted}")
    return problems


# Send one prompt to the chat API
def ask(prompt: str) -> tuple[list[int], str, float]:
    start = time.monotonic()
    response = requests.post(CHAT_URL, params={"questions": prompt}, timeout=180)
    response.raise_for_status()
    data = response.json()
    return data.get("products", []), data.get("response", ""), time.monotonic() - start


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

    print(f"\nTOTAL: {total_passed}/{len(TEST_CASES) * args.runs} runs passed")


if __name__ == "__main__":
    main()
