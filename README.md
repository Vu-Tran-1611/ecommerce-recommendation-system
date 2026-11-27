# AI Services for E-Commerce Platform

This repository contains two AI services integrated with a Laravel e-commerce platform:

1. Product Recommendation System
2. Agentic AI Shopping Assistant

Together, these services support personalized product discovery, LLM-powered product search, and RAG-based customer support.

---

## 1. Overview

This repository contains the AI/ML services used by the e-commerce web application. The Product Recommendation System provides content-based and behavior-based product suggestions, while the Agentic AI Shopping Assistant uses an LLM, tool-calling logic, and RAG to help users search for products and answer store-policy questions.

Both services are exposed through FastAPI endpoints and integrated into the Laravel e-commerce platform.

The live demo is available at: https://demo.fashion-shop.uk

---


## 2. Product Recommendation System

The recommendation system supports two main recommendation flows:

- **Content-Based Similar Product Retrieval**

  One model, `similar_products` (`models/similar_products`, `app/services/similar_products_service.py`):

  - Uses TF-IDF vectors (words and word pairs) for product text features.
  - Uses One-Hot Encoding for brand, category, subcategory, and gender, plus the standardized price.
  - Uses cosine similarity with a KNN-based recommender to retrieve similar products.
  - Supports similar-product recommendations across 200+ products.

- **User-Behavior-Based Recommendation**

  The system uses deep learning recommendation models trained on a 10K+ interaction dataset containing clicks, wishlists, carts, and ratings.

  - BERT4Rec: used for cart and purchase history to capture sequential user behavior.
  - ComiRec: used for wishlist and high-rating behavior to capture multiple user interests.
  - Two-Tower: used for click and search behavior to support fast candidate retrieval.

---

## 3. Recommendation Evaluation

The behavior-based recommenders were evaluated using top-k recommendation metrics:

- Recall@K
- NDCG@K
- Hit@10

These metrics were used to compare how well each model retrieved and ranked relevant products for different user-intent signals.

---

## 4. Agentic AI Shopping Assistant

The AI shopping assistant is a **multi-agent supervisor** system built with LangChain. A supervisor agent reads the user's question and delegates each task to the right sub-agent. Each sub-agent has its own tools.

```
User ─▶ POST /api/chat ─▶ Shopping Supervisor
                            ├─▶ Product Search Agent ─▶ search_products ─▶ Laravel /api/ai/products/search
                            └─▶ Store Policy Agent   ─▶ retrieve_store_policies ─▶ Pinecone (policy PDF)
```

The supervisor also gets the last 10 messages of the conversation (sent by Laravel, which stores them in Redis). Sub-agents stay stateless: the supervisor rewrites follow-ups such as "which one is the cheapest?" into standalone tasks ("Casio watches, cheapest first").

- **Product Search Agent**

  Finds products through the Laravel product search API, with filters for keyword, category, brand, price, and product type.

- **Store Policy Agent (Agentic RAG)**

  Answers policy questions from e-commerce policy documents stored in Pinecone, including shipping, returns, warranty, payment, FAQ, customer support, privacy policy, terms, and about-us. Answers come only from the retrieved context, which lowers the risk of hallucination.

### Chatbot folder structure

```
app/chatbot/
├── llm.py                          # Shared chat model
├── utils/
│   └── laravel_api.py              # Calls Laravel AI API with the service key
├── supervisor/
│   ├── shopping_supervisor.py      # Routes tasks to sub-agents (SUB_AGENTS registry)
│   ├── helpers/
│   │   └── collect_product_ids.py  # Reads product IDs from a supervisor run
│   └── tools/                      # One delegate tool per sub-agent: build, invoke, return
│       ├── delegate_to_product_search_agent.py
│       └── delegate_to_store_policy_agent.py
└── agents/                         # One folder per agent
    ├── product_search/
    │   ├── product_search_agent.py     # Returns a message + product IDs (structured output)
    │   ├── helpers/
    │   │   ├── catalog_filters.py      # Live catalog names from Laravel, cached 10 min
    │   │   └── search_result.py        # Structured result, or found products if stopped early
    │   └── tools/
    │       └── search_products.py
    └── store_policy/
        ├── store_policy_agent.py
        ├── store_policy_ingestion.py   # Loads the policy PDF into Pinecone
        └── tools/
            └── retrieve_store_policies.py
```

Each agent owns its folder, and each tool sits in its own file named after the tool.

Load the policy PDF into Pinecone from the project root:

```bash
python -m app.chatbot.agents.store_policy.store_policy_ingestion
```

### Adding a new agent

1. Create `app/chatbot/agents/<name>/` with a `tools/` folder, one file per tool.
2. Add `<name>_agent.py` with a cached `build_<name>_agent()`, following the existing agents.
3. Add `app/chatbot/supervisor/tools/delegate_to_<name>_agent.py`, which invokes the agent and returns its answer.
4. Register that tool in `SUB_AGENTS` in `shopping_supervisor.py`.

---

## 5. Tech Stack

- FastAPI + Uvicorn
- PyTorch
- scikit-learn
- LangChain (langchain, langchain-openai, langchain-pinecone, langchain-community)
- OpenAI GPT (gpt-5.4-mini)
- Pinecone (vector store for RAG)
- LangSmith (agent tracing and observability)
- pandas, numpy, joblib
- Python-dotenv

---

## 6. Dataset / Knowledge Base

- **Product catalog**: 200+ cleaned products with text and categorical features.
- **User interactions**: 10K+ interactions including clicks, wishlist additions, cart additions, and star ratings (R1–R5).
- **Policy knowledge base**: Store policy PDF covering shipping, returns, refunds, warranty, payment methods, order cancellation, privacy policy, terms, FAQ, and customer support — chunked and embedded into Pinecone.

---

## 7. API Endpoints

All endpoints are prefixed with `/api`.

- **POST /api/recommend**

  Content-based similar product retrieval.

  Request body:
  ```json
  {
    "product_id": 1,
    "model_name": "similar_products",
    "top_k": 10
  }
  ```
  - `model_name`: `similar_products`
  - `version`: not used; accepted for backward compatibility

  Response:
  ```json
  {
    "product_id": 1,
    "model_name": "similar_products",
    "recommendations": [12, 45, 78]
  }
  ```

- **POST /api/recommend/recent**

  Behavior-based recommendation from recent user interactions.

  Request body:
  ```json
  {
    "user_id": 42,
    "model_name": "bert4rec",
    "top_k": 10,
    "interactions": [
      {
        "product_id": 10,
        "category_id": 3,
        "interaction_type": "cart_add"
      }
    ]
  }
  ```
  - `model_name`: `bert4rec` | `comirec` | `twotower`
  - `interaction_type`: `click` | `wishlist_add` | `cart_add` | `R5` | `R4` | `R3` | `R2` | `R1`

  Response:
  ```json
  {
    "model_name": "bert4rec",
    "user_id": 42,
    "recommendations": [101, 202, 303]
  }
  ```

- **POST /api/chat**

  AI shopping assistant.

  JSON body: the new `message` and the recent `history` (oldest first). Assistant messages may end with a `[Shown products: ...]` note so follow-ups can refer to them.
  ```json
  {
    "message": "which one is the cheapest?",
    "history": [
      {"role": "user", "content": "Show me Casio watches"},
      {"role": "assistant", "content": "Here are some Casio watches.\n[Shown products: 1. Casio Junior Sports Kid Watch Red $25 (id 140); ...]"}
    ]
  }
  ```
  Older callers can still send only the query parameter `questions` (no history).

  Response:
  ```json
  {
    "response": "Here are some men's watches under $500.",
    "products": [136, 132, 133]
  }
  ```
  - `response`: plain text with simple Markdown, no HTML.
  - `products`: product IDs picked by the Product Search Agent, most relevant first. Laravel loads them and renders the product cards. Only present when products were found, so policy answers return `response` alone.

---

## 8. Local Setup

1. Clone the repository.
2. Create and activate a Python environment.
   ```bash
   conda env create -f environment.yml
   conda activate <env-name>
   ```
   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables (see section 10).
4. Start the FastAPI server.
   ```bash
   uvicorn app.main:app --reload
   ```
5. Make sure the Laravel e-commerce API is running.
6. Test recommendation and chatbot endpoints.

---

## 9. Environment Variables

Create a `.env` file at the project root with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name
LARAVEL_API_URL=http://your-laravel-app-url
AI_SERVICE_KEY=same_value_as_laravel_AI_SERVICE_KEY
LANGCHAIN_API_KEY=your_langsmith_api_key
```

`AI_SERVICE_KEY` must match `AI_SERVICE_KEY` in the Laravel `.env`. With it, Laravel's AI API allows 600 requests per minute (`AI_SERVICE_RATE_LIMIT`) for this service. Without it, the service shares the public limit of 60 per minute.

---

## 10. Where Models Are Applied in the Website

The following describes where each model applies on https://demo.fashion-shop.uk:

- **KNN (Hybrid: TF-IDF + Cosine Similarity)**: Visible on any product detail page. Scroll to the bottom of the page to find the "You May Also Like These Products" section, which shows content-based similar products.

  This is the `similar_products` model (`models/similar_products`: `knn_model.joblib`, `feature_matrix.joblib`, `metadata.joblib`). Retrain it after catalog changes:

  1. Export the products from the shop database to `data/raw_data/products.csv` (same columns, including `gender`).
  2. Run `python -m training.train_similar_products` from the project root. It cleans the data into `data/cleaned_data`, rebuilds the model in `models/similar_products`, and prints how many top-10 neighbours changed.
  3. Restart the server, since models are loaded at startup.

- **ComiRec**: Shown on the homepage or user feed as the "Continue Exploring Your Style" section. Driven by recent clicks and wishlist activity to capture multiple user interests.

- **Two-Tower Model**: Shown as the "Personalized Recommendations" section. Driven by clicks, wishlist additions, cart additions, and ratings to provide personalized suggestions.

- **BERT4Rec**: A sequence-aware model that considers the order of user interactions to predict what the user is likely to engage with next.

- **AI Shopping Assistant**: Accessible via the floating chat icon at the bottom-right corner of every page. Supports product search with filters and answers store-policy questions using RAG.

---

## 11. Screenshots / Demo

- Detailed explanations and screenshots showing each model’s results are available on my LinkedIn profile: https://www.linkedin.com/in/hoang-tran-963b8b416/

---

## 12. Future Improvements

- Add real-time model retraining pipeline triggered by new interaction data.
- Expand chatbot tools to support order tracking and account management.
- Introduce A/B testing framework to compare recommendation models in production.
- Add caching layer (e.g., Redis) for frequent recommendation requests.
- Improve cold-start handling for new users with no interaction history.
