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

## 2. System Architecture

Coming soon.

---

## 3. Product Recommendation System

The recommendation system supports two main recommendation flows:

- **Content-Based Similar Product Retrieval**

  - Uses TF-IDF vectors for product text features.
  - Uses One-Hot Encoding for categorical features such as category, brand, and product type.
  - Uses cosine similarity with a KNN-based recommender to retrieve similar products.
  - Supports similar-product recommendations across 200+ products.

- **User-Behavior-Based Recommendation**

  The system uses deep learning recommendation models trained on a 10K+ interaction dataset containing clicks, wishlists, carts, and ratings.

  - BERT4Rec: used for cart and purchase history to capture sequential user behavior.
  - ComiRec: used for wishlist and high-rating behavior to capture multiple user interests.
  - Two-Tower: used for click and search behavior to support fast candidate retrieval.

---

## 4. Recommendation Evaluation

The behavior-based recommenders were evaluated using top-k recommendation metrics:

- Recall@K
- NDCG@K
- Hit@10

These metrics were used to compare how well each model retrieved and ranked relevant products for different user-intent signals.

---

## 5. Agentic AI Shopping Assistant

The AI shopping assistant helps users with two types of requests:

- **Product Search Assistance**

  For product-related queries, the assistant uses an LLM-powered agent loop with LangChain. The agent decides when to call tools that query Laravel product APIs for product search, price filtering, and product-detail retrieval.

- **Store-Policy Support with Agentic RAG**

  For policy-related questions, the assistant retrieves relevant context from e-commerce policy documents stored in Pinecone. The knowledge base includes shipping, returns, warranty, payment, FAQ, customer support, privacy policy, terms, and about-us documents.

  The retrieved context is passed to the LLM to generate grounded answers and reduce hallucination risk.

---

## 6. Tech Stack

- FastAPI + Uvicorn
- PyTorch
- scikit-learn
- LangChain (langchain, langchain-openai, langchain-pinecone, langchain-community)
- OpenAI GPT (gpt-5.5)
- Pinecone (vector store for RAG)
- LangSmith (agent tracing and observability)
- pandas, numpy, joblib
- Python-dotenv

---

## 7. Dataset / Knowledge Base

- **Product catalog**: 200+ cleaned products with text and categorical features.
- **User interactions**: 10K+ interactions including clicks, wishlist additions, cart additions, and star ratings (R1–R5).
- **Policy knowledge base**: Store policy PDF covering shipping, returns, refunds, warranty, payment methods, order cancellation, privacy policy, terms, FAQ, and customer support — chunked and embedded into Pinecone.

---

## 8. API Endpoints

All endpoints are prefixed with `/api`.

- **POST /api/recommend**

  Content-based similar product retrieval.

  Request body:
  ```json
  {
    "product_id": 1,
    "model_name": "tfidf_cosine",
    "top_k": 10,
    "version": "v1"
  }
  ```
  - `model_name`: `knn_euclidean` | `tfidf_cosine` | `tfidf_knn_cosine`
  - `version`: `v1` | `v2` | `null` (applicable to `tfidf_cosine`)

  Response:
  ```json
  {
    "product_id": 1,
    "model_name": "tfidf_cosine",
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

  Query parameter: `questions` (string)

  Example: `POST /api/chat?questions=Do you have Nike shoes under 500?`

  Response:
  ```json
  {
    "response": "Yes, here are some Nike shoes under $500: ..."
  }
  ```

---

## 9. Local Setup

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

## 10. Environment Variables

Create a `.env` file at the project root with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name
LARAVEL_API_URL=http://your-laravel-app-url
LANGCHAIN_API_KEY=your_langsmith_api_key
```

---

## 11. Where Models Are Applied in the Website

The following describes where each model applies on https://demo.fashion-shop.uk:

- **KNN (Hybrid: TF-IDF + Cosine Similarity)**: Visible on any product detail page. Scroll to the bottom of the page to find the "You May Also Like These Products" section, which shows content-based similar products.

- **ComiRec**: Shown on the homepage or user feed as the "Continue Exploring Your Style" section. Driven by recent clicks and wishlist activity to capture multiple user interests.

- **Two-Tower Model**: Shown as the "Personalized Recommendations" section. Driven by clicks, wishlist additions, cart additions, and ratings to provide personalized suggestions.

- **BERT4Rec**: A sequence-aware model that considers the order of user interactions to predict what the user is likely to engage with next.

- **AI Shopping Assistant**: Accessible via the floating chat icon at the bottom-right corner of every page. Supports product search with filters and answers store-policy questions using RAG.

---

## 12. Screenshots / Demo

Coming soon.

---

## 13. Future Improvements

- Add real-time model retraining pipeline triggered by new interaction data.
- Expand chatbot tools to support order tracking and account management.
- Introduce A/B testing framework to compare recommendation models in production.
- Add caching layer (e.g., Redis) for frequent recommendation requests.
- Improve cold-start handling for new users with no interaction history.
