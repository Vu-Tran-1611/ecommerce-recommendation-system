import os 
import requests 
from langchain.tools import tool 
from typing import Optional, Dict, Any
from dotenv import load_dotenv
load_dotenv()

LARAVEL_API_URL = os.getenv("LARAVEL_API_URL") 
LARAVEL_PUBLIC_URL = os.getenv("LARAVEL_PUBLIC_URL")
@tool 
def search_products(
    keywords:Optional[str] = None,
    category:Optional[str] = None, 
    subcategory:Optional[str] = None,
    brand:Optional[str] = None,
    min_price:Optional[float] = None,
    max_price:Optional[float] = None, 
    product_type:Optional[str] = None 
)-> Dict[str, Any]: 
    """Search Products from the Laravel ecommerce backend. 
    
    -----Keyword arguments:
    keywords -- Keywords to search for products
    category -- Category to filter products
    -----Available categories: 
    "fashion","electronics","Mobile Phones,"Computers & Laptops","Watches",
    "Health & Beauty","Grocery & Food", "Toys & collectibles" 
    subcategory -- Subcategory to filter products 
    -----Available subcategories:
    "Vest & Blazer", "Hoodie & Sweater", "Jean", "Shirt", "T-shirt", "Trouser",
    "Phone", "Tablet", "Phone Case", "Sim", "Battery",
    "Smart Watch", "Gaming console", "Game Disc", "Keyboard", "Speaker", "Screen", "Mouse",
    "Men Watch", "Women Watch", "Kid Watch",
    "Skincare", "Hair Care", "Supplements", "Personal Care", "Makeup",
    "Snacks", "Beverages", "Instant Food",
    "Laptop", "Plush Toys", "Model Figures"
    brand -- Brand to filter products
    -----Available brands: "Apple", "Samsung", "Nike", "Adidas", "Sony", "LG", "Dell", "HP", "Lenovo", "Asus","Puma","Reebok",...
    min_price -- Minimum price to filter products
    max_price -- Maximum price to filter products
    product_type -- Type of product to filter products 
    -----Available product types: "top", "featured", "new_arrival", "best"

    "If products are found, start with short sentence such as:" \
    " 'Here are some products you might like:' or 'I found some products that match your criteria:' followed by the format:" \
        "* <b class='text-xl'> product name </b>" \
        "<img class='h-[300px] w-[300px]' alt='thumbnail image' src='thumbnail image URL of product' />" \
            "<b>Price </b>: price of product" \
            "<b>Brand </b>: brand of product" \
            "<b>Category </b>: category of product" \
            "<b>link </b>: <a href=' https://demo.fashion-shop.uk/product?product=URL to product'>Product Link</a> " \
    " When listing products in the final answer, show at least 7 products if more than 7 are available."
    If a product search returns 0 results, try one broader search before giving the final answer.
    For example:
    - Remove plural forms: "watches" -> "watch"
    - Use subcategory when possible: "watches for men" -> subcategory "Men Watch"
    - If subcategory search fails, search only by category and max_price
    - Do not retry more than once or twice
    "After retry, if no products are found, say 'I couldn't find any products that match your criteria. Please try different keywords. If the user's request is unclear, ask for clarification."
    Examples: 
    -"Find Apple products under $1000" -> search_products(keywords="Apple",brand="Apple", max_price=1000) 
    -"Show me top clothes" -> search_products(category="fashion",product_type="top") 
    """
    params = {
        "keywords": keywords,
        "category": category,
        "subcategory": subcategory,
        "brand": brand,
        "min_price": min_price,
        "max_price": max_price,
        "product_type": product_type
    }
    response = requests.get(f"{LARAVEL_API_URL}/api/ai/products/search", params=params)
    if response.status_code == 200:
        return response.json() 
    else: 
        return {"error": f"Failed to search products. Status code: {response.status_code}"}
    

    