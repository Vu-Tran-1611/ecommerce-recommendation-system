import os 
import requests 
from langchain.tools import tool 
from typing import Optional, Dict, Any
from dotenv import load_dotenv
load_dotenv()

LARAVEL_API_URL = os.getenv("LARAVEL_API_URL") 

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
    
    Keyword arguments:
    keywords -- Keywords to search for products
    category -- Category to filter products
    Available categories: 
    "fashion","electronics","Mobile Phones,"Computers & Laptops","Watches",
    "Health & Beauty","Grocery & Food", "Toys & collectibles" 
    subcategory -- Subcategory to filter products
    brand -- Brand to filter products
    Available brands: "Apple", "Samsung", "Nike", "Adidas", "Sony", "LG", "Dell", "HP", "Lenovo", "Asus","Puma","Reebok",...
    min_price -- Minimum price to filter products
    max_price -- Maximum price to filter products
    product_type -- Type of product to filter products 
    Available product types: "top", "featured", "new_arrival", "best"

    Return: Dictionary containing search results 
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
    

    