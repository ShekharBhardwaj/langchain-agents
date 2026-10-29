from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_openrouter import ChatOpenRouter
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from langsmith import traceable


MAX_ITERATIONS = 10
MODEL = "qwen/qwen3.6-35b-a3b"
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"

# LM Studio serves an OpenAI-compatible API, so the "openai" provider works with a custom base_url.
# The api_key is ignored by LM Studio but the client requires a non-empty value.
llm = init_chat_model(MODEL, model_provider="openai", base_url=LMSTUDIO_BASE_URL, api_key="lm-studio", temperature=0)


# Tools (LangChain @tool decorator)

@tool
def get_product_price(product: str)-> float:
    """Look up the price of a product in the catalog"""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, 
              "headphones": 149.95, 
              "keyboard": 89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price
    Available tiers are gold, silver, bronze"""
    discount_percentages = {
        "bronze": 5,
        "silver": 12, 
        "gold": 23
    }
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)
    
# Agent loop

traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)
    response = llm_with_tools.invoke([HumanMessage(content=question)])
    print(response.tool_calls or response.content)
    return response


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind tools)! \n")
    result = run_agent("What is the price of a laptop after applying a gold discount?")
