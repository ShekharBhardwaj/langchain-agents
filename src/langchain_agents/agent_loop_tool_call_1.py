from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_openrouter import ChatOpenRouter
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from langsmith import traceable


MAX_ITERATIONS = 100
MODEL = "claude-haiku-5-5"
# MODEL = "qwen/qwen3.6-35b-a3b"
# LMSTUDIO_BASE_URL = "http://localhost:1234/v1"

# LM Studio serves an OpenAI-compatible API, so the "openai" provider works with a custom base_url.
# The api_key is ignored by LM Studio but the client requires a non-empty value.
llm = init_chat_model(MODEL, model_provider="anthropic")


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

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools=tools)
    print(f"Question: {question}")
    print("=" * 60)
    messages = [SystemMessage(content=(
        "you are a helpful assistant. "
        "You have access to product catalog tool "
        "and a discount tool. \n\n"
        "convert item to lowercase before passing it as arg."
        "STRICT RULES - you must follow these exactly:\n"
        "1. NEVER guess or assume any product price. "
        "You MUST call get_product_price first to get the real price \n"
        "2. Only call apply discount AFTER you have received "
        "a price from get_product_price. Pass that exact price "
        "returned by get_product_price - do NOT pass a made-up number. \n"
        "3. NEVER calculate discounts yourself using math. "
        "Always use the apply_discount tool. "
        "4. If the user does not specify a discount tier, "
        "ask them which tier to use - do NOT assume one"
    )),
                HumanMessage(content=question)]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n ---- Iteration {iteration} ----")
        ai_message = llm_with_tools.invoke(messages)
        # Keep the assistant's tool-call message in history so the model sees what it asked for
        messages.append(ai_message)
        # If no tool calls, this is the final answer
        if not ai_message.tool_calls:
            print(f"\n Final Answer: {ai_message.content}")
            return ai_message.content

        # Run each requested tool and send its result back, linked by tool_call_id
        for tool_call in ai_message.tool_calls:
            tool_name = tool_call["name"]
            tool_arg =  tool_call["args"]
            tool_call_id = tool_call["id"]
            tool_to_use = tools_dict[tool_name]
            observation = tool_to_use.invoke(tool_arg)
            print(f"    << {tool_name} returned {observation}")
            messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    raise RuntimeError(f"Agent did not finish within {MAX_ITERATIONS} iterations")


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind tools)! \n")
    result = run_agent("What is the price of a LAPTOP after applying a gold discount?")
    print(result)
