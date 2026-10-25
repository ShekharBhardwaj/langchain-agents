from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic


@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"Searching for {query}")
    return "Tokyo weather is sunny"


anthropic_llm = ChatAnthropic(model="claude-haiku-5-5", effort="low")
tools = [search]
agent = create_agent(model=anthropic_llm, tools=tools)


def main():
    print("Hello I am Langchain agent!")
    result = agent.invoke(
        {"messages": HumanMessage(content="What is the weather in Tokyo")}
    )
    print(result)


if __name__ == "__main__":
    main()
