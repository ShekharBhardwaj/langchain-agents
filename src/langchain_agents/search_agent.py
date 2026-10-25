from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
# from tavily import TavilyClient
from langchain_tavily import TavilySearch



load_dotenv()
# tavily = TavilyClient()


# @tool
# def search(query: str) -> str:
#     """
#     Tool that searches over internet
#     Args:
#         query: The query to search for
#     Returns:
#         The search results
#     """
#     print(f"Searching for {query}")
#     return tavily.search(query=query)




anthropic_llm = ChatAnthropic(model="claude-haiku-5-5", effort="low")
tools = [TavilySearch()]
agent = create_agent(model=anthropic_llm, tools=tools)


def main():
    print("Hello I am Langchain agent!")
    result = agent.invoke(
        {"messages": [HumanMessage(content="Search for three job posting for an AI engineer with langchain experience in Wilmington Delaware and list their details")]}
    )
    print(result)


if __name__ == "__main__":
    main()
