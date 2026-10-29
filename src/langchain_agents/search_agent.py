from typing import List

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
# from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")


anthropic_llm = ChatAnthropic(model="claude-haiku-5-5", effort="low")
tools = [TavilySearch()]
agent = create_agent(model=anthropic_llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello I am Langchain agent!")
    result = agent.invoke(
        {"messages": [HumanMessage(content="Search for three job posting for an AI engineer with langchain experience in Wilmington Delaware and list their details")]}
    )
    print(result)


if __name__ == "__main__":
    main()
