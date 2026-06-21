from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from duckduckgo_search import DDGS
import streamlit as st
import requests
# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=st.secrets["GROQ_API_KEY"],
    temperature=0
)

# Math tools
@tool
def add(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers."""
    return a * b

@tool
def subtract(a: int, b: int) -> int:
    """Subtracts two numbers."""
    return a - b

# Web search tool
@tool
def web_search(query: str) -> str:
        """Search the web using Tinyfish. Use this to get current information accurately."""
        url = "https://api.search.tinyfish.ai"
        headers = {"X-API-Key": st.secrets["TINYFISH_API_KEY"]}
        params = {"query": query}
        
        response = requests.get(url, headers=headers, params=params)
        results = response.json()
        
        output = ""
        for r in results.get("results", []):
            output += f"Title: {r.get('title', '')}\nSummary: {r.get('snippet', '')}\nURL: {r.get('url', '')}\n\n"
        
        return output if output else "No results found."
# All tools together
tools = [add, multiply, subtract, web_search]

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools when needed. use web search and give detailed information and give some bullet points "),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True

)


# UI
st.title("Agentic AI On Stand-By")

question = st.text_input("Ask a question to My Agentic AI")

if question:
    with st.spinner("Agentic AI searching..."):
        result = agent_executor.invoke({"input": question})
        st.write(result["output"]) 