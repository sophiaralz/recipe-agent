import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from schemas import AgentState, RecipeListResponse
from tools import recipe_search_tool

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1
)

def query_node(state: AgentState):
    fridge_str = ", ".join(state["fridge_items"])
    prompt = (
        f"The user has these items in their fridge: {fridge_str}. \n"
        f"User desire: {state["user_prompt"]}. \n"
        f"Write a concise web search query to find online recipes matching these constraints."
    )
    response = llm.invoke(prompt)
    return {"search_query": response.content.strip('"')}

def search_node(state: AgentState):
    results = recipe_search_tool.invoke({"query" : state["search_query"]})
    return {"raw_search_results" : str(results)}

def extract_node(state: AgentState):
    structured_llm = llm.with_structured_output(RecipeListResponse)

    prompt = f"""
    User Fridge Items: {state["fridge_items"]} 
    User Prompt: {state["user_prompt"]}

    Raw Web Search Results: {state["raw_search_results"]}

    Extract at least 2 and maximum 5 matching recipe from the search results. The top results must have all or most items that the user has in their fridge.
    For each recipe:
    1. Extract a valid link or reference URL
    2. Check each required ingredient against the fridge items
    3. Mark 'in_fridge: True' if available or 'in_fridge: False' if missing
    4. Populate 'missing_ingredients' with only items missing from the fridge
    """

    output = structured_llm.invoke(prompt)
    return {"final_recipes": [r.dict() for r in output.recipes]}


# build graph

builder = StateGraph(AgentState)
builder.add_node("generate_query", query_node)
builder.add_node("execute_search", search_node)
builder.add_node("extract_recipes", extract_node)

builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "execute_search")
builder.add_edge("execute_search", "extract_recipes")
builder.add_edge("extract_recipes", END)


recipe_graph = builder.compile()




