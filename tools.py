from langchain_community.tools import DuckDuckGoSearchResults

recipe_search_tool = DuckDuckGoSearchResults(max_results=5)
