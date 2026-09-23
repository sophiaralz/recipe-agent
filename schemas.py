from typing import List, TypedDict
from pydantic import BaseModel, Field

class Item(BaseModel):
    name: str = Field(description="Name of the item")
    in_fridge: bool = Field(description="True if item is in user's fridge, False if missing")
    quantity: int = Field(description="Quantity of the item")


class Recipe(BaseModel):
    title: str = Field(description="Title of the recipe")
    source_url: str = Field(description="Link to the web recipe")
    prep_time: str = Field(description="Total preparation or cook time")
    items: List[Item] = Field(description="List of items tagged with fridge status")
    missing_items: List[str] = Field(description="Items the user needs to buy")
    instructions: List[str] = Field(description="Step-by-step instructions")

class RecipeListResponse(BaseModel):
    recipes: List[Recipe]

class UserRequest(BaseModel):
    fridge_items: List[str]
    prompt: str

class AgentState(TypedDict):
    fridge_items: List[str]
    user_prompt: str
    search_query: str
    raw_search_results: str
    final_recipes: List[dict]


