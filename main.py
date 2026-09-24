from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schemas import UserRequest
from agent import recipe_graph

app = FastAPI(title="Recipe Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok", "message" : "Backend server is running!"}

@app.post("/api/recipes")
async def generate_recipes(request: UserRequest):
    try:
        initial_state = {
            "fridge_items": request.fridge_items,
            "user_prompt": request.prompt,
            "search_query": "",
            "raw_search_results": "",
            "final_recipes": []
        }

        final_state = await recipe_graph.ainvoke(initial_state)
        return {"recipes": final_state["final_recipes"]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

