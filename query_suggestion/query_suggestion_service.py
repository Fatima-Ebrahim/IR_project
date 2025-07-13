from fastapi import FastAPI, Depends, HTTPException
from typing import List, Dict, Annotated
from pydantic import BaseModel
from database.database_handler import DatabaseHandler
from .query_suggestion_handler import QuerySuggestionHandler
from utils import config
from utils.logger_config import logger

shared_handlers: Dict[str, QuerySuggestionHandler] = {}

def get_db_handler():
    db = DatabaseHandler(config.MYSQL_CONFIG)
    try:
        db.connect()
        db.setup_tables()
        yield db
    finally:
        db.disconnect()

def get_suggestion_handler(dataset_name: str, db: DatabaseHandler = Depends(get_db_handler)) -> QuerySuggestionHandler:
    if dataset_name not in shared_handlers:
        logger.info(f"Creating new QuerySuggestionHandler for dataset '{dataset_name}'")
        try:
            shared_handlers[dataset_name] = QuerySuggestionHandler(dataset_name, db)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=f"Assets for '{dataset_name}' not found. Error: {e}")
        except Exception as e:
            logger.error(f"Failed to create suggestion handler: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to initialize suggestion service.")
    return shared_handlers[dataset_name]

HandlerDependency = Annotated[QuerySuggestionHandler, Depends(get_suggestion_handler)]

app = FastAPI(
    title="Query Suggestion Service",
    description="Provides autocomplete suggestions as the user types."
)

class SuggestionResponse(BaseModel):
    next_words: List[str]
    full_queries: List[str]

@app.get("/suggest/{dataset_name}", response_model=SuggestionResponse, tags=["Query Suggestion"])
async def suggest_query(
    dataset_name: str,
    q: str,
    handler: HandlerDependency
):
    if not q:
        return SuggestionResponse(next_words=[], full_queries=[])
    try:
        result = handler.get_suggestions(q)
        return SuggestionResponse(
            next_words=result.get("next_words", []),
            full_queries=result.get("full_queries", [])
        )
    except Exception as e:
        logger.error(f"Error during suggestion generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during suggestion generation.")