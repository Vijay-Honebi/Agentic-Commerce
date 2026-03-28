from fastapi import APIRouter
from conversational_commerce.api.v1.response_template import ResponseTemplate
from conversational_commerce.api.v1.search_utils import test_query

_response_template = ResponseTemplate()
router = APIRouter(prefix="/ai-search", tags=["Product Feature Extraction"])


@router.post(
    "/get-products",
    response_model=dict,
    summary="Extract relevent product ids from user input using semantic search",
)
async def get_products(user_query: dict):
    try:
        return _response_template._success_response_template(test_query(user_query["user_query"]))
    except Exception as e:
        return _response_template._error_response_template(str(e))