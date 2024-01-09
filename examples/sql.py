from services.enums import ModelType
from services.sql_service import SqlService
TEST_QUERIES = [
    "How many customers are there?",
    "Which customer has the hightest credit line?",
    "Which customer made the most expensive single oder?",
    "Which products are included in the most expensive order?",
]

sql_service = SqlService(model_name=ModelType.GPT_4)
sql_service.test_chain(TEST_QUERIES)
sql_service.test_agent(TEST_QUERIES)
