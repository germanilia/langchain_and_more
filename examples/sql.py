from services.enums import ModelType
from services.sql_service import SqlService

sql_serivce = SqlService(model_name=ModelType.GPT_3_5_TURBO_16K)
# sql_serivce.test_chain()
sql_serivce.test_agent()
