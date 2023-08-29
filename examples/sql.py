from services.enums import ModelType
from services.sql_service import SqlService

sql_serivce = SqlService(model_name=ModelType.GPT_4)
sql_serivce.test_chain()
sql_serivce.test_agent()
