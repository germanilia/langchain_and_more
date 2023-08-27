from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from services.document_base import DocumentBase
from services.chat_model_factory import ChatModelFactory
from services.enums import ModelType
from langchain.agents import create_sql_agent
import os
from langchain.agents.agent_types import AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

TEST_QIERIES = [
    "How many customers are there?",
    "Which csutomer has the hightest credit line?",
    "Which csutomer made the most expensieve singel oder?",
    "Which products are included in the most expensieve order?"
]

class SqlService():
    def __init__(self,model_name=ModelType.GPT_4):
        self.llm = ChatModelFactory.get_llm(model_name,0.4)
        sql_string = os.getenv('DATABASE_URI')
        self.db = SQLDatabase.from_uri(sql_string)

    def query_chain(self, message):
        db_chain = SQLDatabaseChain.from_llm(self.llm, self.db, verbose=True)
        return db_chain.run(message)
    
    def query_agent(self, message):
        agent_executor = create_sql_agent(
            llm=self.llm,
            toolkit=SQLDatabaseToolkit(db=self.db, llm=self.llm),
            verbose=True,
            handle_parsing_errors=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )
        return agent_executor.run(message)

    def test_chain(self):
        for query in TEST_QIERIES:
            DocumentBase.pretty_print(self.query_chain(query))
    
    def test_agent(self):
        for query in TEST_QIERIES:
            DocumentBase.pretty_print(self.query_agent(query))
    