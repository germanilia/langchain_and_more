from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from services.document_base import DocumentBase
from services.chat_model_factory import ChatModelFactory
from services.enums import ModelType
from langchain.agents import create_sql_agent
import os
from langchain.agents.agent_types import AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit



class SqlService():
    def __init__(self,model_name:ModelType):
        self.llm = ChatModelFactory.get_llm(model_name,0.4)
        sql_string = os.getenv('DATABASE_URI',"None")
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

    def test_chain(self, test_queries: list[str]) -> None:
        for query in test_queries:
            try:
                DocumentBase.pretty_print(f"Testing query: {query}, using chain")
                DocumentBase.pretty_print(self.query_chain(query))
            except Exception as e:
                DocumentBase.pretty_print(f"Failed with query: {query}, using chain")
    
    def test_agent(self, test_queries: list[str]):
        for query in test_queries:
            try:
                DocumentBase.pretty_print(f"Testing query: {query}, using agent")
                DocumentBase.pretty_print(self.query_agent(query))
            except Exception as e:
                DocumentBase.pretty_print(f"Failed with query: {query}, using chain")
    