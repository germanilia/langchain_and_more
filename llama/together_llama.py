
from together_llm import TogetherModel

llama2 = TogetherModel()
retriever = llama2.get_retriever("static/arm/")
llama2.ask("Who are in the board of directors of ARM company?")
llama2.ask_retriever("Who is in the board of directors?",retriever)
llama2.ask("What is the main buisnes the company of ARM company?")
llama2.ask_retriever("What is the main buisnes the company?",retriever)
llama2.ask("What are the risk facing arm company is facing?")
llama2.ask_retriever("What are the risk facing the company?",retriever)
llama2.ask("Who are the main competitors of ARM company?")
llama2.ask_retriever("Who are the main competitors?",retriever)
llama2.ask("What are the advantages of the company?")
llama2.ask_retriever("Who are the main competitors?",retriever)
