from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
import textwrap
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from together_llm import TogetherLLM

class LLAMA2:
    def __init__(self, model_name: str = "togethercomputer/llama-2-70b-chat"):
        self.model_name = model_name
        self.llm = TogetherLLM(
            model=model_name,
            temperature=0.1,
            max_tokens=4096
        )


    def wrap_text_preserve_newlines(self,text, width=110):
        # Split the input text into lines based on newline characters
        lines = text.split('\n')

        # Wrap each line individually
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

        # Join the wrapped lines back together using newline characters
        wrapped_text = '\n'.join(wrapped_lines)

        return wrapped_text


    def process_llm_response(self,llm_response):
        print(self.wrap_text_preserve_newlines(llm_response['result']))
        print('\n\nSources:')
        for source in llm_response["source_documents"]:
            print(source.metadata['source'])

    def load_docs(self, sources_folder="static/", glob="./*.pdf", chunk_size=2000, chunk_overlap=200):
        loader = DirectoryLoader(sources_folder, glob, loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)
        return texts

    def get_retriever(self,sources_folder="static/"):
        persist_directory = 'db'
        ## Here is the nmew embeddings being used
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(documents=self.load_docs(sources_folder),
                                        embedding=embedding_function,
                                        persist_directory=persist_directory)
        return vectordb.as_retriever(search_kwargs={"k": 5})
    
    def ask(self,query,retriever):
        qa_chain = RetrievalQA.from_chain_type(llm=self.llm,
                                       chain_type="stuff",
                                       retriever=retriever,
                                       return_source_documents=True)
        llm_response = qa_chain(query)
        self.process_llm_response(llm_response)

llama2 = LLAMA2()
retriever = llama2.get_retriever("static/arm/")
llama2.ask("What is document about?",retriever)
llama2.ask("Who is in the board of directors?",retriever)
llama2.ask("What is the main buisnes the company?",retriever)
llama2.ask("What are the risk facing the company?",retriever)
llama2.ask("Who are the main competitors?",retriever)
llama2.ask("What are the advantages of the company?",retriever)
