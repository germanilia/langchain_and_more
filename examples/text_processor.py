from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextProcessor:
    @staticmethod
    def clean_html_from_text(text):
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)
    
    @staticmethod
    def split_docs(docs):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=5000, chunk_overlap=0)
        return text_splitter.split_documents(docs)