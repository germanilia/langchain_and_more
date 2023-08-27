from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PDFMinerPDFasHTMLLoader
from services.text_processor import TextProcessor
class DocumentLoader:
    def __init__(self, path):
        self.path = path

    def load_and_split(self):
        docs = self.load()
        return TextProcessor.split_docs(docs)
        
    
    def load(self):
        if self.path.startswith("http"):
            loader = WebBaseLoader(self.path)
            docs = loader.load()
            return docs
        elif ".pdf" in self.path:
            loader = PDFMinerPDFasHTMLLoader(self.path)
            docs = loader.load()
            docs[0].page_content = TextProcessor.clean_html_from_text(docs[0].page_content)
            return docs
    