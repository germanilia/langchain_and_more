from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PDFMinerPDFasHTMLLoader
from text_processor import TextProcessor
class DocumentLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        if self.path.startswith("http"):
            loader = WebBaseLoader(self.path)
            docs = loader.load()
            return TextProcessor.split_docs(docs)
        elif ".pdf" in self.path:
            loader = PDFMinerPDFasHTMLLoader(self.path)
            docs = loader.load()
            docs[0].page_content = TextProcessor.clean_html_from_text(docs[0].page_content)
            return TextProcessor.split_docs(docs)