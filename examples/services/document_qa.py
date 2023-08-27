from services.document_loader import DocumentLoader
from services.document_analyzer import DocumentAnalyzer
from services.document_base import DocumentBase
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain
from services.enums import ModelType,ChainType


class DocumentQA(DocumentBase):
    def __init__(self, path, model_name=ModelType.GPT_3_5_TURBO_16K):
        super().__init__(model_name)
        self.analizer = DocumentAnalyzer(path)
        self.loader = DocumentLoader(path)
        docs = self.loader.load()
        self.document = docs[0].page_content

    def qa(self, question):
        qa_chain = load_qa_chain(self.llm, chain_type=ChainType.MAP_REDUCE.value)
        qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
        answer = qa_document_chain.run(input_document=self.document, question=question)
        DocumentBase.pretty_print(answer)
        return answer




