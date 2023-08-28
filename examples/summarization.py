from services.enums import ModelType,ChainType
from services.document_analyzer import DocumentAnalyzer

analyzer = DocumentAnalyzer("static/tesla/TSLA-Q4-2022-Update.pdf", model_name=ModelType.AZURE)
analyzer.summarize(ChainType.STUFF)
analyzer.summarize(ChainType.REFINE)
analyzer.summarize(ChainType.MAP_REDUCE)
analyzer.stuff_chain()
analyzer.map_reduce()
analyzer.refine()