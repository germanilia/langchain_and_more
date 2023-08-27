from services.enums import ModelType,ChainType
from services.document_analyzer import DocumentAnalyzer

analyzer = DocumentAnalyzer("examples/static/TSLA-Q4-2022-Update.pdf", model_name=ModelType.AZURE)
analyzer.summarize(ChainType.STUFF)
# analyzer.summarize(ChainType.REFINE)
# analyzer.summarize(ChainType.REDUCE)
# analyzer.stuff_chain()
# analyzer.map_reduce()
# analyzer.refine()