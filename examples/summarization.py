from services.document_analyzer import DocumentAnalyzer

analyzer = DocumentAnalyzer("examples/static/TSLA-Q4-2022-Update.pdf", model_name="azure")
# analyzer.summarize("stuff")
# analyzer.summarize("refine")
# analyzer.summarize("map_reduce")
analyzer.stuff_chain()
analyzer.map_reduce()
analyzer.refine()