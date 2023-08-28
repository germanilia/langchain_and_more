from services.enums import ModelType
from services.document_qa import DocumentQA


qa = DocumentQA("static/TSLA-Q4-2022-Update.pdf", model_name=ModelType.AZURE)
qa.qa("What is the revenue for the quarter?")
qa.qa("What is the revenue for the 1st quarter?")
qa.qa("When is the cybertruck coming out?")



