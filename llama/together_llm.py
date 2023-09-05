from langchain.utils import get_from_dict_or_env
from langchain.llms.base import LLM
from pydantic import Extra, root_validator
from typing import Any, Dict
import os
import together
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import base64
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
import textwrap
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv('llama/.env')


class TogetherLLM(LLM):
    """Together large language models."""

    model: str = "togethercomputer/llama-2-70b-chat"
    """model endpoint to use"""

    together_api_key: str = os.getenv("TOGETHER_API_KEY")
    """Together API key"""

    temperature: float = 0.7
    """What sampling temperature to use."""

    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion."""

    class Config:
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the API key is set."""
        api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = api_key
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,
                                          model=self.model,
                                          max_tokens=self.max_tokens,
                                          temperature=self.temperature,
                                          )
        output = output['output']['choices'][0]
        try:
            return output['text']
        except KeyError:
            return output['image_base64']


class TogetherModel:
    def __init__(self, model_name: str = "togethercomputer/llama-2-13b-chat"):
        self.model_name = model_name
        self.llm = TogetherLLM(
            model=model_name,
            temperature=0.1,
            max_tokens=4096
        )

    def wrap_text_preserve_newlines(self, text, width=110):
        # Split the input text into lines based on newline characters
        lines = text.split('\n')

        # Wrap each line individually
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

        # Join the wrapped lines back together using newline characters
        wrapped_text = '\n'.join(wrapped_lines)

        return wrapped_text

    def process_llm_response(self, llm_response):
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

    def get_retriever(self, sources_folder="static/"):
        persist_directory = 'db'
        # Here is the nmew embeddings being used
        embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(documents=self.load_docs(sources_folder),
                                         embedding=embedding_function,
                                         persist_directory=persist_directory)
        return vectordb.as_retriever(search_kwargs={"k": 5})

    def ask_retriever(self, query, retriever):
        qa_chain = RetrievalQA.from_chain_type(llm=self.llm,
                                               chain_type="stuff",
                                               retriever=retriever,
                                               return_source_documents=True)
        llm_response = qa_chain(query)
        self.process_llm_response(llm_response)

    def ask(self, query):
        llm_response = self.llm(query)
        print(self.wrap_text_preserve_newlines(llm_response))

    def image(self, query, output_image=None):
        encoded_image_data = self.llm(query)
        image_data = base64.b64decode(encoded_image_data)
        output_image = output_image or str(uuid.uuid4())[:8] + ".jpg"

        with open(output_image, 'wb') as file:
            file.write(image_data)
        print(f"Image saved to {output_image}")
