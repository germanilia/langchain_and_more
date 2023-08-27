from services.enums import ModelType
from langchain.chains import MapReduceDocumentsChain,ReduceDocumentsChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from services.document_base import DocumentBase

from services.document_loader import DocumentLoader
from langchain.prompts import PromptTemplate

class DocumentAnalyzer(DocumentBase):
    def __init__(self, path, model_name=ModelType.GPT_3_5_TURBO_16K):
        self.loader = DocumentLoader(path)
        self.docs = self.loader.load_and_split()
        super().__init__(model_name)
        
    
    def summarize(self, chain_type):
        chain = self.get_chain(chain_type.value)
        result = chain.run(self.docs)
        self.pretty_print(result)
        return result
    
    def get_chain(self, chain_type):
        chain = load_summarize_chain(self.llm, chain_type=chain_type)
        return chain

    def stuff_chain(self):
    
        prompt_template = """Write a concise summary of the following:
        "{text}"
        CONCISE SUMMARY:"""
        prompt = PromptTemplate.from_template(prompt_template)

        # Define LLM chain
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(
            llm_chain=llm_chain, document_variable_name="text"
        )
        restult = stuff_chain.run(self.docs)
        self.pretty_print(restult)
        return restult


    def map_reduce(self):
        # Map
        map_template = """The following is a set of documents
        {docs}
        Based on this list of docs, please identify the main themes 
        Helpful Answer:"""
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=self.llm, prompt=map_prompt)

        # Reduce
        reduce_template = """The following is set of summaries:
        {doc_summaries}
        Take these and distill it into a final, consolidated summary of the main themes. 
        Helpful Answer:"""
        reduce_prompt = PromptTemplate.from_template(reduce_template)
        reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)
        combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
        )
        reduce_documents_chain = ReduceDocumentsChain(
            combine_documents_chain=combine_documents_chain,
            collapse_documents_chain=combine_documents_chain,
            token_max=10000,
        ) 

        # Combined
        map_reduce_chain = MapReduceDocumentsChain(
            llm_chain=map_chain,
            reduce_documents_chain=reduce_documents_chain,
            document_variable_name="docs",
            return_intermediate_steps=False,
            verbose=True,
        )
        restult = map_reduce_chain.run(self.docs)
        self.pretty_print(restult)
        return restult

    def refine(self):
        prompt_template = """Write a concise summary of the following:
        {text}
        CONCISE SUMMARY:"""
        prompt = PromptTemplate.from_template(prompt_template)

        refine_template = (
    """
    You are a financial analyst at a bloomberg, you are analyazing "ARM" documents
    You need to decide if you want to invest in this IPO.
    You have a lot of documents to read, and you need to summarize them. here is a summary of the documents you have read so far:
    {existing_answer}
    You have the opportunity to refine the existing summay (only if needed) with some more context below:
    "------------\n"
    "{text}\n"
    "------------\n"
    "If the context isn't useful, return the original summary."
    """
        )
        refine_prompt = PromptTemplate.from_template(refine_template)
        chain = load_summarize_chain(
            llm=self.llm,
            chain_type="refine",
            question_prompt=prompt,
            refine_prompt=refine_prompt,
            return_intermediate_steps=True,
            input_key="input_documents",
            output_key="output_text",
        )                        
        result = chain({"input_documents": self.docs}, return_only_outputs=True)
        self.pretty_print(result["output_text"])
        return result["output_text"]

    