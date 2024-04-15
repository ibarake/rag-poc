from clients import vector_store, llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

class chat:
    def __init__(self):
        self.retriever = vector_store.as_retriever()

    # Function to format documents for readability
    def format_documents(self, docs) -> str:
        return "\n\n".join(doc.page_content for doc in docs)
    

    def search_documents(self, query: str, num_results: int = 5):
        return vector_store.similarity_search_with_relevance_scores(query, k=num_results)
    
    # Example of document processing and querying
    def create_rag_pipeline(self):
        # Set up the vector store retriever
        retriever = self.retriever

        # Set up a custom prompt for querying
        template_text = (
            """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Finish the info: make sure not to write your own but to write what is in the documents.

        {context}

        Question: {question}

        Helpful Answer:"""
        )
        custom_prompt = PromptTemplate.from_template(template_text)

        # Query using the language model and custom prompt
        rag_chain = (
            {"context": retriever | self.format_documents, "question": RunnablePassthrough()}
            | custom_prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain