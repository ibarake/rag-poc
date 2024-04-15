from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from clients import vector_store

class Documents:
    LOADERS = {
        '.md': UnstructuredMarkdownLoader,
        '.csv': CSVLoader,
    }
    ENCODING_NAME = "cl100k_base"

    def __init__(self, filespath: str):
        self.DATA_PATH = filespath

    def create_embeddings(self):
        documents = self.load_data(".csv") + self.load_data(".md")
        chunks = self.split_text(documents)

        return chunks

    def load_data(self, data_type):
        loader_cls = self.LOADERS.get(data_type)
        if loader_cls is None:
            raise ValueError(f"No loader available for data type: {data_type}")
        md_loader = DirectoryLoader(self.DATA_PATH, glob=f"**/*{data_type}", loader_cls=loader_cls)
        data = md_loader.load()
        return data

    def split_text(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=500,
            length_function=len,
            add_start_index=True
        )
        docs = text_splitter.split_documents(documents)
        response = self.store_in_db(docs)

        print(f"Split {len(documents)} documents into {len(docs)} chunks")
        print(f"Stored {len(response)} embeddings in the database")
        return docs, response

    def num_tokens_from_string(self, string: str) -> int:
        encoding = tiktoken.get_encoding(self.ENCODING_NAME)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def store_in_db(self, docs: list[Document]):
        try:
            db_vectors = vector_store.add_documents(
                documents=docs
            )
            return db_vectors
        except Exception as e:
            print(f"Error creating embeddings: {e}")
            return []