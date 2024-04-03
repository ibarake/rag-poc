from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from clients import openai_api 
import tiktoken

class Embeddings:
    
    LOADERS = {
        '.md' : UnstructuredMarkdownLoader,
        '.csv' : CSVLoader,
    }

    EMBEDDING_MODEL = "text-embedding-ada-002"

    ENCODING_NAME = "cl100k_base"
    

    def __init__(self, filespath: str):
        self.DATA_PATH = filespath
        DOCUMENTS = self.load_data(".csv") + self.load_data(".md")
        CHUNKS = [str(chunk) for chunk in self.split_text(DOCUMENTS)]
        EMBEDDINGS = []
        total_tokens = 0
        for chunk in CHUNKS:
            tokens = self.num_tokens_from_string(chunk)
            if total_tokens + tokens < 8192:
                total_tokens += tokens
                EMBEDDINGS.append(self.create_embedding([chunk]))
            else:
                break

        return EMBEDDINGS
    
    def load_data(self, data_type):
        md_loader = DirectoryLoader(self.DATA_PATH, glob=f"**/*{data_type}", loader_cls=self.LOADERS[data_type])
        data = md_loader.load()
        return data

    def split_text(self, documents: list[Document]):
        # TODO: improve text splitting to fit the data better
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=500,
            length_function=len,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(documents)

        print(f"Split {len(documents)} documents into {len(chunks)} chunks")

        return chunks
    
    def num_tokens_from_string(self, string: str) -> int:
        encoding = tiktoken.get_encoding(self.ENCODING_NAME)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    
    def create_embedding(self, chunks: list[str]):
        response = openai_api.embeddings.create(model=self.EMBEDDING_MODEL, input=chunks[0])
        return response.data[0].embedding
