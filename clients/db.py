import os
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from supabase.client import Client, create_client

supabase_url = os.environ.get("SUPA_URL")
supabase_key = os.environ.get("SUPA_KEY")
db: Client = create_client(supabase_url, supabase_key)

openai_embeddings = OpenAIEmbeddings()

vector_store = SupabaseVectorStore(
                client=db,
                embedding=openai_embeddings,
                table_name="documents",
                query_name="match_documents"
            )