from app import chat

chat_app = chat()
# search = chat_app.search_documents("We offer to revolutionize websites.")
# print(search)
pipeline = chat_app.create_rag_pipeline()
response = pipeline.invoke("We offer to revolutionize websites.")
print(response)