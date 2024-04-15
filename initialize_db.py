from app import Documents, Database

def initialize_db():
    db = Database()
    data = Documents("./documents")
    embeddings = data.create_embeddings()

initialize_db()