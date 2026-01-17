import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader # abre o PDF e extrai o texto puro
from langchain_text_splitters import RecursiveCharacterTextSplitter # quebra o texto em pedaços menores
from langchain_huggingface import HuggingFaceEmbeddings # cria embeddings usando modelo local gratuito
from langchain_community.vectorstores import Chroma # banco de dados vetorial local

# Carrega variáveis de ambiente
load_dotenv()

# Caminhos
DOCS_PATH = "./documents"
DB_PATH = "./data/chroma_db"

def ingest_documents():
    """
    Lê PDFs, quebra em pedaços e salva no Vector DB usando modelo local (CPU).
    """
    print(f"📂 Carregando documentos de {DOCS_PATH}...")
    
    # 1. Carregar PDFs
    loader = PyPDFDirectoryLoader(DOCS_PATH)
    documents = loader.load()
    
    if not documents:
        print("❌ Nenhum documento encontrado. Coloque PDFs na pasta 'documents'.")
        return

    # 2. Quebrar texto (Chunks)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"🧩 Documentos quebrados em {len(chunks)} pedaços.")

    # 3. Criar Embeddings 
    print("⬇️ Baixando/Carregando modelo gratuito (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("💾 Criando banco vetorial (Isso pode demorar um pouco)...")
    
    # Cria o banco e persiste no disco
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    
    print("✅ Banco de dados atualizado com sucesso!")
    return vector_db

def get_retriever():
    """
    Função para recuperar o buscador usando o mesmo modelo gratuito.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    # Busca os 3 trechos mais parecidos
    return vector_db.as_retriever(search_kwargs={"k": 30})

if __name__ == "__main__":
    ingest_documents()