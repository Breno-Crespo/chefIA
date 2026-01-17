import os
import sys
from dotenv import load_dotenv

# Bibliotecas de IA
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Carrega as senhas do .env
load_dotenv()

# Configurações
DB_PATH = "./data/chroma_db"
GROQ_MODEL = "llama-3.3-70b-versatile"  # Modelo

def get_retriever():
    
   # Usa o mesmo modelo da HuggingFace que usado no RAG
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Conecta ao banco existente
    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_function
    )
    
    # Cria o Retriever que traz os trechos mais relevantes
    return vector_db.as_retriever(search_kwargs={"k": 30})

def run_chat_loop():
    print("🧠 Inicializando o Cérebro do Agente (Groq + RAG)...")
    
    # 1. LLM 
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2  # Baixa criatividade para ser fiel aos dados
    )

    # 2. Memória
    try:
        retriever = get_retriever()
    except Exception as e:
        print(f"❌ Erro ao conectar no banco: {e}")
        return

    # 3. O Prompt (A Personalidade)
    system_prompt = (
        "Você é uma vovó virtual. Responda com carinho, use emojis de comida e sempre dê uma dica extra de culinaria no final. "
        "Use APENAS o contexto fornecido abaixo para responder à dúvida do cliente. "
        "Se a resposta não estiver no contexto, diga educadamente que não sabe "
        "e sugira abrir um chamado técnico. Responda sempre em Português do Brasil.\n\n"
        "--- CONTEXTO ---\n"
        "{context}\n"
        "----------------"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 4. Criar a "Corrente" de Processamento (Chain)
    # Chain de Documentos: Junta a resposta do LLM com os Docs achados
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    # Chain de Recuperação: Faz a busca -> Passa pra Chain de cima
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n✅ Agente Helpdesk Online! (Digite 'sair' para encerrar)")
    print("---------------------------------------------------------")

    # 5. Loop de Conversa
    while True:
        try:
            user_input = input("👤 Cliente: ")
            if user_input.lower() in ["sair", "exit", "tchau"]:
                print("🤖 Agente: Até logo! O atendimento foi encerrado.")
                break
            
            print("⏳ Consultando base de conhecimento...")
            
            # A Mágica acontece aqui: .invoke()
            response = rag_chain.invoke({"input": user_input})
            
            print(f"🤖 Agente: {response['answer']}\n")
            
            # Debug: Se quiser ver de qual parte do PDF ele tirou a resposta, descomente abaixo:
            # for doc in response['context']:
            #     print(f"   [Fonte]: {doc.page_content[:60]}...")
            #     print("   ---")

        except KeyboardInterrupt:
            print("\nEncerrando...")
            break
        except Exception as e:
            print(f"❌ Erro na resposta: {e}")

if __name__ == "__main__":
    run_chat_loop()