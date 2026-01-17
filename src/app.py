import streamlit as st # Biblioteca principal do Streamlit
import os
from dotenv import load_dotenv # Carregar variáveis de ambiente do arquivo .env
from langchain_groq import ChatGroq # Modelo LLM da Groq
from langchain.chains import create_retrieval_chain # Cria a cadeia de recuperação
from langchain.chains.combine_documents import create_stuff_documents_chain # Combina documentos em uma cadeia
from langchain_core.prompts import ChatPromptTemplate # Cria prompts para chat
from rag_engine import get_retriever

# 1. Configuração da Página (Título e Ícone)
st.set_page_config(page_title="ChefBot IA", page_icon="👨‍🍳")

st.title("👨‍🍳 ChefBot - Seu Assistente Culinário")
st.caption("Pergunte sobre receitas do livro Nutribook!")

# 2. Carregar variáveis de ambiente
load_dotenv()

# 3. Configuração da IA (Com Cache para não recarregar toda hora)
@st.cache_resource
def get_rag_chain():
    # Modelo LLM (O mesmo que configuramos antes)
    llm = ChatGroq(model_name="llama-3.3-70b-versatile")
    
    # O "Bibliotecário" (Retriever) que busca no ChromaDB
    retriever = get_retriever()
    
    # O Prompt (A personalidade do robô)
    system_prompt = (
        "Você é um assistente culinário especializado e amigável. "
        "Use os seguintes pedaços de contexto recuperados para responder à pergunta. "
        "Se você não souber a resposta, diga honestamente que não sabe. "
        "Seja detalhista nas receitas.\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Montando a corrente (Chain)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# Inicializa a IA
try:
    rag_chain = get_rag_chain()
except Exception as e:
    st.error(f"Erro ao carregar a IA. Verifique se o Groq API Key está no .env e se o banco de dados foi criado. Detalhe: {e}")
    st.stop()

# 4. Histórico do Chat (Memória da Sessão)
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Qual receita você quer aprender hoje?"}]

# 5. Exibir mensagens anteriores na tela
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 6. Caixa de Entrada do Usuário
if prompt := st.chat_input("Digite sua pergunta... (Ex: Como fazer Mousse?)"):
    # Mostra a mensagem do usuário na tela
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Gera a resposta da IA
# Gera a resposta da IA
    with st.chat_message("assistant"):
        with st.spinner("Consultando o livro de receitas... 📖"):
            response = rag_chain.invoke({"input": prompt})
            bot_reply = response["answer"]
            st.write(bot_reply)
            
            # Cria uma caixinha expansível (daquelas que clica para abrir)
            with st.expander("📚 Ver páginas consultadas"):
                for i, doc in enumerate(response["context"]):
                    # Pega o nome do arquivo (se tiver) e a página
                    origem = doc.metadata.get("source", "Arquivo desconhecido")
                    pagina = doc.metadata.get("page", "Sem pág")
                    
                    # Mostra bonito na tela
                    st.write(f"**Fonte {i+1}:** {origem} (Pág. {pagina})")
                    # Mostra um pedacinho do texto que ele leu
                    st.caption(f"Trecho: {doc.page_content[:100]}...")
                    st.divider() # Uma linha separadora
            # --------------------------------
    
    # Salva a resposta no histórico
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})