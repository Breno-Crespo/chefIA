import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from rag_engine import get_retriever

# --- 1. CONFIGURAÇÃO DA PÁGINA (Deve ser a primeira linha) ---
st.set_page_config(
    page_title="NutriChef IA",
    page_icon="🥦",
    layout="wide", # Layout mais largo para aproveitar a tela
    initial_sidebar_state="expanded"
)

# --- 2. ESTILO CSS PERSONALIZADO (A Mágica Visual) ---
st.markdown("""
<style>
    /* Cor de Fundo e Textos */
    .stApp {
        background-color: #f9f9f9; /* Cinza bem clarinho (Clean) */
    }
    
    /* Título Principal */
    h1 {
        color: #2E8B57; /* SeaGreen (Verde bonito) */
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Subtítulo */
    .stCaption {
        text-align: center;
        font-size: 1.1em;
        color: #666;
    }

    /* Caixa de Chat (Input) */
    .stTextInput > div > div > input {
        border-radius: 20px;
    }

    /* Botões */
    .stButton > button {
        border-radius: 20px;
        background-color: #2E8B57;
        color: white;
        border: none;
    }
    .stButton > button:hover {
        background-color: #3CB371;
        color: white;
    }
    
    /* Esconder menu padrão do Streamlit (Hambúrguer) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 3. LOGICA DO BACKEND (Mantida Igual) ---
load_dotenv()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state:
        st.session_state[session_id] = ChatMessageHistory()
    return st.session_state[session_id]

@st.cache_resource
def get_rag_chain():
    # Se precisar trocar a chave ou modelo, é aqui
    llm = ChatGroq(model_name="llama-3.3-70b-versatile")
    retriever = get_retriever()

    # Contextualização da Pergunta
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Resposta Final
    system_prompt = (
        "Você é o 'NutriChef', um assistente culinário especializado em receitas saudáveis do livro Nutribook. "
        "Seja gentil, use emojis 🥗🥘 e formate bem a resposta (use tópicos). "
        "Se não souber a resposta no contexto, diga que não sabe, não invente. "
        "\n\nContexto:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

# --- 4. SIDEBAR (Barra Lateral) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3408/3408759.png", width=100) # Ícone de Chef
    st.title("NutriChef IA")
    st.markdown("---")
    st.markdown("**Sobre:**\nEste assistente utiliza Inteligência Artificial para navegar.")
    st.markdown("**Tecnologias:**\nPython, LangChain, Groq & Streamlit.")
    
    st.markdown("---")
    # Botão para limpar conversa
    if st.button("🗑️ Limpar Conversa"):
        st.session_state.messages = []
        st.session_state["usuario_padrao"] = ChatMessageHistory()
        st.rerun()

# --- 5. INTERFACE PRINCIPAL ---
st.title("🥗 NutriChef - Seu Assistente Saudável")
st.caption("Pergunte sobre receitas, ingredientes e dicas de preparo!")

# Inicializa o robô
try:
    rag_chain = get_rag_chain()
except Exception as e:
    st.error(f"Erro ao carregar o Chef: {e}")
    st.stop()

# Histórico Visual (Frontend)
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Olá! Sou o NutriChef 👨‍🍳. O que vamos cozinhar hoje?"}]

# Renderiza as mensagens antigas com AVATARS
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user", avatar="👤").write(msg["content"])
    else:
        st.chat_message("assistant", avatar="👨‍🍳").write(msg["content"])

# Caixa de Entrada
if prompt := st.chat_input("Digite sua dúvida culinária..."):
    # Usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").write(prompt)

    # Resposta do Bot
    with st.chat_message("assistant", avatar="👨‍🍳"):
        with st.spinner("O Chef está consultando o livro... 📖"):
            response = rag_chain.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": "usuario_padrao"}}
            )
            bot_reply = response["answer"]
            st.write(bot_reply)
            
            # Fontes Estilizadas
            with st.expander("📚 Ver receita original (Fonte)"):
                for i, doc in enumerate(response["context"]):
                    origem = doc.metadata.get("source", "PDF").split("/")[-1] # Pega só o nome do arquivo
                    pagina = doc.metadata.get("page", "?")
                    st.markdown(f"**📄 {origem}** - *Página {pagina}*")
                    st.caption(f"...{doc.page_content[:150].replace(chr(10), ' ')}...") # Remove quebras de linha
                    st.divider()

    # Salva histórico
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})