import streamlit as st
import os
from dotenv import load_dotenv

# --- IMPORTS DO LANGCHAIN ---
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun

# Imports para construção manual do Agente (LCEL)
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str

# Import do motor de busca (RAG)
from rag_engine import get_retriever

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Chef Agente",
    page_icon="👨‍🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS PERSONALIZADO ---
st.markdown("""
<style>
    .stApp { background-color: #f4f7f6; }
    h1 { color: #2E8B57; font-family: 'Helvetica', sans-serif; text-align: center; }
    .stChatInput { border-radius: 20px; }
    .stButton>button { background-color: #2E8B57; color: white; border-radius: 20px; border: none; }
    .stButton>button:hover { background-color: #3CB371; }
    div[data-testid="stExpander"] div[role="button"] p { font-size: 1.1rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

load_dotenv()

# --- 3. DEFINIÇÃO DAS FERRAMENTAS (TOOLS) 🛠️ ---

@tool
def calculadora_culinaria(expressao: str) -> str:
    """
    Realiza cálculos matemáticos. Use para ajustar quantidades de ingredientes.
    """
    try:
        # Segurança básica
        safe_expr = expressao.replace('^', '**').replace('x', '*')
        if not all(c in "0123456789.+-*/() " for c in safe_expr):
             return "Erro: Apenas números e operações matemáticas simples são permitidos."
        return str(eval(safe_expr))
    except Exception:
        return "Erro ao calcular."

def get_tools():
    retriever = get_retriever()
    retriever.search_kwargs['k'] = 3 # Limite de leitura para não travar
    
    # Tool 1: O Livro (PDF)
    search_book = create_retriever_tool(
        retriever,
        "search_nutribook",
        "Pesquisa receitas EXCLUSIVAS do livro Nutribook. Use sempre que o usuário perguntar sobre uma comida do livro."
    )
    
    # Tool 2: A Internet (DuckDuckGo)
    search_web = DuckDuckGoSearchRun(
        name="search_web",
        description="Pesquisa na internet por receitas ou informações que NÃO estão no livro."
    )
    
    return [search_book, search_web, calculadora_culinaria]

# --- 4. CONFIGURAÇÃO DO AGENTE (CONSTRUÇÃO MANUAL/LCEL) 🧠 ---

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state:
        st.session_state[session_id] = ChatMessageHistory()
    return st.session_state[session_id]

@st.cache_resource
def get_agent():
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    tools = get_tools()
    
    template = """
    Você é o Chef, um assistente culinário amigável e especialista. 👨‍🍳
    Sua missão é ajudar o usuário com receitas deliciosas e ajustes precisos.
    
    Ferramentas disponíveis:
    {tools}
    
    Para usar uma ferramenta, você DEVE usar o formato:
    
    Thought: preciso verificar X
    Action: [uma das ferramentas: {tool_names}]
    Action Input: [entrada]
    Observation: [resultado]
    
    Quando você tiver a resposta final, use o formato:
    
    Thought: agora tenho a resposta
    Final Answer: [Sua resposta final]
    
    ⚠️ IMPORTANTE PARA A RESPOSTA FINAL:
    1. Use um tom acolhedor e emojis (🥗, 🥘, 🔥, 🍰).
    2. Se for uma receita, NUNCA use texto corrido. Use LISTAS com marcadores ou números.
    3. Separe claramente "Ingredientes" de "Modo de Preparo".
    4. Responda sempre em Português do Brasil.
    
    Comece!
    
    Histórico:
    {chat_history}
    
    Pergunta: {input}
    Thought: {agent_scratchpad}
    """
    
    prompt = PromptTemplate.from_template(template)
    
    # Injeção das ferramentas
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools])
    )
    
    llm_with_stop = llm.bind(stop=["\nObservation"])
    
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
            "chat_history": lambda x: x.get("chat_history", [])
        }
        | prompt
        | llm_with_stop
        | ReActSingleInputOutputParser()
    )
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )
    
    return RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history"
    )

# --- 5. INTERFACE (FRONTEND) ---

with st.sidebar:
    st.title("Chef Agente")
    st.markdown("**Status:** Fogão ligado! 🔥")
    st.markdown("Este agente pensa antes de responder! Ele pode consultar o PDF e pesquisar na Web.")
    st.markdown("---")
    if st.button("🗑️ Limpar Memória"):
        st.session_state.messages = []
        st.session_state["session_id"] = ChatMessageHistory()
        st.rerun()

st.title("👨‍🍳 Chef Inteligente")
st.caption("Pergunte sobre receitas, peça para dobrar porções ou busque novidades na web!")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Sou seu Chef Pessoal. O que vamos preparar hoje? 🥗"}]

for msg in st.session_state.messages:
    avatar = "👨‍🍳" if msg["role"] == "assistant" else "👤"
    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

if prompt := st.chat_input("Ex: Receita de Coxinha Fit ou dobre a receita de bolo..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").write(prompt)

    with st.chat_message("assistant", avatar="👨‍🍳"):
        with st.spinner("O Chef está na cozinha... 🍳"):
            try:
                agent = get_agent()
                response = agent.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": "session_id"}}
                )
                
                output = response["output"]
                st.write(output)
                
                # Expander para ver as fontes
                steps = response.get("intermediate_steps", [])
                if steps:
                    with st.expander("🕵️‍♂️ Ver Fontes e Raciocínio"):
                        for i, (action, result) in enumerate(steps):
                            st.markdown(f"**Passo {i+1}:** Usou `{action.tool}`")
                            if action.tool == "search_nutribook":
                                st.info("📖 **Livro PDF**")
                                st.caption(f"...{str(result)[:400]}...")
                            elif action.tool == "search_web":
                                st.info("🌍 **Internet**")
                                st.caption(f"...{str(result)[:400]}...")
                            elif action.tool == "calculadora_culinaria":
                                st.success(f"🧮 **Cálculo:** {action.tool_input} = {result}")
                            st.divider()

                st.session_state.messages.append({"role": "assistant", "content": output})
                
            except Exception as e:
                st.error(f"Erro: {e}")
                st.info("Tente limpar a memória.")