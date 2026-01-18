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

# Imports Manuais (LCEL)
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str

# RAG
from rag_engine import get_retriever

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="iChef App",
    page_icon="👨‍🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CSS PERSONALIZADO (Visual Premium) ---
st.markdown("""
<style>
    /* Fundo e Fontes */
    .stApp { background-color: #f8f9fa; }
    h1 { color: #1E4D2B; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    
    /* Balões de Chat */
    .stChatMessage { padding: 1rem; border-radius: 10px; margin-bottom: 10px; }
    div[data-testid="stChatMessageContent"] { background-color: #ffffff; border-radius: 15px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #e9ecef; }
    
    /* Botões */
    .stButton>button { 
        border-radius: 20px; 
        border: 1px solid #1E4D2B; 
        color: #1E4D2B; 
        background-color: transparent;
        transition: all 0.3s;
    }
    .stButton>button:hover { 
        background-color: #1E4D2B; 
        color: white; 
        border: 1px solid #1E4D2B;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()

# --- 3. FERRAMENTAS ---

@tool
def calculadora_culinaria(expressao: str) -> str:
    """Realiza cálculos matemáticos (soma, multiplicação, divisão). Ex: '200 * 2'."""
    try:
        safe_expr = expressao.replace('^', '**').replace('x', '*')
        if not all(c in "0123456789.+-*/() " for c in safe_expr):
             return "Erro: Caracteres inválidos."
        return str(eval(safe_expr))
    except:
        return "Erro ao calcular."

def get_tools():
    retriever = get_retriever()
    retriever.search_kwargs['k'] = 3
    
    search_book = create_retriever_tool(
        retriever,
        "search_nutribook",
        "Pesquisa receitas no PDF Nutribook."
    )
    
    search_web = DuckDuckGoSearchRun(
        name="search_web",
        description="Pesquisa receitas ou informações na Internet."
    )
    
    return [search_book, search_web, calculadora_culinaria]

# --- 4. AGENTE COM CONTEXTO DINÂMICO ---

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state:
        st.session_state[session_id] = ChatMessageHistory()
    return st.session_state[session_id]

@st.cache_resource
def get_agent():
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    tools = get_tools()
    
    # Template recebe 'diet_restrictions' dinamicamente
    template = """
    Você é o iChef, um Chef Profissional e Nutricionista. 👨‍🍳
    
    ⚠️ PERFIL DO USUÁRIO (RESPEITE RIGOROSAMENTE):
    Restrições/Preferências: {diet_restrictions}
    
    Ferramentas: {tools}
    
    Use o formato ReAct:
    Thought: ...
    Action: [{tool_names}]
    Action Input: ...
    Observation: ...
    
    Final Answer: [Sua resposta final em Português]
    
    Regras de Resposta:
    1. Seja acolhedor e use emojis.
    2. Se o usuário tiver restrição (ex: Vegano), NUNCA sugira carne/ovos/leite, ou sugira substitutos.
    3. Receitas devem ter listas de ingredientes e passo-a-passo.
    
    Histórico: {chat_history}
    Pergunta: {input}
    Thought: {agent_scratchpad}
    """
    
    prompt = PromptTemplate.from_template(template)
    
    prompt = prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools])
    )
    
    llm_with_stop = llm.bind(stop=["\nObservation"])
    
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
            "chat_history": lambda x: x.get("chat_history", []),
            "diet_restrictions": lambda x: x.get("diet_restrictions", "Nenhuma") # <--- Injeção de Contexto
        }
        | prompt
        | llm_with_stop
        | ReActSingleInputOutputParser()
    )
    
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, return_intermediate_steps=True
    )
    
    return RunnableWithMessageHistory(
        agent_executor, get_session_history, input_messages_key="input", history_messages_key="chat_history"
    )

# --- 5. INTERFACE (SIDEBAR & MAIN) ---

# --- SIDEBAR: CONTROLES ---
with st.sidebar:
    st.title("Configurações")
    st.markdown("---")
    
    # Filtros de Dieta
    st.subheader("🍏 Preferências")
    dieta = st.multiselect(
        "Restrições Alimentares:",
        ["Vegano", "Vegetariano", "Sem Glúten", "Sem Lactose", "Low Carb", "Sem Açúcar"]
    )
    
    # Formata a string para o Agente
    diet_str = ", ".join(dieta) if dieta else "Nenhuma restrição específica."
    
    st.markdown("---")
    
    # Botão de Download
    if st.session_state.get("messages"):
        chat_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
        st.download_button("📥 Baixar Conversa", chat_text, file_name="receitas_nutrichef.txt")

    if st.button("🗑️ Limpar conversa"):
        st.session_state.messages = []
        st.session_state["session_id"] = ChatMessageHistory()
        st.rerun()

# --- MAIN: CHAT ---
st.title("👨‍🍳 iChef")
st.caption(f"Seu assistente culinário pessoal. Perfil ativo: **{diet_str}**")

# Botões de Ação Rápida (Sugestões)
col1, col2, col3 = st.columns(3)
acao_rapida = None

if col1.button("🎲 Receita Surpresa"):
    acao_rapida = "Me sugira uma receita saudável e criativa baseada no meu perfil."
if col2.button("🍰 Doce Fit"):
    acao_rapida = "Quero uma receita de sobremesa saudável e com poucas calorias."
if col3.button("🍱 Marmita da Semana"):
    acao_rapida = "Me dê 3 ideias de pratos para marmitas congeladas."

# Inicializa histórico
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Configure suas preferências ao lado e me diga: o que vamos cozinhar hoje? 🥗"}]

# Renderiza histórico
for msg in st.session_state.messages:
    avatar = "👨‍🍳" if msg["role"] == "assistant" else "👤"
    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

# Lógica de Input (Texto ou Botão)
prompt = st.chat_input("Digite sua dúvida...")
if acao_rapida:
    prompt = acao_rapida

if prompt:
    # Usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").write(prompt)

    # Resposta
    with st.chat_message("assistant", avatar="👨‍🍳"):
        with st.spinner("Cozinhando ideias..."):
            try:
                agent = get_agent()
                
                # Invocação passando as restrições da sidebar
                response = agent.invoke(
                    {"input": prompt, "diet_restrictions": diet_str},
                    config={"configurable": {"session_id": "session_id"}}
                )
                
                output = response["output"]
                st.write(output)
                
                # Detalhes técnicos
                steps = response.get("intermediate_steps", [])
                if steps:
                    with st.expander("🔎 Ver Ingredientes da Pesquisa (Fontes)"):
                        for i, (action, result) in enumerate(steps):
                            st.caption(f"🔨 **Ferramenta:** `{action.tool}` | **Input:** `{action.tool_input}`")
                            st.code(str(result)[:300] + "...") # Preview curto
                            st.divider()

                st.session_state.messages.append({"role": "assistant", "content": output})
                
            except Exception as e:
                st.error(f"Erro: {e}")
                st.info("Tente reformular.")