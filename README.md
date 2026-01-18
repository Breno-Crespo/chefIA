# 👨‍🍳 NutriChef Agent - Assistente Culinário Inteligente

> **Status:** 🟢 Online | **Versão:** 2.0 (Agente ReAct)

O **NutriChef** não é apenas um chatbot simples. É um **Agente Autônomo** desenvolvido com a arquitetura **ReAct** (Reason + Act). Ele é capaz de raciocinar sobre as perguntas do usuário e decidir autonomamente qual ferramenta usar para resolver o problema.

Diferente de IAs que apenas geram texto, o NutriChef possui uma "caixa de ferramentas" que permite consultar documentos privados, navegar na internet e realizar cálculos matemáticos para ajustar receitas.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-v0.1-green)
![Groq](https://img.shields.io/badge/LLM-Llama%203.3-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)

---

## 🚀 Funcionalidades e Ferramentas

O Agente tem acesso às seguintes ferramentas e decide quando usar cada uma:

| Ferramenta | Descrição |
| :--- | :--- |
| **`search_nutribook`** | Ferramenta de **RAG (Retrieval-Augmented Generation)**. Busca receitas e dicas de saúde exclusivamente no PDF local ("Nutribook"). É a fonte de verdade primária. |
| **`search_web`** | Conecta o Agente à **Internet (DuckDuckGo)**. Usada para encontrar receitas que não estão no livro ou buscar curiosidades culinárias em tempo real. |
| **`calculadora_culinaria`** | Uma engine matemática segura. O Agente usa para **dobrar receitas**, converter medidas (gramas para xícaras) ou dividir porções. |

---

## 🧠 Exemplo de Raciocínio (Chain of Thought)

Quando você pede: *"Tenho 5 visitas. Dobre a receita de coxinha fit do livro."*

O NutriChef executa os seguintes passos nos bastidores:
1.  **Thought:** Preciso primeiro achar a receita original e seus ingredientes.
2.  **Action:** `search_nutribook("coxinha fit")` -> *Retorno: "Rende 2 porções. Ingredientes: 200g de frango..."*
3.  **Thought:** A receita é para 2, o usuário quer para 5 (aprox 2.5x) ou apenas dobrar. Vou calcular o dobro dos ingredientes principais.
4.  **Action:** `calculadora_culinaria("200 * 2")` -> *Retorno: 400*.
5.  **Final Answer:** Gera a resposta final formatada, listando os ingredientes ajustados.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.10+
* **Orquestração:** LangChain (LCEL & Agents)
* **LLM (Cérebro):** Llama 3.3-70b (via Groq API) - *Alta velocidade de inferência*
* **Interface:** Streamlit (com CSS personalizado e Session State)
* **Banco Vetorial:** ChromaDB (Persistência local)
* **Monitoramento:** LangSmith (Tracing de execução e debug)

---

## ⚙️ Instalação e Execução Local

### 1. Clone o repositório
```bash
git clone [https://github.com/SEU-USUARIO/chefbot-ia.git](https://github.com/SEU-USUARIO/chefbot-ia.git)
cd chefbot-ia

2. Crie o ambiente virtual
python -m venv venv

# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
3. Instale as dependências
pip install -r requirements.txt

4. Configuração das Chaves (.env)
Crie um arquivo .env na raiz do projeto com suas credenciais:

Ini, TOML
# Chave da Groq (Obrigatória para o LLM)
GROQ_API_KEY=gsk_sua_chave_aqui...

# Configurações do LangSmith (Opcional - Para monitoramento)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="[https://api.smith.langchain.com](https://api.smith.langchain.com)"
LANGCHAIN_API_KEY=lsv2_sua_chave_langsmith...
LANGCHAIN_PROJECT="chefbot-ia"
5. Prepare o Banco de Dados (Ingestão)
Coloque seu PDF na pasta documents/ e execute:

python src/rag_engine.py
6. Execute a Aplicação

streamlit run src/app.py
📂 Estrutura do Projeto
Plaintext
chefbot-ia/
├── documents/          # Onde ficam os PDFs para ingestão
├── data/               # Banco de dados vetorial (ChromaDB) - Gerado automaticamente
├── src/
│   ├── app.py          # Aplicação principal (Streamlit + Agente)
│   └── rag_engine.py   # Lógica de ingestão e recuperação de documentos
├── .env                # Variáveis de ambiente (Não subir para o Git!)
├── .gitignore          # Arquivos ignorados pelo Git
├── README.md           # Documentação
└── requirements.txt    # Dependências do projeto
👨‍💻 Autor
Desenvolvido por Breno Crespo. Projeto criado para portfólio de Engenharia de IA, explorando Agentes Autônomos e RAG.