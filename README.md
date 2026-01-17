# 👨‍🍳 ChefBot IA - Assistente Culinário Inteligente

Este projeto é um assistente virtual baseado em RAG (Retrieval-Augmented Generation) que ensina receitas saudáveis. Ele lê um livro de receitas em PDF ("Nutribook"), entende o contexto e responde perguntas do usuário de forma natural através de uma interface web.

## 🚀 Tecnologias Utilizadas

* **Python 3.10+**
* **LangChain:** Para orquestração da IA e fluxo de RAG.
* **Streamlit:** Para a interface web interativa.
* **Groq API (Llama 3):** Modelo de linguagem de alta performance e baixa latência.
* **ChromaDB:** Banco de dados vetorial para busca semântica.
* **PyPDF:** Processamento de documentos PDF.

## ⚙️ Como Instalar e Rodar

### 1. Clone o repositório
```bash
2. Crie um ambiente virtual
Bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
3. Instale as dependências
Bash
pip install -r requirements.txt
4. Configure as chaves de acesso
Crie um arquivo .env na raiz do projeto e adicione sua chave da Groq:

Plaintext
GROQ_API_KEY=sua_chave_aqui_gsk_...
5. Prepare o Banco de Dados (Ingestão)
Coloque seu PDF na pasta documents/ e rode:

Bash
python src/rag_engine.py
6. Inicie a Aplicação
Bash
streamlit run src/app.py
🛠️ Solução de Problemas (Windows)
Se encontrar erros relacionados ao SQLite3 ou ChromaDB, pode ser necessário atualizar a DLL do SQLite no Windows.

Baixe a versão mais recente do sqlite3.dll no site oficial.

Coloque o arquivo dentro da pasta venv/Scripts/.

autor
Desenvolvido por Breno Crespo.