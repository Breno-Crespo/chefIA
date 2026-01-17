from langchain_community.document_loaders import PyPDFDirectoryLoader

print("📖 Lendo PDF para teste...")
loader = PyPDFDirectoryLoader("./documents")
docs = loader.load()

# Vamos procurar a palavra chave
encontrou = False
for doc in docs:
    if "Ninhos de Abelha" in doc.page_content:
        print(f"\n✅ ACHEI na página {doc.metadata.get('page', '?')}:")
        print(doc.page_content[:500]) # Mostra os primeiros 500 caracteres
        encontrou = True
        break

if not encontrou:
    print("\n❌ O robô NÃO consegue ler a palavra 'Ninhos de Abelha' neste PDF.")