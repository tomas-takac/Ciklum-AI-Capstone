## Step 1: Loading the PDF Data

from langchain_community.document_loaders import PyPDFLoader
import glob
import os

# 1. Define the path to your PDF file

# Find any PDF file in the resources/pdf directory
pdf_files = glob.glob("./resources/pdf/*.pdf")

if not pdf_files:
    raise FileNotFoundError("No PDF files found in ./resources/pdf/")

PDF_PATH = pdf_files[0]

# Check if the file exists before trying to load it
if not os.path.exists(PDF_PATH):
    print(f"Error: The file '{PDF_PATH}' was not found.")
else:
    # 2. Initialize the loader
    # This prepares the tool to read your PDF
    loader = PyPDFLoader(PDF_PATH)

    # 3. Load the document and code
    # This reads the content and returns a list of 'Document' objects (one per page)
    pages = loader.load()

    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    # Define the root of your project
    REPO_PATH = "./"

    # 1. Load Python files correctly using DirectoryLoader
    python_loader = DirectoryLoader(
        REPO_PATH,
        glob="**/*.py",
        loader_cls=TextLoader
    )
    code_docs = python_loader.load()

    # 2. Load Config and Style files
    config_files = ["Dockerfile", "docker-compose.yaml", "requirements.txt", "style_guide.txt"]
    config_docs = []

    for file in config_files:
        if os.path.exists(file):
            print(f"Loading config/style file: {file}")
            config_docs.extend(TextLoader(file).load())

    # 3. Combine everything: PDF + Source Code + Configs + Style Guide
    all_documents = pages + code_docs + config_docs

    # 4. Verification (Optional but helpful)
    print(f"Successfully loaded {len(all_documents)} pages.")

    # You can inspect the content of the first page:
    print("\n--- Content of Page 1 (Snippet) ---")
    print(all_documents[0].page_content[:500] + "...")
    print(f"Source: {all_documents[0].metadata}")


## Step 2: Splitting the Text (Chunking)

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Initialize the Text Splitter
# This splitter tries to keep related text together by using different separators (like '\n\n', '\n', ' ', '')
# chunk_size: The maximum size of each chunk (e.g., 1000 characters)
# chunk_overlap: How many characters to overlap between chunks to maintain context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

# 2. Split the loaded document all_documents into chunks
chunks = text_splitter.split_documents(all_documents)

# 3. Verification
print(f"\n--- Splitting Results ---")
print(f"Original all_documents loaded: {len(all_documents)}")
print(f"Total chunks created: {len(chunks)}")
print(f"Example Chunk 1 Source: {chunks[0].metadata}")


## Step 3: Embedding and Storing Data in ChromaDB
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- Start where Step 2 left off, assuming you have the 'chunks' list ---

# 1. Initialize the Embedding Model
# We use the Sentence Transformer model you installed (all-MiniLM-L6-v2 is standard)
# This model converts text into vectors.
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Define the path to your vector database
DB_PATH = "./my_rag_db"

import shutil

# Delete contents instead of the directory itself if exists
if os.path.exists(DB_PATH):
    print(f"Cleaning up old database contents at {DB_PATH}...")
    for filename in os.listdir(DB_PATH):
        file_path = os.path.join(DB_PATH, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    print("Database contents cleared.")

print("\n--- Creating NEW Vector Database ---")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_function,
    persist_directory=DB_PATH
)

# 3. Verification
print("ChromaDB vector store created successfully.")
print(f"Total documents indexed in Chroma: {vectorstore._collection.count()}")


## Step 4: Setting up the LLM and Retriever (Modern LCEL Version)

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize the Local LLM (Ollama)
LLM_MODEL = "llama3"
import os
ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm = OllamaLLM(model=LLM_MODEL, base_url=ollama_url)

# 2. Create the Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 3. Define the Prompt Template
template = """Use the given context to answer the question. 
If you don't know the answer, say you don't know.

Context: {context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# 4. Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 5. Create the RAG Chain using LCEL
rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
)

# 6. Interactive Chat Loop
print("\n" + "="*50)
print("RAG System Ready! Type your questions below.")
print("Type 'exit' or 'quit' to stop.")
print("="*50)

while True:
    query = input("\nUser Question: ")

    if query.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break

    if not query.strip():
        continue

    print("Thinking...")
    try:
        # STEP 1: GENERATION
        draft = rag_chain.invoke(query)

        # STEP 2: REFLECTION (The "Agentic" part)
        reflection_query = f"""
        Review this LinkedIn post draft:
        ---
        {draft}
        ---
        Does it meet these requirements from the style guide?
        - 5-7 sentences long?
        - Mentions Ciklum AI Academy?
        - Tags @Ciklum?
        
        If not, rewrite it to be perfect. Otherwise, return the draft.
        """
        final_post = llm.invoke(reflection_query)
        print("\nFinal Reflected Answer:\n", final_post)
    except Exception as e:
        print(f"An error occurred: {e}")
