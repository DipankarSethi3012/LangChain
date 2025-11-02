# -----------------------------------------------------------
# Step 0: Import all Required Modules
# -----------------------------------------------------------

# HuggingFaceEmbeddings: Used to convert text chunks into numerical vector embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# PyPDFLoader: Used to load and read PDF files in LangChain format
from langchain_community.document_loaders import PyPDFLoader

# FAISS: Facebook AI Similarity Search - a vector database to store and query embeddings locally
from langchain_community.vectorstores import FAISS

# RecursiveCharacterTextSplitter: Splits large text into smaller, overlapping chunks for better embedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document schema: Represents text + metadata in LangChain
from langchain_core.documents import Document

# dotenv: Used to load environment variables from a .env file
from dotenv import load_dotenv

# os: Used for checking file existence and path handling
import os


# -----------------------------------------------------------
# Step 1: Load Environment Variables (if any)
# -----------------------------------------------------------

# This line loads all the environment variables from a .env file into the system environment
load_dotenv()


# -----------------------------------------------------------
# Step 2: Load the PDF Document
# -----------------------------------------------------------

# Path to the PDF file that we want to process
# Use 'r' before the string to make it a raw string (to avoid escape sequence errors like \n, \t, etc.)
pdf_path = ""

# Check if the PDF file actually exists at the given path
# If not found, raise an error message
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f" PDF not found at: {pdf_path}")

# Create a PDF loader object using the PyPDFLoader class
loader = PyPDFLoader(pdf_path)

# Load the entire PDF document
# The loader reads all the text from the PDF and returns it as LangChain 'Document' objects
documents = loader.load()

# Print confirmation that the PDF has been successfully loaded
print(f"Loaded {len(documents)} document(s) from PDF.")


# -----------------------------------------------------------
# Step 3: Split the Document into Smaller Chunks
# -----------------------------------------------------------

# The RecursiveCharacterTextSplitter divides long text into smaller parts
# chunk_size → maximum number of characters per chunk
# chunk_overlap → overlapping characters between consecutive chunks (maintains context)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Split the loaded PDF documents into smaller chunks
chunks = splitter.split_documents(documents)

# Print how many text chunks were created
print(f"Split into {len(chunks)} text chunks.")


# -----------------------------------------------------------
# Step 4: Create Embeddings using HuggingFace Model
# -----------------------------------------------------------

# This embedding model converts text into high-dimensional numeric vectors
# Each text chunk is transformed into a numerical vector representation
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Print confirmation that the embedding model is ready
print("Embedding model loaded successfully.")


# -----------------------------------------------------------
# Step 5: Store the Embeddings in FAISS (Local Vector Database)
# -----------------------------------------------------------

# Here we create a FAISS index from the text chunks and their corresponding embeddings
# This allows efficient semantic similarity search later
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save the FAISS index locally on the disk (so that it can be loaded again later)
vectorstore.save_local("faiss_index")

# Print confirmation that FAISS index has been created and saved
print("Embeddings created and saved in FAISS vector database.")


# -----------------------------------------------------------
# Step 6: Define Semantic Search Function
# -----------------------------------------------------------

def semantic_search(query, k=3, threshold=1.5):
    """
    This function performs semantic search on the stored FAISS vector database.
    It finds the most relevant chunks from the document for a given user query.
    """

    # Load the locally stored FAISS index along with the embedding model
    new_vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Perform similarity search based on the query text
    # 'k' represents how many top similar chunks we want to retrieve
    results = new_vectorstore.similarity_search_with_score(query, k=k)

    # If no results are found, return a message
    if not results:
        return "No relevant documents found."

    # Prepare a list to store the relevant chunks of text
    relevant_texts = []

    # Iterate through each result (document + similarity score)
    for doc, score in results:
        # FAISS uses cosine similarity; smaller score = more similar
        # We include text chunks that are within the threshold range
        if score < threshold * 2:
            relevant_texts.append(doc.page_content)

    # If no relevant text was found within threshold
    if not relevant_texts:
        return "No relevant information found."

    # Join all relevant chunks into a single string with separators for clarity
    return "\n\n---\n\n".join(relevant_texts)


# -----------------------------------------------------------
# Step 7: Interactive Q&A Loop for User Queries
# -----------------------------------------------------------

# This loop keeps running until the user types 'exit'
# It allows the user to ask questions repeatedly
while True:
    # Take input question from the user
    question = input("\nEnter your question (or 'exit' to quit): ")

    # Check if the user wants to exit the program
    if question.lower() == 'exit':
        print("Exiting Semantic Search. Goodbye!")
        break

    # Call the semantic_search() function with the user's question
    answer = semantic_search(question)

    # Print the answer to the terminal
    print("\nAnswer:\n", answer)
