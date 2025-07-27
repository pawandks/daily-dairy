import os, re, json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
import wikipedia

# Load environment variables
load_dotenv()

# Initialize LLM and memory
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Tool: Wikipedia
def wiki_tool(query: str) -> str:
    try:
        return wikipedia.summary(query, sentences=3)
    except:
        return "No wiki result."

# Tool: Calculator
def calc_tool(expr: str) -> str:
    return str(eval(expr)) if re.match(r'^[0-9+\-*/(). ]+$', expr) else "Invalid expression"

# Define tools
tools = [
    Tool("Wikipedia", wiki_tool, "Search wiki"),
    Tool("Calculator", calc_tool, "Evaluate math")
]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# JSON log file path
LOG_FILE = "agent_log.json"

# Function to append a Q&A pair to the JSON file
def save_to_json(question, answer):
    log_entry = {"question": question, "answer": answer}
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r+", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.append(log_entry)
            f.seek(0)
            json.dump(data, f, indent=4)
    else:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([log_entry], f, indent=4)

# Main execution
if __name__ == "__main__":
    questions = [
        "Who is Einstein?",
        "Calculate 15*8",
        "What are your hours?",
        "What is the capital of France?",
        "Calculate (8 + 6) * 3"
    ]

    for q in questions:
        answer = agent.run(q)
        print("Q:", q)
        print("A:", answer, "\n")
        save_to_json(q, answer)

        
# 🧠 RAG-based Question Answering Bot using Gemini 1.5 Flash and Chroma Vectorstore

import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 🔐 Load environment variables (e.g., Gemini API key)
load_dotenv()

# 💬 Initialize LLM (Gemini Flash) and Embedding Model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 📄 Load and split document into chunks
loader = TextLoader("company_docs.txt")  # Make sure this file exists
docs = loader.load()
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

# 🧠 Create vector store from document chunks
# 🔄 This is the core of RAG: combining retrieval with generation
vect = Chroma.from_documents(chunks, embedding=embedding_model, persist_directory="db_conv/")
retriever = vect.as_retriever(k=3)

# 🔗 RAG Chain: Uses retriever + LLM to answer queries
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# 📁 Q&A log file
log_file = "qa_log.json"

# 📝 Save Q&A pairs to JSON file
def save_to_file(question, answer):
    data = {"question": question, "answer": answer}
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
    else:
        logs = []
    logs.append(data)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4)

# 🚀 Run: Ask questions and save answers using RAG
if __name__ == "__main__":
    questions = ["Hours?", "Return policy?", "Premium plan cost?"]
    for q in questions:
        ans = chain.invoke(q)
        print("Q:", q, "\nA:", ans, "\n")
        save_to_file(q, ans)
