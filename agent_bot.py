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
