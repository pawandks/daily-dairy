import customtkinter as ctk
import threading
import queue
import io
import sys
import os
from typing import Dict, Any, List
from datetime import datetime

# Core LangChain imports from your script
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.tools import Tool
# Other imports
import tiktoken
from duckduckgo_search import DDGS
import requests
from urllib.parse import urlparse

# =============================================================================
# YOUR ORIGINAL LangChainAgentSetup CLASS (with a minor modification)
# =============================================================================
class LangChainAgentSetup:
    """Complete LangChain setup with Gemini Flash 2.5 and tools"""
    
    # ... [PASTE YOUR ENTIRE LangChainAgentSetup CLASS HERE] ...
    # ... I will paste it below for completeness ...
    def __init__(self, google_api_key: str = None):
        """Initialize the LangChain agent setup"""
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable.")
        
        self.llm = None
        self.embeddings = None
        self.memory = None
        self.tools = []
        self.agent = None
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        print("✅ LangChain Agent Setup initialized")
    
    def setup_llm(self):
        """Step 2: Setup Gemini Flash 2.
        5 LLM"""
        print("\n🔧 Setting up Gemini Flash 1.5 LLM...")
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.google_api_key,
                temperature=0.7
            )
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.google_api_key
            )
            test_response = self.llm.invoke("Hello! Confirm you are working.")
            print(f"✅ LLM Test Response: {test_response.content[:50]}...")
            return True
        except Exception as e:
            print(f"❌ Error setting up LLM: {e}")
            return False
    
    def setup_memory(self):
        """Step 3: Setup conversation memory"""
        print("\n🧠 Setting up conversation memory...")
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="output"
        )
        print("✅ Memory setup complete")
        return True
    
    def create_search_tools(self):
        """Step 4: Create search tools using DDGS"""
        print("\n🔍 Creating search tools...")
        ddg_search = DuckDuckGoSearchAPIWrapper()
        search_tool = Tool(
            name="web_search",
            description="Search the web for current information. Input should be a search query string.",
            func=ddg_search.run
        )
        def custom_ddgs_search(query: str) -> str:
            """Custom DDGS search implementation"""
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=3))
                    if not results: return "No search results found."
                    return "\n".join([f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}" for r in results])
            except Exception as e:
                return f"Search error: {str(e)}"
        
        custom_search_tool = Tool(
            name="ddgs_search",
            description="Advanced DuckDuckGo search with formatted results. Input should be a search query.",
            func=custom_ddgs_search
        )
        self.tools.extend([search_tool, custom_search_tool])
        print("✅ Search tools created")
        return True

    def create_utility_tools(self):
        """Step 5: Create utility tools"""
        print("\n🛠️ Creating utility tools...")
        def count_tokens(text: str) -> str:
            return f"Token count: {len(self.encoding.encode(text))}"
        
        token_counter = Tool(
            name="token_counter",
            description="Count the number of tokens in a given text.",
            func=count_tokens
        )
        def get_current_time(query: str = "") -> str:
            return f"Current date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        time_tool = Tool(
            name="current_time",
            description="Get the current date and time.",
            func=get_current_time
        )
        def validate_url(url: str) -> str:
            try:
                parsed = urlparse(url)
                if not parsed.scheme or not parsed.netloc: return f"Invalid URL format: {url}"
                response = requests.head(url, timeout=5)
                return f"URL {url} is accessible (Status: {response.status_code})"
            except Exception as e:
                return f"URL validation failed: {str(e)}"
        
        url_validator = Tool(
            name="url_validator",
            description="Validate and check if a URL is accessible.",
            func=validate_url
        )
        self.tools.extend([token_counter, time_tool, url_validator])
        print("✅ Utility tools created")
        return True

    def create_agent(self):
        """Step 6: Create the agent with all tools"""
        print("\n🤖 Creating agent...")
        try:
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self.memory,
                verbose=True,
                max_iterations=4,
                early_stopping_method="generate",
                handle_parsing_errors=True, # Added for robustness
            )
            print("✅ Agent created successfully")
            print(f"📋 Available tools: {[tool.name for tool in self.tools]}")
            return True
        except Exception as e:
            print(f"❌ Error creating agent: {e}")
            return False

    def run_agent(self, query: str) -> str:
        """Step 7: Run the agent with a query"""
        print(f"\n🏃 Running agent with query: '{query}'")
        try:
            # Use invoke for more structured output if available, else run
            if hasattr(self.agent, 'invoke'):
                response = self.agent.invoke({"input": query})
                return response.get("output", "No output found.")
            else:
                return self.agent.run(query)

        except Exception as e:
            print(f"Agent execution error: {e}")
            return f"Agent error: I encountered a problem processing your request. Please try again."

# =============================================================================
# CAPTURES PRINT OUTPUT FOR THE GUI
# =============================================================================
class PrintLogger(io.TextIOBase):
    def __init__(self, log_queue: queue.Queue):
        self.log_queue = log_queue
        self.buffer = ''

    def write(self, text):
        self.buffer += text
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            for line in lines[:-1]:
                self.log_queue.put(line)
            self.buffer = lines[-1]
        return len(text)

    def flush(self):
        if self.buffer:
            self.log_queue.put(self.buffer)
            self.buffer = ''

# =============================================================================
# THE MAIN GUI APPLICATION
# =============================================================================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # --- App Setup ---
        self.title("Gemini Flash Agent GUI")
        self.geometry("1000x700")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- State Management ---
        self.langchain_setup = None
        self.log_queue = queue.Queue()
        self.print_logger = PrintLogger(self.log_queue)
        
        # --- Create Widgets ---
        self.create_sidebar()
        self.create_main_content()
        self.show_setup_view()
        
        # Start processing log queue
        self.after(100, self.process_log_queue)

    # --- Widget Creation ---
    def create_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=2, sticky="nsw")

        logo_label = ctk.CTkLabel(self.sidebar_frame, text="Agent Status", font=ctk.CTkFont(size=20, weight="bold"))
        logo_label.pack(pady=(20, 10))

        # Status Indicators
        self.status_indicators = {
            "LLM": self.create_status_indicator(self.sidebar_frame, "LLM Ready"),
            "Memory": self.create_status_indicator(self.sidebar_frame, "Memory Ready"),
            "Tools": self.create_status_indicator(self.sidebar_frame, "Tools Ready"),
            "Agent": self.create_status_indicator(self.sidebar_frame, "Agent Ready")
        }
        
        # Available Tools Display
        tools_label = ctk.CTkLabel(self.sidebar_frame, text="Available Tools:", anchor="w", font=ctk.CTkFont(weight="bold"))
        tools_label.pack(pady=(20, 5), padx=20, fill="x")
        self.tools_textbox = ctk.CTkTextbox(self.sidebar_frame, height=150, activate_scrollbars=False)
        self.tools_textbox.pack(pady=5, padx=20, fill="x")
        self.tools_textbox.configure(state="disabled")

    def create_status_indicator(self, parent, text):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=5)
        label = ctk.CTkLabel(frame, text=text, anchor="w")
        label.pack(side="left", fill="x", expand=True)
        # Using a label as a colored circle
        indicator = ctk.CTkLabel(frame, text="●", width=15, text_color="gray") 
        indicator.pack(side="right")
        return indicator

    def create_main_content(self):
        # Container for switching views
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

        # 1. Setup View
        self.setup_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.setup_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.setup_frame, text="Welcome!", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(20, 10))
        ctk.CTkLabel(self.setup_frame, text="Enter your Google AI API Key to begin.").pack(pady=5)
        self.api_key_entry = ctk.CTkEntry(self.setup_frame, placeholder_text="Enter Google API Key here...", width=400, show="*")
        self.api_key_entry.pack(pady=10)
        self.start_button = ctk.CTkButton(self.setup_frame, text="Start Agent", command=self.start_setup_thread, width=150)
        self.start_button.pack(pady=10)
        self.setup_log_textbox = ctk.CTkTextbox(self.setup_frame, height=300)
        self.setup_log_textbox.pack(pady=10, fill="both", expand=True)
        self.setup_log_textbox.configure(state="disabled", wrap="word")

        # 2. Chat View
        self.chat_frame = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.chat_frame.grid_columnconfigure(0, weight=1)
        self.chat_frame.grid_rowconfigure(0, weight=1) # Chat history resizes
        self.chat_frame.grid_rowconfigure(1, weight=0) # Details accordion is fixed
        self.chat_frame.grid_rowconfigure(2, weight=0) # Input is fixed

        self.chat_history_frame = ctk.CTkScrollableFrame(self.chat_frame, label_text="Conversation")
        self.chat_history_frame.grid(row=0, column=0, sticky="nsew", pady=(0,10))
        
        self.thinking_frame = ctk.CTkFrame(self.chat_frame, height=100)
        self.thinking_frame.grid(row=1, column=0, sticky="ew", pady=(0,10))
        self.thinking_frame.grid_remove() # Hide it initially
        
        ctk.CTkLabel(self.thinking_frame, text="Agent Reasoning:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(5,0))
        self.thinking_textbox = ctk.CTkTextbox(self.thinking_frame, wrap="word", height=120)
        self.thinking_textbox.pack(fill="both", expand=True, padx=10, pady=5)
        self.thinking_textbox.configure(state="disabled")

        input_frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        input_frame.grid(row=2, column=0, sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)
        
        self.user_entry = ctk.CTkEntry(input_frame, placeholder_text="Ask the agent anything...")
        self.user_entry.grid(row=0, column=0, sticky="ew", padx=(0,10))
        self.send_button = ctk.CTkButton(input_frame, text="Send", width=100, command=self.send_message_thread)
        self.send_button.grid(row=0, column=1)

        self.user_entry.bind("<Return>", lambda event: self.send_message_thread())
        
    def show_setup_view(self):
        self.chat_frame.grid_remove()
        self.setup_frame.grid(row=0, column=0, sticky="nsew")
        self.update_status_indicator("All", False)
        
    def show_chat_view(self):
        self.setup_frame.grid_remove()
        self.chat_frame.grid(row=0, column=0, sticky="nsew")

    # --- Backend Logic & Threading ---
    def start_setup_thread(self):
        api_key = self.api_key_entry.get()
        if not api_key:
            self.log_to_gui("ERROR: API Key cannot be empty.")
            return

        self.start_button.configure(state="disabled", text="Initializing...")
        self.setup_log_textbox.configure(state="normal")
        self.setup_log_textbox.delete("1.0", "end")
        self.setup_log_textbox.configure(state="disabled")

        thread = threading.Thread(target=self.run_setup_process, args=(api_key,), daemon=True)
        thread.start()
        
    def run_setup_process(self, api_key):
        sys.stdout = self.print_logger
        
        try:
            self.langchain_setup = LangChainAgentSetup(google_api_key=api_key)
            
            if self.langchain_setup.setup_llm():
                self.update_status_indicator("LLM", True)
            else: raise Exception("LLM Setup Failed")

            if self.langchain_setup.setup_memory():
                self.update_status_indicator("Memory", True)
            else: raise Exception("Memory Setup Failed")

            if self.langchain_setup.create_search_tools() and self.langchain_setup.create_utility_tools():
                self.update_status_indicator("Tools", True)
                # Update tools list in GUI
                tool_names = "\n".join([f"• {tool.name}" for tool in self.langchain_setup.tools])
                self.tools_textbox.configure(state="normal")
                self.tools_textbox.delete("1.0", "end")
                self.tools_textbox.insert("1.0", tool_names)
                self.tools_textbox.configure(state="disabled")
            else: raise Exception("Tool Creation Failed")

            if self.langchain_setup.create_agent():
                self.update_status_indicator("Agent", True)
                self.log_to_gui("\n🎉 Setup successful! You can now chat with the agent.")
                self.after(1000, self.show_chat_view) # Switch view on main thread
            else: raise Exception("Agent Creation Failed")
            
        except Exception as e:
            print(f"FATAL ERROR: {e}")
            self.log_to_gui(f"FATAL ERROR during setup. Please check the key and your connection.")
            self.update_status_indicator("All", False)
        finally:
            sys.stdout = sys.__stdout__ # Restore stdout
            self.start_button.configure(state="normal", text="Start Agent")

    def send_message_thread(self):
        query = self.user_entry.get()
        if not query:
            return

        self.add_chat_bubble(query, "user")
        self.user_entry.delete(0, "end")
        self.send_button.configure(state="disabled")

        # Show "thinking" frame and clear previous thoughts
        self.thinking_frame.grid()
        self.thinking_textbox.configure(state="normal")
        self.thinking_textbox.delete("1.0", "end")
        self.thinking_textbox.insert("1.0", "Agent is thinking...")
        self.thinking_textbox.configure(state="disabled")

        thread = threading.Thread(target=self.run_agent_process, args=(query,), daemon=True)
        thread.start()

    def run_agent_process(self, query):
        sys.stdout = self.print_logger # Capture agent's thoughts
        
        response = self.langchain_setup.run_agent(query)
        
        sys.stdout = sys.__stdout__ # Restore stdout

        # The response needs to be handled in the main thread
        self.after(0, self.handle_agent_response, response)

    # --- GUI Update Methods (run on main thread) ---
    def process_log_queue(self):
        while not self.log_queue.empty():
            message = self.log_queue.get_nowait()
            if self.setup_frame.winfo_viewable():
                self.log_to_gui(message)
            else: # If chat view is active, log goes to the "thinking" box
                self.log_to_thinking_box(message)
        self.after(100, self.process_log_queue)

    def log_to_gui(self, message):
        self.setup_log_textbox.configure(state="normal")
        self.setup_log_textbox.insert("end", message + "\n")
        self.setup_log_textbox.see("end")
        self.setup_log_textbox.configure(state="disabled")

    def log_to_thinking_box(self, message):
        if "Agent is thinking..." in self.thinking_textbox.get("1.0", "end"):
            # Clear the placeholder text on first log message
            self.thinking_textbox.configure(state="normal")
            self.thinking_textbox.delete("1.0", "end")

        self.thinking_textbox.configure(state="normal")
        self.thinking_textbox.insert("end", message + "\n")
        self.thinking_textbox.see("end")
        self.thinking_textbox.configure(state="disabled")

    def add_chat_bubble(self, text, user_type):
        if user_type == "user":
            bubble = ctk.CTkFrame(self.chat_history_frame, fg_color="#2b2d42")
            bubble.pack(fill="x", padx=(80, 5), pady=5)
            label = ctk.CTkLabel(bubble, text=text, wraplength=600, justify="right", anchor="e")
            label.pack(fill="x", padx=10, pady=10)
        else: # "agent"
            bubble = ctk.CTkFrame(self.chat_history_frame, fg_color="#4f5263")
            bubble.pack(fill="x", padx=(5, 80), pady=5)
            label = ctk.CTkLabel(bubble, text=text, wraplength=600, justify="left", anchor="w")
            label.pack(fill="x", padx=10, pady=10)

        # Auto-scroll to the bottom
        self.after(50, self.chat_history_frame._parent_canvas.yview_moveto, 1.0)
    
    def handle_agent_response(self, response):
        self.add_chat_bubble(response, "agent")
        self.send_button.configure(state="normal")
        # Optional: hide the thinking frame after response is delivered
        # self.thinking_frame.grid_remove()

    def update_status_indicator(self, component_name, is_success):
        color = "lightgreen" if is_success else "gray"
        
        if component_name == "All":
            for indicator in self.status_indicators.values():
                indicator.configure(text_color=color)
            return

        indicator = self.status_indicators.get(component_name)
        if indicator:
            indicator.configure(text_color=color)

# =============================================================================
# RUN THE APP
# =============================================================================
if __name__ == "__main__":
    app = App()
    app.mainloop()