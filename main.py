# --- Friendly AI Trip Planner in Python ---

# 1. Installation
# Before running, you need to install the Google AI Python library.
# Open your terminal or command prompt and run:
# pip install google-generativeai

import google.generativeai as genai
import json
import getpass

# --- 2. Define the Agent's "Tools" (Functions) ---
# These are the Python functions the AI agent can decide to call.
# The docstrings are important, as the AI uses them to understand what each tool does.

def find_flights(destination: str, origin: str, date: str) -> dict:
    """
    Finds available flights for a given origin, destination, and date.

    Args:
        destination (str): The destination city.
        origin (str): The origin city.
        date (str): The date of travel.

    Returns:
        dict: A dictionary containing flight information or an error message.
    """
    print(f"\n🤖 Trippy is using the 'find_flights' tool...")
    print(f"   -> Searching for flights from {origin} to {destination} for {date}.")
    # In a real application, this would call a flight API.
    # We return mock data for this example.
    return {
        "status": "success",
        "flights": [
            {"flight_number": "AI-202", "departure": "08:00 AM", "arrival": "10:00 AM", "price": "₹7,500"},
            {"flight_number": "6E-555", "departure": "11:30 AM", "arrival": "01:30 PM", "price": "₹6,800"},
        ]
    }

def book_hotel(location: str, nights: int, budget: str) -> dict:
    """
    Finds and books a hotel in a given location for a specified number of nights and budget.

    Args:
        location (str): The city or area for the hotel.
        nights (int): The number of nights to stay.
        budget (str): The budget for the hotel, e.g., 'under ₹10,000'.

    Returns:
        dict: A dictionary containing hotel information or an error message.
    """
    print(f"\n🤖 Trippy is using the 'book_hotel' tool...")
    print(f"   -> Searching for a hotel in {location} for {nights} nights within a budget of {budget}.")
    # In a real application, this would call a hotel booking API.
    return {
        "status": "success",
        "hotels": [
            {"name": "The Grand Palace", "rating": 4.5, "price_per_night": "₹8,000"},
            {"name": "Comfy Stays", "rating": 4.0, "price_per_night": "₹5,500"},
        ]
    }

def get_tourist_attractions(city: str) -> dict:
    """
    Gets a list of popular tourist attractions for a given city.

    Args:
        city (str): The city for which to find attractions.

    Returns:
        dict: A dictionary containing a list of attractions or an error message.
    """
    print(f"\n🤖 Trippy is using the 'get_tourist_attractions' tool...")
    print(f"   -> Finding tourist attractions in {city}.")
    # In a real application, this would call a travel guide API.
    attractions = {
        "Mumbai": ["Gateway of India", "Marine Drive", "Elephanta Caves"],
        "Delhi": ["India Gate", "Qutub Minar", "Humayun's Tomb"],
        "Goa": ["Baga Beach", "Fort Aguada", "Dudhsagar Falls"],
    }
    return {
        "status": "success",
        "attractions": attractions.get(city, [f"No attractions found for {city}."])
    }

# --- 3. Main Application Logic ---

def main():
    """The main function to run the AI Trip Planner agent."""
    
    print("--- Welcome to the AI Trip Planner! ---")
    try:
        api_key = getpass.getpass("Please enter your Google AI API Key: ")
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Error configuring API key: {e}")
        return

    # --- Model and Tool Configuration ---
    
    # Define the friendly persona for the AI
    system_instruction = ("You are a friendly and enthusiastic travel planner named 'Trippy'. "
                          "Your goal is to help users plan their dream vacations with a cheerful "
                          "and helpful attitude. Use emojis where appropriate to make the "
                          "conversation more engaging.")

    # List the available tools for the model
    tools = [
        find_flights,
        book_hotel,
        get_tourist_attractions,
    ]
    
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash-latest',
        tools=tools,
        system_instruction=system_instruction
    )

    # Start the conversation chat
    convo = model.start_chat(enable_automatic_function_calling=False)
    
    print("\n🤖 Trippy: Hi there! I'm Trippy, your personal AI travel agent. ✈️ I'd be thrilled to help you plan an amazing trip. Where are we off to? (Type 'quit' to exit)")

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                print("\n🤖 Trippy: Happy travels! Come back soon! 👋")
                break

            # Send user message to the model
            response = convo.send_message(user_input)
            
            # The agent's main loop: keep running as long as the model wants to call functions.
            while response.candidates[0].function_calls:
                function_calls = response.candidates[0].function_calls
                
                # In this version, we'll execute all function calls the model requests
                # before sending the results back.
                api_requests = []
                for function_call in function_calls:
                    # Find the corresponding Python function
                    tool_function = globals().get(function_call.name)
                    if not tool_function:
                        raise ValueError(f"Unknown tool: {function_call.name}")
                    
                    # Call the function with the arguments provided by the model
                    tool_result = tool_function(**function_call.args)
                    
                    # Append the result to send back to the model
                    api_requests.append(
                        genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=function_call.name,
                                response={"content": json.dumps(tool_result)}
                            )
                        )
                    )
                
                # Send the function results back to the model
                response = convo.send_message(api_requests)

            # Once the loop finishes, the model has a final text answer.
            print(f"\n🤖 Trippy: {response.text}")

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again.")

# --- Run the application ---
if __name__ == "__main__":
    main()
