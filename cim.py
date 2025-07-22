import os
import json
import re
import requests
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
#from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model

from tools.checkAccountBalance import CheckAccountBalance
from tools.checkAccountExistence import CheckAccountExistence
from tools.processTransfer import ProcessTransfer

# ───────────────────────────────────────────────────────────────
# 1) Build each agent ONCE at import time
#    (thread-safe thanks to a simple lock around invoke)
# ───────────────────────────────────────────────────────────────
load_dotenv(override=True)

#from langchain_huggingface import ChatHuggingFace
#from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#from dotenv import load_dotenv
#import os
#
#model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
#
## Load model locally or from HF (this will download the model)
#tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.getenv("HF_TOKEN"))
#model = AutoModelForCausalLM.from_pretrained(model_id, token=os.getenv("HF_TOKEN"))
#
## Create HF pipeline
#pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
#
## Now pass the pipeline into ChatHuggingFace
#llm = ChatHuggingFace(llm=pipe, temperature=0)


import streamlit as st

# Load the API key securely
groq_api_key = st.secrets["GROQ_API_KEY"]

# Get the corrected API key
#groq_key = os.getenv("GROQ_API_KEY", "").strip()

llm = init_chat_model(
    "llama3-70b-8192", 
    model_provider="groq", 
    api_key=groq_api_key,  # Explicitly pass the corrected API key
    temperature=0.5
)
# Initialize LLM
#llm = ChatOpenAI()


from functools import lru_cache

@lru_cache(maxsize=None)
def get_combined_agent():
    tools = [
        CheckAccountExistence(),
        CheckAccountBalance(),
        ProcessTransfer()
    ]
    return create_react_agent(llm, tools=tools)

class TransferState(TypedDict):
    user_query: str
    source_account: str
    destination_account: str
    transfer_amount: str
    user_token: str
    ai_response: str
    redirect_url: str
    transaction_result: str

def get_user_primary_account(user_token: str) -> str:
    """Fetch the user's primary account ID via API."""
    try:
        api_key = st.secrets["CIM_API_KEY"]
        api_url = "https://api.cim.example.com/api/balance"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Token": user_token
        }

        print("DEBUG: Fetching primary account with headers:", headers)
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()

        data = response.json()
        account_id = data.get("account_id", "None")
        print("DEBUG: Primary account fetched:", account_id)
        return account_id

    except requests.RequestException as e:
        print("DEBUG: Exception occurred while fetching primary account:", str(e))
        return "None"

def request_source_account() -> str:
    """Prompt the user to provide their source account."""
    prompt = """
    You are a bank assistant in a mobile banking app.

    Ask the user to specify their source account number or nickname.
    Be professional, concise (1 sentence), and mobile-friendly.
    """

    response = llm.invoke(prompt)
    return response.content

def extract_account_and_amount(user_query: str, user_token: str) -> dict:
    """Extract both destination account and amount using the NOVA system, with fallback to manual extraction."""
    print(f"DEBUG: Running NOVA unified extractor on: {user_query}")
    print(f"DEBUG: Provided user_token: {user_token}")

    prompt = f"""
You are NOVA — a team of expert assistants collaborating to extract account and amount from user messages in a banking app.

📌 ROLES:

1. 🧠 Intent Analyzer  
   - Understands the user’s intent (e.g., transfer, send, top up, etc.)

2. 💼 Account Identifier  
   - Extracts the likely saved payee account nickname or name  
   - Can be a human name ("Supun") or custom nickname ("myboc", "savings", "1234-5678")  
   - If none is found, returns "None"

3. 💸 Amount Extractor  
   - Detects any amount mentioned, with or without currency  
   - Converts "$1,000", "500 rs", "2000" → "2000"  
   - If no valid amount is found, returns "None"

4. 🛠️ Schema Enforcer  
   - Ensures output follows this exact format:  
     {{ "account": "<account>", "amount": <int_or_null>, "user_token": "{user_token}" }}
   - Use `null` (not `None`) for missing values to ensure valid JSON

5. ✅ Finalizer  
   - Returns only the final flat JSON object (no extra explanations or formatting)
   - Ensures all values are properly JSON-formatted

💡 Sri Lankan Banks (to help match context):
Amana Bank, Bank of Ceylon (BOC), Sampath Bank, Seylan, HNB, NDB, People’s Bank, DFCC, HSBC, SBI, etc.

🧪 Examples:
- "Send $500 to myboc" → {{ "account": "myboc", "amount": 500, "user_token": "{user_token}" }}
- "Top up savings with 1000" → {{ "account": "savings", "amount": 1000, "user_token": "{user_token}" }}
- "Transfer to Supun" → {{ "account": "Supun", "amount": null, "user_token": "{user_token}" }}
- "Just checking" → {{ "account": "None", "amount": null, "user_token": "{user_token}" }}

User query: "{user_query}"
"""

    messages = [SystemMessage(content=prompt), HumanMessage(content=user_query)]

    try:
        response = llm.invoke(messages).content.strip()
        print(f"DEBUG: Raw NOVA response: {response}")
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            parsed.setdefault("account", "None")
            parsed.setdefault("amount", None)
            parsed["user_token"] = user_token
            return parsed
    except Exception as e:
        print(f"DEBUG: Error in NOVA unified extraction: {e}")
        
    # Fallback: return default structure
    return {
        "account": "None",
        "amount": None,
        "user_token": user_token
    }

def generate_welcome_message() -> str:
    """Generate welcome message for the transfer service."""
    prompt = """
    You are LEO, the intelligent virtual assistant for a mobile banking app at LB Finance.
    
    Welcome the user to the CIM topup service and ask them to provide:
    1. The payee account number or nickname (from My Payees)
    2. The amount to transfer
    
    Be professional, friendly, concise (under 4 sentences), and suitable for a mobile app notification.
    """
    
    response = llm.invoke(prompt)
    return response.content

def handle_general_inquiry(user_query: str) -> str:
    """Handle general questions about LB Finance and redirect to transactions."""
    from unified_agent import chat_with_unified_agent
    
    # Use the unified agent for all inquiries
    response = chat_with_unified_agent(user_query)
    return response.get("message", "I'm here to help with LB Finance services.")

def is_transaction_intent(user_query: str) -> bool:
    """Check if user query has transaction intent (account name, amount, or transfer keywords)."""
    transaction_keywords = [
        'transfer', 'send', 'topup', 'top up', 'pay', 'payment', 'money',
        'account', 'amount', 'balance', 'transaction', 'deposit'
    ]
    
    # Check for transaction keywords
    has_transaction_keywords = any(keyword in user_query.lower() for keyword in transaction_keywords)
    
    # Check for numbers (potential amounts)
    has_numbers = bool(re.search(r'\d', user_query))
    
    # Check for account-like patterns (names, account numbers)
    has_account_pattern = bool(re.search(r'[a-zA-Z]{2,}', user_query))
    
    return has_transaction_keywords or (has_numbers and has_account_pattern)

def request_account_info() -> str:
    """Request destination account for transfer."""
    prompt = """
    You are a bank assistant in a mobile banking app.
    
    Ask the user to specify the payee account nickname or number from their My Payees list.
    Be professional, concise (1 sentence), and mobile-friendly.
    """
    
    response = llm.invoke(prompt)
    return response.content

def request_amount_info() -> str:
    """Request transfer amount from user."""
    prompt = """
    You are a bank assistant in a mobile banking app.
    
    Ask the user for the specific amount they want to transfer, including currency (e.g., USD).
    Be professional, concise (1 sentence), and mobile-friendly.
    """
    
    response = llm.invoke(prompt)
    return response.content

def generate_account_not_exist_message() -> str:
    """Generate a message for non-existing payee accounts."""
    prompt = """
    You are a bank assistant in a mobile banking app.

    Inform the user that the specified payee account is not in their My Payees list and ask them to try another account or add it to My Payees.
    Be professional, concise (1 sentence), and mobile-friendly.
    """

    response = llm.invoke(prompt)
    return response.content

def generate_insufficient_balance_message() -> str:
    """Generate a message for insufficient balance."""
    prompt = """
    You are a bank assistant in a mobile banking app.

    Inform the user that they have insufficient funds in their account for the transaction and ask them to enter a lower amount.
    Be professional, concise (1 sentence), and mobile-friendly.
    """

    response = llm.invoke(prompt)
    return response.content

def process_user_input(state: TransferState) -> TransferState:
    """Process user input and extract account/amount information using NOVA unified extractor."""
    user_query = state.get("user_query", "")
    user_token = state.get("user_token", "")

    # Check if we're in continuation mode (has existing transaction data)
    is_transaction_continuation = bool(
        state.get("destination_account") or 
        state.get("transfer_amount") or
        state.get("source_account")
    )

    # Check for general inquiries first (but not during transaction continuation)
    # Also check if this is a simple response that could be account/amount data
    simple_response_patterns = [
        r'^to\s+\w+$',  # "to alice"
        r'^\d+(?:\.\d{2})?$',  # just numbers like "200" or "200.50"
        r'^yes|no|ok|sure|proceed|continue$'  # confirmation words
    ]
    
    # Account name patterns (excluding common greetings and conversation words)
    common_greetings = ['hello', 'hi', 'hey', 'good', 'morning', 'afternoon', 'evening', 'thanks', 'thank', 'please']
    could_be_account_name = (
        re.match(r'^\w+$', user_query.strip(), re.IGNORECASE) and 
        user_query.strip().lower() not in common_greetings and
        not re.search(r'\d', user_query)  # no numbers in account names
    )
    
    could_be_transaction_data = (
        any(re.match(pattern, user_query.strip(), re.IGNORECASE) for pattern in simple_response_patterns) or
        could_be_account_name
    )

    if not is_transaction_continuation and not is_transaction_intent(user_query) and not could_be_transaction_data:
        print("🔵 Handling general inquiry")
        state["ai_response"] = handle_general_inquiry(user_query)
        return state

    # Fetch primary account if missing
    if not state.get("source_account"):
        state["source_account"] = get_user_primary_account(user_token)

    if state["source_account"] == "None":
        print("DEBUG: Source account could not be fetched. Prompting user for input.")
        state["ai_response"] = request_source_account()
        return state

    # Use unified NOVA system to extract account and amount
    print("🧠 NOVA: Extracting account and amount together...")
    extracted = extract_account_and_amount(user_query, user_token)

    # Update state only if values are not already set
    if state.get("destination_account") in [None, "None"] and extracted.get("account"):
        state["destination_account"] = extracted["account"]

    if state.get("transfer_amount") in [None, "None"] and extracted.get("amount"):
        state["transfer_amount"] = str(extracted["amount"])

    print(f"✅ Extracted account: {state.get('destination_account')}")
    print(f"✅ Extracted amount: {state.get('transfer_amount')}")

    # Evaluate completeness of state
    has_source = state.get("source_account") and state.get("source_account") != "None"
    has_destination = state.get("destination_account") and state.get("destination_account") != "None"
    has_amount = state.get("transfer_amount") and state.get("transfer_amount") not in [
        "None", "Insufficient balance", "Account not found or not owned by user"
    ]

    if not has_source:
        state["ai_response"] = "Unable to identify your account. Please try again or contact support."
    elif not has_destination and not has_amount:
        # Show welcome message for transaction intent
        print("🟡 Showing welcome message")
        state["ai_response"] = generate_welcome_message()
    elif state.get("transfer_amount") in ["Insufficient balance", "Account not found or not owned by user"]:
        state["transfer_amount"] = None
        state["ai_response"] = (
            generate_insufficient_balance_message()
            if state.get("transfer_amount") == "Insufficient balance"
            else generate_account_not_exist_message()
        )
    elif not has_destination:
        state["ai_response"] = request_account_info()
    elif not has_amount:
        state["ai_response"] = request_amount_info()

    return state


def validate_transfer_data(state: TransferState) -> TransferState:
    """Validate that all required transfer data is present and account exists."""
    print("DEBUG: Validating transfer data. Current state:", state)

    # If this was a general inquiry response (contains LB Finance info), skip validation
    current_response = state.get("ai_response", "")
    if ("lb finance" in current_response.lower() or 
        "banking" in current_response.lower() or 
        "financial institution" in current_response.lower()) and \
       ("transfer" in current_response.lower() or "topup" in current_response.lower()):
        print("DEBUG: Detected general inquiry response, skipping validation")
        return state

    # Check if all required fields are present
    source_account = state.get("source_account")
    destination_account = state.get("destination_account")
    amount = state.get("transfer_amount")
    user_token = state.get("user_token")

    if not source_account or source_account == "None":
        state["ai_response"] = "Source account is missing. Please provide it."
    elif not destination_account or destination_account == "None":
        state["ai_response"] = "Destination account is missing. Please provide it."
    elif not amount or amount == "None":
        state["ai_response"] = "Transfer amount is missing. Please provide it."
    elif not user_token or user_token == "None":
        state["ai_response"] = "User token is missing. Please provide it."
    else:
        # Check if destination account exists
        from controllers.apiController import APIController
        api_controller = APIController()
        
        print(f"DEBUG: Checking if account '{destination_account}' exists for user {user_token}")
        account_check_result = api_controller.check_account_existence(user_token, destination_account)
        
        if account_check_result is None:
            print("DEBUG: Account does not exist.")
            state["ai_response"] = generate_account_not_exist_message()
            state["destination_account"] = None  # Reset so user can try again
        else:
            print("DEBUG: Account exists, now checking balance.")
            # Check if user has sufficient balance
            try:
                amount_value = float(amount.replace(",", "").replace("$", "").strip())
                balance_check_result = api_controller.check_account_balance(user_token, amount_value)
                
                print(f"DEBUG: Balance check result: {balance_check_result}")
                
                if balance_check_result == "Insufficient balance":
                    print("DEBUG: Insufficient balance detected.")
                    state["ai_response"] = generate_insufficient_balance_message()
                    state["transfer_amount"] = None  # Reset so user can try again
                else:
                    print("DEBUG: Balance is sufficient, proceeding with validation.")
                    state["ai_response"] = "All transfer data is valid."
            except ValueError:
                state["ai_response"] = "Invalid amount format. Please specify the amount correctly."
                state["transfer_amount"] = None

    print("DEBUG: Validation complete. Updated state:", state)
    return state

def complete_transfer(state: TransferState) -> TransferState:
    """Complete the transfer process using the banking API."""
    source_account      = state.get("source_account")
    destination_account = state.get("destination_account")
    amount              = state.get("transfer_amount")
    user_token          = state.get("user_token")

    # Debug logs to validate state data
    print("DEBUG: Validating state data before constructing payload:")
    print(f"DEBUG: source_account: {source_account}")
    print(f"DEBUG: destination_account: {destination_account}")
    print(f"DEBUG: amount: {amount}")
    print(f"DEBUG: user_token: {user_token}")

    try:
        amount_int = int(
            amount.replace(",", "").replace("$", "").replace(" euros", "").strip()
        )
    except ValueError:
        state["ai_response"] = "Invalid amount format. Please specify the amount correctly."
        state["redirect_url"] = None
        state["transaction_result"] = None
        return state

    # Ensure all required parameters are present
    if not all([source_account, destination_account, amount_int, user_token]):
        state["ai_response"] = "Missing required transfer details. Please try again."
        state["redirect_url"] = None
        state["transaction_result"] = None
        return state

    prompt = f"""
Process a money transfer using the process_transfer tool.

IMPORTANT: Call the process_transfer tool with these exact parameters as a structured input under the key `tool_input`:
{{
    "tool_input": {{
        "user_token": "{user_token}",
        "source_account": "{source_account}",
        "destination_account": "{destination_account}",
        "amount": {amount_int}
    }}
}}

Transfer details:
- Amount: {amount} ({amount_int})
- From: {source_account}  
- To: {destination_account}
- User: {user_token}

Call the tool now with the exact parameters above.
"""
    messages = [SystemMessage(content=prompt)]

    try:
        # Use the agent with proper message structure
        from toolInputs.processTransferInput import ProcessTransferInput
        from tools.processTransfer import ProcessTransfer
        
        # Create the input data and call the tool directly
        input_data = ProcessTransferInput(
            source_account=source_account,
            destination_account=destination_account,
            amount=amount_int,
            user_token=user_token
        )
        
        tool = ProcessTransfer()
        response = tool.process_transfer(input_data)
        print("DEBUG: Response received from tool:", response)
    except Exception as e:
        state["ai_response"] = f"Something went wrong while processing your request: {str(e)}"
        state["redirect_url"] = None
        state["transaction_result"] = None
        return state

    state["transaction_result"] = response

    if response.startswith("Transfer successful"):
        state["ai_response"] = f"Success! Transferred {amount} to account {destination_account}."
        state["redirect_url"] = (
            f"http://localhost:3000/topup?"
            f"account={destination_account}&amount={amount}&status=success"
        )
    else:
        state["ai_response"] = f"Transfer failed: {response}. Please try again."
        state["redirect_url"] = (
            f"http://localhost:3000/topup?"
            f"account={destination_account}&amount={amount}&status=failed"
        )

    # Reset only relevant fields for next transfer
    state["destination_account"] = None
    state["transfer_amount"] = None
    return state


def reset_state_after_success(state: TransferState) -> TransferState:
    """Reset state after successful transaction for new transfers."""
    final_response = state.get("ai_response", "")
    redirect_url = state.get("redirect_url", "")
    transaction_result = state.get("transaction_result", "")
    
    reset_state = TransferState()
    reset_state["ai_response"] = final_response
    reset_state["redirect_url"] = redirect_url
    reset_state["transaction_result"] = transaction_result
    reset_state["user_token"] = state.get("user_token")
    reset_state["source_account"] = state.get("source_account")
    
    return reset_state

def finalize_process(state: TransferState) -> TransferState:
    """Final cleanup and state preparation."""
    # If we're in finalize without a transaction result, clear any old redirect URLs
    if not state.get("transaction_result"):
        state["redirect_url"] = None
    return state

def has_complete_transfer_data(state: TransferState) -> str:
    """Check if all required transfer data is present."""
    source_account = state.get("source_account")
    destination_account = state.get("destination_account")
    amount = state.get("transfer_amount")
    user_token = state.get("user_token")

    has_source = source_account and source_account != "None"
    has_destination = destination_account and destination_account != "None"
    has_amount = amount and amount != "None" and not amount.startswith("Error") and amount not in ["Insufficient balance", "Account not found or not owned by user"]
    has_token = user_token and user_token != ""

    print("DEBUG: Checking completeness:", {
        "has_source": has_source,
        "has_destination": has_destination,
        "has_amount": has_amount,
        "has_token": has_token
    })

    if has_source and has_destination and has_amount and has_token:
        print("DEBUG: Transfer data is complete.")
        return "complete"
    else:
        print("DEBUG: Transfer data is incomplete.")
        return "incomplete"

builder = StateGraph(TransferState)
builder.add_node("process_input", process_user_input)
builder.add_node("validate_data", validate_transfer_data)
builder.add_node("complete_transfer", complete_transfer)
builder.add_node("reset_state", reset_state_after_success)
builder.add_node("finalize", finalize_process)

builder.add_edge(START, "process_input")
print("DEBUG: Added edge from START to process_input.")

builder.add_edge("process_input", "validate_data")
print("DEBUG: Added edge from process_input to validate_data.")

builder.add_conditional_edges(
    "validate_data",
    has_complete_transfer_data,
    {
        "complete": "complete_transfer",
        "incomplete": "finalize"
    }
)
print("DEBUG: Added conditional edges from validate_data to complete_transfer and finalize.")

builder.add_edge("complete_transfer", "reset_state")
print("DEBUG: Added edge from complete_transfer to reset_state.")

builder.add_edge("reset_state", "finalize")
print("DEBUG: Added edge from reset_state to finalize.")

builder.add_edge("finalize", END)
print("DEBUG: Added edge from finalize to END.")

memory = MemorySaver()

# Compile the graph for LangGraph Studio (without checkpointer)
graph = builder.compile()

# Compile the graph for local execution (with checkpointer)
local_graph = builder.compile(checkpointer=memory)

def execute_graph(thread_id: str, user_input: str, user_token: str = None) -> str:
    """Execute the transfer graph with user input and token, supporting human-in-the-loop interactions."""
    
    # Check if user wants to start fresh and modify thread_id accordingly
    # Only for explicit new transaction requests, not continuation inputs
    explicit_new_transaction_phrases = [
        'new transfer', 'another transfer', 'new transaction', 'start over', 
        'different transfer', 'fresh start', 'need to do a transaction',
        'need to do another transaction', 'do another transaction'
    ]
    
    is_new_conversation = any(keyword in user_input.lower() for keyword in explicit_new_transaction_phrases)
    
    if is_new_conversation:
        # Use a fresh thread ID to ensure clean state
        import time
        thread_id = f"{thread_id}_fresh_{int(time.time())}"
        print(f"DEBUG: Starting fresh conversation with new thread ID: {thread_id}")
    
    thread_config = {"configurable": {"thread_id": thread_id}}

    # Get previous state if it exists (for human-in-the-loop continuity)
    previous_state = local_graph.get_state(thread_config).values if local_graph.get_state(thread_config) else {}
    
    # Determine if this is a continuation of previous conversation
    # Only continue if there's a previous state AND no completed transaction
    is_continuation = bool(previous_state and previous_state.get("ai_response") and 
                          not previous_state.get("transaction_result"))
    
    if is_continuation:
        print("DEBUG: Continuing from previous state...")
        print("DEBUG: Previous state:", previous_state)
        
        # Parse the new user input to extract missing information
        current_state = previous_state.copy()
        current_state["user_query"] = user_input
        
        # Clear any old redirect URLs and transaction results for continuation
        current_state["redirect_url"] = None
        current_state["transaction_result"] = None
        
        # Try to extract missing account or amount from the new input
        extracted = extract_account_and_amount(user_input, user_token or current_state.get("user_token"))
        
        # Update missing fields only
        if not current_state.get("destination_account") and extracted and extracted.get("account") and extracted["account"] != "None":
            current_state["destination_account"] = extracted["account"]
            print(f"DEBUG: Updated missing destination account: {extracted['account']}")
            
        if not current_state.get("transfer_amount") and extracted and extracted.get("amount"):
            current_state["transfer_amount"] = str(extracted["amount"])
            print(f"DEBUG: Updated missing transfer amount: {extracted['amount']}")
            
        # If user provided just an amount (like "100" or "$50")
        if not current_state.get("transfer_amount"):
            amount_match = re.search(r'(\d+(?:\.\d{2})?)', user_input.replace(',', ''))
            if amount_match:
                current_state["transfer_amount"] = amount_match.group(1)
                print(f"DEBUG: Extracted amount from simple input: {amount_match.group(1)}")
        
        # If user provided just an account name (including "to alice" patterns)
        if not current_state.get("destination_account") or current_state.get("destination_account") == "None":
            # Handle "to [account]" pattern
            to_match = re.search(r'to\s+(\w+)', user_input.lower())
            if to_match:
                current_state["destination_account"] = to_match.group(1)
                print(f"DEBUG: Extracted account from 'to [account]' pattern: {to_match.group(1)}")
            # Simple heuristic: if input doesn't contain numbers and isn't a common word, treat as account
            elif not re.search(r'\d', user_input) and user_input.strip().lower() not in ['yes', 'no', 'ok', 'sure', 'please']:
                current_state["destination_account"] = user_input.strip()
                print(f"DEBUG: Extracted account from simple input: {user_input.strip()}")
        
        # Handle confirmation responses like "yes" during transaction flow
        if user_input.strip().lower() in ['yes', 'y', 'ok', 'sure', 'proceed', 'continue']:
            print("DEBUG: User confirmed transaction continuation")
            # Don't treat confirmation as general inquiry - let the transaction flow continue
                
    else:
        print("DEBUG: Starting new conversation...")
        # Prepare initial state for new conversation
        current_state = {}
        current_state["user_query"] = user_input
        current_state["user_token"] = user_token
        # Ensure no old data carries over
        current_state["redirect_url"] = None
        current_state["transaction_result"] = None

    print("DEBUG: Current state prepared:", current_state)

    # Run the graph
    try:
        for step_count, _ in enumerate(local_graph.stream(current_state, thread_config, stream_mode="updates"), start=1):
            print(f"DEBUG: Step {step_count} executed.")
            if step_count > 50:  # Safety limit to prevent infinite loops
                print("DEBUG: Safety limit reached. Breaking out of the loop.")
                break
    except Exception as e:
        print("DEBUG: Exception occurred while running the graph:", str(e))
        return f"Something went wrong while processing your request: {str(e)}"

    # Get final state after graph run
    final_state = local_graph.get_state(thread_config).values if local_graph.get_state(thread_config) else {}
    print("DEBUG: Final state after graph run:", final_state)

    ai_response = final_state.get("ai_response", "Sorry, no response generated.")
    redirect_url = final_state.get("redirect_url")
    transaction_result = final_state.get("transaction_result")

    # Compose response
    response = ai_response
    
    # Only include redirect URL if there's an actual transaction result (success or failure)
    if redirect_url and transaction_result:
        response += f" Navigate to: {redirect_url}"

    return response

##bankingagent.streamlit.app