"""
Unified LB Finance Agent
Combines general LB Finance knowledge with transaction processing
"""
import os
import json
import re
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from tools.checkAccountBalance import CheckAccountBalance
from tools.checkAccountExistence import CheckAccountExistence
from tools.processTransfer import ProcessTransfer

load_dotenv(override=True)

# Initialize LLM
groq_key = os.getenv("GROQ_API_KEY", "").strip()
llm = init_chat_model(
    "llama3-70b-8192", 
    model_provider="groq", 
    api_key=groq_key,
    temperature=0.3  # Lower temperature for more consistent responses
)

class UnifiedLBFinanceAgent:
    """
    Unified agent that handles both general LB Finance inquiries and transactions
    """
    
    def __init__(self):
        # Create the agent with banking tools
        self.tools = [
            CheckAccountExistence(),
            CheckAccountBalance(),
            ProcessTransfer()
        ]
        self.agent = create_react_agent(llm, tools=self.tools)
        
        # LB Finance knowledge base
        self.lb_finance_info = """
        LB Finance PLC is a leading financial institution in Sri Lanka, established in 1971.
        
        Services:
        - Mobile Banking & Digital Payments
        - Personal Banking (Savings, Current Accounts)
        - Loans (Personal, Home, Vehicle)
        - Fixed Deposits & Investments
        - Credit Cards
        - Money Transfers & Remittances
        - Foreign Exchange
        
        Branches: Over 200+ branches across Sri Lanka
        ATM Network: 500+ ATMs island-wide
        Customer Service: 24/7 support via mobile app and hotline
        
        Digital Services:
        - LB Finance Mobile App
        - Internet Banking
        - QR Payments
        - Bill Payments
        - Account Management
        """
    
    def chat(self, user_input: str, user_token: str = "user123", conversation_context: dict = None) -> dict:
        """
        Main chat interface that intelligently routes between general chat and transactions
        """
        
        # Extract previous transaction context if available
        previous_account = None
        previous_amount = None
        previous_action = None
        previous_status = None
        
        if conversation_context and conversation_context.get('last_response'):
            last_transaction = conversation_context['last_response'].get('transaction_data')
            previous_status = conversation_context['last_response'].get('status')
            
            # Only use previous context if the last transaction was incomplete (needs more info)
            # Reset context after successful transactions
            if (last_transaction and 
                conversation_context['last_response'].get('needs_more_info', False) and
                previous_status != 'success'):
                previous_account = last_transaction.get('account')
                previous_amount = last_transaction.get('amount')
                previous_action = last_transaction.get('action')
        
        # Create the system prompt for the unified agent
        system_prompt = f"""
You are LEO, the intelligent virtual assistant for LB Finance mobile banking app.

CORE RESPONSIBILITIES:
1. **LB Finance Expert**: Answer questions about LB Finance services, branches, products, etc.
2. **Transaction Processor**: Handle money transfers, account checks, and banking operations
3. **Conversation Router**: Intelligently detect when to switch between chat and transactions
4. **Context Awareness**: Remember previous transaction details within the conversation

LB FINANCE KNOWLEDGE:
{self.lb_finance_info}

CONVERSATION CONTEXT:
- Previous account mentioned: {previous_account or "None"}
- Previous amount mentioned: {previous_amount or "None"}  
- Previous action: {previous_action or "None"}
- Previous status: {previous_status or "None"}

CONTEXT RULES:
- If user provides just an amount (like "200") and there's an INCOMPLETE previous transaction with account, combine them
- If user provides just an account name (like "alice") and there's an INCOMPLETE previous transaction with amount, combine them
- IMPORTANT: Reset context after successful transactions - don't carry over completed transaction data
- Only maintain context for incomplete transactions that need more information
- Each new account name should start a fresh transaction unless continuing an incomplete one

CONVERSATION GUIDELINES:

For GENERAL INQUIRIES (company info, services, help):
- Provide helpful information about LB Finance
- Be friendly, professional, and knowledgeable
- Always end by asking if they need help with transactions

For TRANSACTION REQUESTS (transfer, send money, check balance):
- Use the available banking tools to process requests
- Extract account names/numbers and amounts from user input
- Combine with previous context when appropriate
- Guide users through missing information step by step
- Current user token: {user_token}

DETECTION RULES:
- Transaction keywords: transfer, send, pay, topup, balance, account, amount, money
- Account patterns: names like "alice", "bob", account numbers, "to [name]" patterns
- Amount patterns: numbers with/without currency symbols
- Context completion: single values that complete previous partial transactions

RESPONSE FORMAT:
Always respond with a JSON object:
{{
    "response_type": "general" | "transaction" | "mixed",
    "message": "Your response to the user",
    "transaction_data": {{
        "account": "extracted_or_previous_account",
        "amount": extracted_or_previous_amount,
        "action": "transfer" | "balance_check" | "account_check" | null
    }},
    "needs_more_info": true/false,
    "missing_fields": ["account", "amount"] // if needs_more_info is true
}}

EXAMPLES:

User: "What services does LB Finance offer?"
Response: {{
    "response_type": "general",
    "message": "LB Finance offers comprehensive banking services including personal banking, loans, credit cards, mobile banking, and money transfers. We have 200+ branches and 500+ ATMs across Sri Lanka. Would you like to make a transfer or check your account today?",
    "transaction_data": null,
    "needs_more_info": false,
    "missing_fields": []
}}

User: "Send 500 to alice"
Response: {{
    "response_type": "transaction",
    "message": "I'll help you transfer 500 to alice. Let me process this transfer for you.",
    "transaction_data": {{
        "account": "alice",
        "amount": 500,
        "action": "transfer"
    }},
    "needs_more_info": false,
    "missing_fields": []
}}

User: "to bob" (when no previous context)
Response: {{
    "response_type": "transaction", 
    "message": "I see you want to send money to bob. Please specify the amount you'd like to transfer.",
    "transaction_data": {{
        "account": "bob",
        "amount": null,
        "action": "transfer"
    }},
    "needs_more_info": true,
    "missing_fields": ["amount"]
}}

User: "200" (when previous INCOMPLETE transaction had account "bob")
Response: {{
    "response_type": "transaction",
    "message": "Perfect! I'll transfer 200 to bob. Let me process this for you.",
    "transaction_data": {{
        "account": "bob",
        "amount": 200,
        "action": "transfer"
    }},
    "needs_more_info": false,
    "missing_fields": []
}}

User: "alice" (after successful transaction, should start new)
Response: {{
    "response_type": "transaction", 
    "message": "I see you want to send money to alice. Please specify the amount you'd like to transfer.",
    "transaction_data": {{
        "account": "alice",
        "amount": null,
        "action": "transfer"
    }},
    "needs_more_info": true,
    "missing_fields": ["amount"]
}}

Current conversation context: {conversation_context or "New conversation"}
User input: "{user_input}"

IMPORTANT: 
- If the user provides just a number and there's an INCOMPLETE previous transaction with account, combine them for a complete transaction
- If the user provides just an account name, check if there's an INCOMPLETE previous transaction with amount to combine
- After successful transactions, treat new inputs as fresh transaction requests
- Don't carry over completed transaction context to new requests

Respond with the appropriate JSON format based on the user's input and context.
"""

        try:
            # Get response from the agent
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ]
            
            response = llm.invoke(messages)
            
            # Try to parse the JSON response
            try:
                parsed_response = json.loads(response.content)
                return self._process_agent_response(parsed_response, user_token)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "response_type": "general",
                    "message": response.content,
                    "transaction_data": None,
                    "status": "success"
                }
                
        except Exception as e:
            return {
                "response_type": "error",
                "message": f"Sorry, I encountered an error: {str(e)}",
                "transaction_data": None,
                "status": "error"
            }
    
    def _process_agent_response(self, parsed_response: dict, user_token: str) -> dict:
        """
        Process the agent's response and execute any required transactions
        """
        response_type = parsed_response.get("response_type", "general")
        transaction_data = parsed_response.get("transaction_data")
        
        # If it's a transaction with complete data, execute it
        if (response_type == "transaction" and 
            transaction_data and 
            not parsed_response.get("needs_more_info", False)):
            
            action = transaction_data.get("action")
            account = transaction_data.get("account")
            amount = transaction_data.get("amount")
            
            if action == "transfer" and account and amount:
                return self._execute_transfer(account, amount, user_token, parsed_response)
            elif action == "balance_check":
                return self._check_balance(user_token, parsed_response)
            elif action == "account_check" and account:
                return self._check_account(account, user_token, parsed_response)
        
        # Return the response as-is for general inquiries or incomplete transactions
        parsed_response["status"] = "success"
        return parsed_response
    
    def _execute_transfer(self, destination_account: str, amount: int, user_token: str, base_response: dict) -> dict:
        """Execute a money transfer"""
        try:
            from controllers.apiController import APIController
            from toolInputs.processTransferInput import ProcessTransferInput
            
            # Get source account
            api_controller = APIController()
            source_account = self._get_user_primary_account(user_token)
            
            if source_account == "None":
                base_response["message"] = "Unable to identify your account. Please contact support."
                base_response["status"] = "error"
                return base_response
            
            # Check if destination account exists
            account_exists = api_controller.check_account_existence(user_token, destination_account)
            if not account_exists:
                base_response["message"] = f"The account '{destination_account}' is not in your payees list. Please add it first or try another account."
                base_response["status"] = "error"
                return base_response
            
            # Check balance
            balance_check = api_controller.check_account_balance(user_token, float(amount))
            if balance_check == "Insufficient balance":
                base_response["message"] = f"Insufficient funds for transferring {amount}. Please try a lower amount."
                base_response["status"] = "error"
                return base_response
            
            # Execute transfer
            input_data = ProcessTransferInput(
                source_account=source_account,
                destination_account=destination_account,
                amount=int(amount),
                user_token=user_token
            )
            
            tool = ProcessTransfer()
            result = tool.process_transfer(input_data)
            
            if result.startswith("Transfer successful"):
                base_response["message"] = f"✅ Successfully transferred {amount} to {destination_account}!"
                base_response["status"] = "success"
                base_response["redirect_url"] = f"http://localhost:3000/topup?account={destination_account}&amount={amount}&status=success"
            else:
                base_response["message"] = f"❌ Transfer failed: {result}"
                base_response["status"] = "error"
                
            return base_response
            
        except Exception as e:
            base_response["message"] = f"Error processing transfer: {str(e)}"
            base_response["status"] = "error"
            return base_response
    
    def _check_balance(self, user_token: str, base_response: dict) -> dict:
        """Check account balance"""
        try:
            from controllers.apiController import APIController
            api_controller = APIController()
            
            # This would call a balance inquiry API
            base_response["message"] = "Balance check functionality would be implemented here."
            base_response["status"] = "success"
            return base_response
            
        except Exception as e:
            base_response["message"] = f"Error checking balance: {str(e)}"
            base_response["status"] = "error"
            return base_response
    
    def _check_account(self, account: str, user_token: str, base_response: dict) -> dict:
        """Check if account exists"""
        try:
            from controllers.apiController import APIController
            api_controller = APIController()
            
            exists = api_controller.check_account_existence(user_token, account)
            if exists:
                base_response["message"] = f"✅ Account '{account}' is in your payees list."
            else:
                base_response["message"] = f"❌ Account '{account}' is not found in your payees list."
            
            base_response["status"] = "success"
            return base_response
            
        except Exception as e:
            base_response["message"] = f"Error checking account: {str(e)}"
            base_response["status"] = "error"
            return base_response
    
    def _get_user_primary_account(self, user_token: str) -> str:
        """Get user's primary account for transfers"""
        try:
            import requests
            api_key = os.getenv("CIM_API_KEY")
            api_url = "https://api.cim.example.com/api/balance"
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "User-Token": user_token
            }
            
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            return data.get("account_id", "123456")  # Default for testing
            
        except:
            return "123456"  # Default for testing

# Global instance
unified_agent = UnifiedLBFinanceAgent()

def chat_with_unified_agent(user_input: str, user_token: str = "user123", context: dict = None) -> dict:
    """
    Main function to chat with the unified LB Finance agent
    """
    return unified_agent.chat(user_input, user_token, context)
