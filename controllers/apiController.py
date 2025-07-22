import requests
import os
from dotenv import load_dotenv
import jwt

load_dotenv(override=True)

class APIController():
    """Controller for handling banking API interactions."""
    
    def __init__(self):
        self.api_key = os.getenv("CIM_API_KEY")
        self.base_url = "https://api.cim.example.com"
        self.jwt_secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key")
    
    def verify_user_token(self, token: str) -> str:
        """Verify JWT token and return user_id."""
        try:
            payload = jwt.decode(token, self.jwt_secret_key, algorithms=["HS256"])
            return payload.get("user_id")
        except Exception:
            return None
    
    def check_account_balance(self, user_token: str, amount: float) -> str:
        """Check if the user's primary account has sufficient balance via API."""
        try:
            user_id = self.verify_user_token(user_token)
            if not user_id:
                return "Invalid or missing user token"

            api_url = f"{self.base_url}/api/balance"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "User-Token": user_token
            }
            
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            if data.get("user_id") != user_id:
                return "Account not owned by user"
            balance = data.get("balance", 0)
            
            if amount > balance:
                return "Insufficient balance"
            return str(amount)
        
        except requests.RequestException as e:
            return f"Error checking balance: {str(e)}"
    
    def check_account_existence(self, user_token: str, account: str) -> str:
        """Check if the account is in the user's My Payees list via API."""
        try:
            user_id = self.verify_user_token(user_token)
            if not user_id:
                return "Invalid or missing user token"

            api_url = f"{self.base_url}/api/payees"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "User-Token": user_token
            }
            
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            payees = data.get("payees", [])
            if account not in payees:
                return f"Account {account} is not saved in My Payees"
            return account
        
        except requests.RequestException as e:
            return f"Error checking payee: {str(e)}"
    
    def process_transfer(self, user_token: str, source_account: str, destination_account: str, amount: float) -> str:
        """Process a transfer from source to destination account via API."""
        try:
            user_id = self.verify_user_token(user_token)
            if not user_id:
                return "Invalid or missing user token"

            api_url = f"{self.base_url}/api/transfer"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "User-Token": user_token
            }
            
            payload = {
                "source_account_id": source_account,
                "destination_account_id": destination_account,
                "amount": amount
            }
            
            response = requests.post(api_url, json=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            if data.get("status") == "success":
                return f"Transfer successful: Transaction ID {data.get('transaction_id')}"
            else:
                return f"Transfer failed: {data.get('message', 'Unknown error')}"
        
        except requests.RequestException as e:
            return f"Error processing transfer: {str(e)}"
        

