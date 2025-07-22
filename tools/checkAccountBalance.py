from langchain_core.tools import Tool
from toolInputs.checkAccountBalanceInput import CheckAccountBalanceInput
from controllers.apiController import APIController

class CheckAccountBalance(Tool):
    """Tool to check if the user's primary account has sufficient balance via API."""
    
    def __init__(self):
        super().__init__(
            name="check_account_balance",
            func=self.check_balance,
            description="Check if the user's primary account has sufficient balance for the transaction via the banking API.",
            args_schema=CheckAccountBalanceInput
        )
    
    def check_balance(self, amount: int, user_token: str) -> str:
        """Check if the user's account balance is sufficient using the API controller."""
        controller = APIController()
        return controller.check_account_balance(user_token, float(amount))