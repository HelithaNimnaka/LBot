from langchain_core.tools import Tool
from toolInputs.checkAccountExistenceInput import CheckAccountExistenceInput
from controllers.apiController import APIController

class CheckAccountExistence(Tool):
    """Tool to check if an account is in the user's My Payees list via API."""
    
    def __init__(self):
        super().__init__(
            name="check_account_existence",
            func=self.check_existence,
            description="Check if the specified account is in the user's My Payees list using the banking API.",
            args_schema=CheckAccountExistenceInput
        )
    
    #def check_existence(self, account: str, user_token: str) -> str:
    #    """Check if the account exists in the user's payee list using the API controller."""
    #    controller = APIController()
    #    return controller.check_account_existence(user_token, account)
    

    def check_existence(self, account: str, user_token: str) -> str:
        # Fix if account is an object instead of string
        if isinstance(account, dict):
            account = account.get("name", "UNKNOWN")
        controller = APIController()
        return controller.check_account_existence(user_token, account)
