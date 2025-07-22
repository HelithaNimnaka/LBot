from langchain_core.tools import StructuredTool
from controllers.apiController import APIController
from toolInputs.processTransferInput import ProcessTransferInput

print("DEBUG: Initializing ProcessTransfer tool with schema:", ProcessTransferInput.schema_json())

class ProcessTransfer(StructuredTool):
    """Tool to process a transfer from the user's account to a payee account via API."""
    
    def __init__(self):
        super().__init__(
            name="process_transfer",
            func=self.process_transfer,
            description="Process a money transfer from the user's account to a payee account using the banking API.",
            args_schema=ProcessTransferInput
        )
    
    def process_transfer(self, input_data: ProcessTransferInput) -> str:
        """Process the transfer using the API controller."""
        print("DEBUG: Received structured input:", input_data.dict())
        controller = APIController()
        return controller.process_transfer(
            input_data.user_token,
            input_data.source_account,
            input_data.destination_account,
            float(input_data.amount)
        )