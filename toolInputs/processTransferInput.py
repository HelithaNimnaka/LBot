from pydantic import BaseModel

class ProcessTransferInput(BaseModel):
    source_account: str
    destination_account: str
    amount: int
    user_token: str