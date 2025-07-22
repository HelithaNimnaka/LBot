#from pydantic import BaseModel
#
#class CheckAccountBalanceInput(BaseModel):
#    amount: int
#    user_token: str
#
    

from pydantic import BaseModel, validator
from typing import Union

class CheckAccountBalanceInput(BaseModel):
    amount: Union[int, dict]
    user_token: str

    @validator("amount", pre=True)
    def extract_number_from_object(cls, value):
        if isinstance(value, dict):
            return value.get("value", 0)
        return value
