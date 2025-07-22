#from pydantic import BaseModel
#
#class CheckAccountExistenceInput(BaseModel):
#    account: str
#    user_token: str


from pydantic import BaseModel, validator
from typing import Union

class CheckAccountExistenceInput(BaseModel):
    account: Union[str, dict]
    user_token: str

    @validator("account", pre=True)
    def extract_name_from_object(cls, value):
        if isinstance(value, dict):
            return value.get("name", "UNKNOWN")
        return value
