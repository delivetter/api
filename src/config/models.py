from pydantic import BaseModel


class JWT(BaseModel):
    secret: str


class ConfigType(BaseModel):
    jwt: JWT
