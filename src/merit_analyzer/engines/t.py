import inspect
from typing import get_type_hints, List, Dict, Literal, TypedDict
from rich import print
from pydantic import BaseModel, TypeAdapter, create_model
from enum import Enum

class Exec(Enum):
    CEO = "ceo"
    CTO = "cto"
    CMO = "cmo"

class Person(BaseModel):
    name: str
    age: int

class Org(BaseModel):
    employees: List[Person]
    org_name: str
    execs: Dict[Exec, Person]


def t(org: str, action: Literal["Yes", "No"]):
    return ""

sig = inspect.signature(t)
hints = get_type_hints(t)
model = create_model("Args", **{name: (tp, ...) for name, tp in hints.items()})

print(model.model_json_schema())