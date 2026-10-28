"""Small declarative schemas used by CAPRICORN model components."""

from pydantic import BaseModel, ConfigDict, Field


class OutputSchema(BaseModel):
    """Describe an output type and its ordered candidate questions."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    type: str = Field(min_length=1)
    questions: tuple[str, ...]
