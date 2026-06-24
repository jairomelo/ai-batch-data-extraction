from pydantic import BaseModel, Field
from typing import List, Optional

class OilData(BaseModel):
    image_path: str
    date_estimate: str
    location: str = Field(default="El Burma, Tunisia")
    extractive_phase: str = Field(description="E.g., Topographic/Seismic Exploration, Drilling & Well Creation, etc.")
    equipment_and_infrastructure: List[str]
    substances_and_residues: List[str]
    ecology_and_landscape: List[str]
    people_present: bool
    text_transcription: Optional[str]
    relational_description: str
    confidence_score: str
    
class SchemaTest(BaseModel):
    name: str
    surname: str