from crewai_tools import PDFSearchTool
from crewai.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field, ConfigDict
import os


class PDFSearchInput(BaseModel):
    """Input schema for PDFSearchTool."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    query: str = Field(..., description="The search query to find in the PDF document.")
    pdf_path: Optional[str] = Field(None, description="Optional path to a specific PDF file. If not provided, will use the default PDF set during initialization.")


class PDFSearch(BaseTool):
    name: str = "PDF Search"
    description: str = (
        "A tool for searching through PDF documents. Can search through a specific PDF file "
        "or use a pre-configured PDF document. Returns relevant text snippets from the PDF "
        "that match the search query."
    )
    args_schema: Type[BaseModel] = PDFSearchInput
    _pdf_tool: Optional[PDFSearchTool] = None

    def __init__(self, pdf_path: Optional[str] = None):
        super().__init__()
        config = {
            "llm": {
                "provider": "google",
                "config": {
                    "model": os.getenv("MODEL", "gemini/gemini-1.5-pro"),
                    "api_key": os.getenv("GEMINI_API_KEY")
                }
            },
            "embedder": {
                "provider": "google",
                "config": {
                    "model": "models/embedding-001",
                    "api_key": os.getenv("GEMINI_API_KEY")
                }
            }
        }
        self._pdf_tool = PDFSearchTool(pdf=pdf_path, config=config)

    def _run(self, query: str, pdf_path: Optional[str] = None) -> str:
        if pdf_path:
            # If a new PDF path is provided, create a new tool instance
            self._pdf_tool = PDFSearchTool(pdf=pdf_path)
        
        if not self._pdf_tool:
            raise ValueError("No PDF document has been configured. Please provide a PDF path during initialization or in the search query.")
            
        return self._pdf_tool.run(query) 