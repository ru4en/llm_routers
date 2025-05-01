from typing import Dict, Optional, Union
from llm_routers.router import Router

class ToolRouter(Router):
    def __init__(
        self,
        tools: Dict[str, str],
        model_name: str = "facebook/bart-large-mnli",
        top_n: int = 3,
        threshold: float = 0.1,
        device: Union[int, str] = "auto",
        verbose: bool = False,
    ) -> None:
        # Map parameters correctly to parent class
        super().__init__(
            candidates=tools,
            model_name=model_name,
            top_n=top_n,
            threshold=threshold,
            device=device,
            verbose=verbose
        )
