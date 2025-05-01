from typing import Dict, Optional, Union
from llm_routers.router import Router

class AgentRouter(Router):
    def __init__(
        self,
        agents: Dict[str, str],
        model_name: str = "facebook/bart-large-mnli",
        top_n: int = 1,
        threshold: float = 0.1,
        device: Union[int, str] = "auto",
        verbose: bool = False,
    ) -> None:
        super().__init__(
            candidates=agents,
            model_name=model_name,
            top_n=top_n,
            threshold=threshold,
            device=device,
            verbose=verbose
        )

