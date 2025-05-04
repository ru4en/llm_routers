from typing import Dict, Optional, Union
from llm_routers.router import Router

class AgentRouter(Router):
    """
    Specialized Router for routing to agents.
    """
    
    DEFAULT_MODEL = {
        "pipeline": "zero-shot-classification",
        "model": "facebook/bart-large-mnli"
    }
    
    def __init__(
        self,
        agents: Dict[str, str],
        model_name: Optional[str] = None,
        threshold: float = 0.1,
        top_n: int = 1,
        device: Union[int, str] = "auto",
        config_path: Optional[str] = None,
        verbose: bool = False,
        async_mode: bool = True
    ) -> None:
        """
        Initialize the AgentRouter.
        
        Args:
            agents: Dictionary of agent names to descriptions
            model_name: Name of the model to use
            threshold: Minimum confidence score (0.0 to 1.0)
            top_n: Number of top agents to return
            device: Compute device ("cpu", "cuda", "auto", or GPU index)
            config_path: Path to YAML/JSON configuration file
            verbose: Enable detailed logging
            async_mode: Use asynchronous processing for classification
        """
        super().__init__(
            candidates=agents,
            model_name=model_name or self.DEFAULT_MODEL["model"],
            pipeline_name=self.DEFAULT_MODEL["pipeline"],
            threshold=threshold,
            top_n=top_n,
            device=device,
            config_path=config_path,
            verbose=verbose,
            async_mode=async_mode
        )


