
from typing import Dict
from router import Router

class AgentRouter(Router):
    def __init__(
        self,
        agents: Dict[str, str],
        classifier: str = "zero-shot-classification",
        model: str = "facebook/bart-large-mnli",
        n: int = 1,
        threshold: float = 0.1,
    ) -> None:
        super().__init__(agents, classifier, model, n, threshold)

