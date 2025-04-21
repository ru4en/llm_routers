
from typing import Dict
from router import Router

class ToolRouter(Router):
    def __init__(
        self,
        tools: Dict[str, str],
        classifier: str = "zero-shot-classification",
        model: str = "facebook/bart-large-mnli",
        n: int = 3,
        threshold: float = 0.1,
    ) -> None:
        super().__init__(tools, classifier, model, n, threshold)
