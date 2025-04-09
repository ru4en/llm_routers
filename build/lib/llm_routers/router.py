import logging
from typing import Any, Dict, List, Tuple

from transformers import pipeline

logging.disable(logging.CRITICAL)

class Router:
    def __init__(
        self,
        candidates: Dict[str, str],
        classifier: str = "zero-shot-classification",
        model: str = "facebook/bart-large-mnli",
        n: int = 1,
        threshold: float = 0.1,
    ) -> None:
        self.candidates: Dict[str, str] = candidates
        self.model: str = model
        self.classifier: Any = pipeline(classifier, model=model)
        self.n: int = n
        self.threshold: float = threshold

    def route_query(self, query: str) -> List[Tuple[str, float]]:
        candidate_values: List[str] = list(self.candidates.values())
        result: Dict[str, Any] = self.classifier(query, candidate_labels=candidate_values)
        logging.debug(result)
        
        description_to_name: Dict[str, str] = {desc: name for name, desc in self.candidates.items()}
        
        score_dict: Dict[str, float] = {
            description_to_name[label]: score
            for label, score in zip(result["labels"], result["scores"])
        }
        
        sorted_scores: List[Tuple[str, float]] = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
        filtered_scores: List[Tuple[str, float]] = [(k, v) for k, v in sorted_scores if v >= self.threshold]
        
        return filtered_scores[:self.n]

