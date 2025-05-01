import logging
from typing import Dict, List, Optional, Tuple, Union
import os
from pathlib import Path
from transformers import pipeline
import yaml
import json
from tqdm import tqdm

class Router:
    """
    A text routing system that directs queries to handlers using transformer models.
    Supports zero-shot classification with configurable thresholds and models.
    """
    
    # Simplified model configuration
    DEFAULT_MODEL = {
        "pipeline": "zero-shot-classification",
        "model": "facebook/bart-large-mnli"
    }
    
    def __init__(
        self,
        candidates: Optional[Dict[str, str]] = None,
        model_name: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        threshold: float = 0.1,
        top_n: int = 1,
        device: Union[int, str] = "auto",
        config_path: Optional[str] = None,
        verbose: bool = False
    ) -> None:
        """
        Initialize the Router.
        
        Args:
            candidates: Dictionary of handler names to descriptions
            model_config: Custom model configuration (overrides default)
            threshold: Minimum confidence score (0.0 to 1.0)
            top_n: Number of top candidates to return
            device: Compute device ("cpu", "cuda", "auto", or GPU index)
            config_path: Path to YAML/JSON configuration file
            verbose: Enable detailed logging
        """
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize from config file or parameters
        if config_path:
            self._load_config(config_path)
        else:
            self.candidates = candidates or {}
            self.threshold = threshold
            self.top_n = top_n
            
            # Use provided model config or default
            self.model_name = model_name or self.DEFAULT_MODEL["model"]
            self.pipeline_name = pipeline_name or self.DEFAULT_MODEL["pipeline"]
            
        # Set device
        self.device = self._resolve_device(device)
        
        # Initialize model if we have candidates
        if self.candidates:
            self._initialize_model()
    
    def _resolve_device(self, device: Union[int, str]) -> Union[int, str]:
        """Determine the appropriate compute device."""
        if device != "auto":
            return device
            
        try:
            import torch
            return 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            logging.warning("PyTorch not found, using CPU")
            return "cpu"
    
    def _load_config(self, config_path: Path | str) -> None:
        """Load configuration from file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
            
        try:
            with open(path) as f:
                config = yaml.safe_load(f) if path.suffix in {'.yml', '.yaml'} \
                        else json.load(f) if path.suffix == '.json' \
                        else None
                        
            if config is None:
                raise ValueError(f"Unsupported config format: {path.suffix}")
                
            # Update attributes from config
            self.candidates = config.get('candidates', {})
            self.model_name = config.get('model_name', self.DEFAULT_MODEL["model"])
            self.pipeline_name = config.get('pipeline_name', self.DEFAULT_MODEL["pipeline"])
            self.threshold = config.get('threshold', 0.1)
            self.top_n = config.get('top_n', 1)
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid config file format: {e}")
    
    def _initialize_model(self) -> None:
        """Initialize the classification model with fallback options."""
        original_model_name = self.model_name
        
        # Try to fix common model name issues
        if "-V2" in original_model_name:
            self.model_name = original_model_name.replace("-V2", "")
            logging.warning(f"Detected '-V2' suffix in model name. Trying with corrected name: {self.model_name}")
        
        # First attempt with the specified or corrected model
        try:
            logging.info(f"Initializing model: {self.model_name}")
            self.classifier = pipeline(
                self.pipeline_name,
                model=self.model_name,
                device=self.device,
                token=os.getenv("HUGGINGFACE_TOKEN")
            )
            logging.info("Model initialization successful")
            return
        except Exception as e:
            logging.warning(f"Failed to load specified model: {e}")
        
        # Fallback to default model
        try:
            fallback_model = self.DEFAULT_MODEL["model"]
            logging.warning(f"Attempting to use fallback model: {fallback_model}")
            self.model_name = fallback_model
            self.classifier = pipeline(
                self.pipeline_name,
                model=fallback_model,
                device=self.device,
                token=os.getenv("HUGGINGFACE_TOKEN")
            )
            logging.info("Fallback model initialization successful")
        except Exception as e:
            logging.error(f"All model initialization attempts failed")
            raise RuntimeError(f"Model initialization failed: {str(e)}. Original model '{original_model_name}' and fallback model also failed.")
    
    def route_query(self, query: str) -> List[Tuple[str, float]]:
        """
        Route a query to the most appropriate handler(s).
        
        Args:
            query: The query to route
            
        Returns:
            List of (handler_name, confidence_score) tuples above threshold
        """
        if not query.strip():
            logging.warning("Empty query received")
            return []
            
        if not self.candidates:
            logging.warning("No candidates available")
            return []
            
        try:
            # Get scores using zero-shot classification
            result = self.classifier(
                query,
                candidate_labels=list(self.candidates.values()),
                multi_label=False
            )
            
            # Map scores back to handler names
            description_to_name = {desc: name for name, desc in self.candidates.items()}
            scored_handlers = [
                (description_to_name[label], score)
                for label, score in zip(result["labels"], result["scores"])
                if score >= self.threshold
            ]
            
            return scored_handlers[:self.top_n]
            
        except Exception as e:
            logging.error(f"Routing error: {e}")
            return []
    
    def batch_route(
        self,
        queries: List[str],
        batch_size: int = 8,
        show_progress: bool = True
    ) -> List[List[Tuple[str, float]]]:
        """
        Route multiple queries efficiently in batches.
        
        Args:
            queries: List of queries to route
            batch_size: Queries to process per batch
            show_progress: Show progress bar
            
        Returns:
            List of routing results per query
        """
        if not queries:
            return []
            
        results = []
        batches = range(0, len(queries), batch_size)
        
        if show_progress:
            batches = tqdm(batches, desc="Routing queries")
            
        for i in batches:
            batch = queries[i:i + batch_size]
            results.extend(self.route_query(q) for q in batch)
            
        return results
    
    def add_candidate(self, name: str, description: str) -> None:
        """Add a new candidate handler."""
        self.candidates[name] = description
        logging.debug(f"Added candidate: {name}")
    
    def remove_candidate(self, name: str) -> bool:
        """Remove a candidate handler."""
        if name in self.candidates:
            del self.candidates[name]
            logging.debug(f"Removed candidate: {name}")
            return True
        return False
    
    def save_config(self, config_path: Path | str) -> None:
        """Save current configuration to file."""
        config = {
            'candidates': self.candidates,
            'model_name': self.model_name,
            'pipeline_name': self.pipeline_name,
            'threshold': self.threshold,
            'top_n': self.top_n,
        }
        
        path = Path(config_path)
        try:
            with open(path, 'w') as f:
                if path.suffix in {'.yml', '.yaml'}:
                    yaml.dump(config, f, default_flow_style=False)
                elif path.suffix == '.json':
                    json.dump(config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config format: {path.suffix}")
                    
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to save config: {e}")
    
    def __repr__(self) -> str:
        return (f"Router(candidates={len(self.candidates)}, "
                f"model='{self.model_name}', "
                f"threshold={self.threshold:.2f})")