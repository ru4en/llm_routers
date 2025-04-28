import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import yaml
import json
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm


class Router:
    """
    A flexible text routing system that directs queries to the most appropriate handler
    based on semantic similarity using transformer models.
    
    Features:
    - Multiple classification model support
    - Configurable via YAML/JSON files
    - Customizable thresholds and batch processing
    - Comprehensive logging and debugging options
    - Progress indicators for batch operations
    """
    
    DEFAULT_MODELS = {
        "zero-shot": {
            "pipeline": "zero-shot-classification",
            "model": "facebook/bart-large-mnli",
        },
        "sentiment": {
            "pipeline": "sentiment-analysis",
            "model": "distilbert-base-uncased-finetuned-sst-2-english",
        },
        "ner": {
            "pipeline": "ner",
            "model": "dbmdz/bert-large-cased-finetuned-conll03-english",
        }
    }
    
    def __init__(
        self,
        candidates: Optional[Dict[str, str]] = None,
        classifier_type: str = "zero-shot",
        model_name: Optional[str] = None,
        n: int = 1,
        threshold: float = 0.1,
        device: Union[int, str] = "auto",
        config_path: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the Router with candidates and model settings.
        
        Args:
            candidates: Dictionary mapping handler names to their descriptions
            classifier_type: Type of classification ("zero-shot", "sentiment", "ner", etc.)
            model_name: Name of the model to use (overrides the default for the classifier)
            n: Number of top candidates to return
            threshold: Minimum confidence score to include a candidate
            device: Device to run the model on ("cpu", "cuda", "auto", or specific GPU index)
            config_path: Path to configuration file (YAML or JSON)
            verbose: Whether to output detailed logs
        """
        # Set up logging
        self._configure_logging(verbose)
        
        # Load config if provided
        if config_path:
            self._load_config(config_path)
        else:
            self.candidates = candidates or {}
            self.classifier_type = classifier_type
            self.n = n
            self.threshold = threshold
            
            # Set model name based on classifier type if not provided
            if model_name:
                self.model_name = model_name
            else:
                if classifier_type in self.DEFAULT_MODELS:
                    self.pipeline_name = self.DEFAULT_MODELS[classifier_type]["pipeline"]
                    self.model_name = self.DEFAULT_MODELS[classifier_type]["model"]
                else:
                    raise ValueError(f"Unknown classifier type: {classifier_type}. "
                                     f"Available types: {list(self.DEFAULT_MODELS.keys())}")
        
        # Set up device
        self.device = self._resolve_device(device)
        logging.info(f"Using device: {self.device}")
        
        # Initialize the model
        self._initialize_model()
    
    def _configure_logging(self, verbose: bool) -> None:
        """Configure logging based on verbosity."""
        if verbose:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            logging.disable(logging.CRITICAL)
    
    def _resolve_device(self, device: Union[int, str]) -> Union[int, str]:
        """Resolve the appropriate device to use."""
        if device == "auto":
            try:
                import torch
                return 0 if torch.cuda.is_available() else "cpu"
            except ImportError:
                logging.warning("PyTorch not found, defaulting to CPU")
                return "cpu"
        return device
    
    def _load_config(self, config_path: str) -> None:
        """
        Load configuration from a YAML or JSON file.
        
        Args:
            config_path: Path to the configuration file
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        logging.info(f"Loading configuration from {config_path}")
        
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        # Set attributes from config
        self.candidates = config.get('candidates', {})
        self.classifier_type = config.get('classifier_type', 'zero-shot')
        self.model_name = config.get('model_name')
        self.pipeline_name = config.get('pipeline_name')
        self.n = config.get('n', 1)
        self.threshold = config.get('threshold', 0.1)
        
        # If model_name not specified, use default
        if not self.model_name and self.classifier_type in self.DEFAULT_MODELS:
            self.pipeline_name = self.DEFAULT_MODELS[self.classifier_type]["pipeline"]
            self.model_name = self.DEFAULT_MODELS[self.classifier_type]["model"]
    
    def _initialize_model(self) -> None:
        """Initialize the classification model."""
        try:
            logging.info(f"Initializing {self.classifier_type} classifier with model {self.model_name}")
            self.classifier = pipeline(
                self.pipeline_name, 
                model=self.model_name, 
                device=self.device
            )
            logging.info("Model initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize model: {str(e)}")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
    
    def add_candidate(self, name: str, description: str) -> None:
        """
        Add a new candidate handler.
        
        Args:
            name: Name of the handler
            description: Description of what the handler does
        """
        self.candidates[name] = description
        logging.debug(f"Added candidate: {name} - {description}")
    
    def remove_candidate(self, name: str) -> bool:
        """
        Remove a candidate handler.
        
        Args:
            name: Name of the handler to remove
            
        Returns:
            bool: True if removed, False if not found
        """
        if name in self.candidates:
            del self.candidates[name]
            logging.debug(f"Removed candidate: {name}")
            return True
        return False
    
    def route_query(self, query: str) -> List[Tuple[str, float]]:
        """
        Route a query to the most appropriate handler(s).
        
        Args:
            query: The user query to route
            
        Returns:
            List of (handler_name, confidence_score) tuples, sorted by confidence
        """
        if not self.candidates:
            logging.warning("No candidates defined. Cannot route query.")
            return []
        
        candidate_values = list(self.candidates.values())
        
        try:
            # Handle different pipeline types
            if self.pipeline_name == "zero-shot-classification":
                result = self.classifier(query, candidate_labels=candidate_values)
                labels = result["labels"]
                scores = result["scores"]
            else:
                # For other pipeline types, implement custom routing logic
                logging.warning(f"Custom routing for {self.pipeline_name} using basic similarity")
                # Simplified fallback - just match keywords
                scores = []
                labels = []
                for label in candidate_values:
                    common_words = set(query.lower().split()) & set(label.lower().split())
                    score = len(common_words) / max(len(query.split()), len(label.split()))
                    scores.append(score)
                    labels.append(label)
            
            logging.debug(f"Classification result: {dict(zip(labels, scores))}")
            
            # Map back to handler names
            description_to_name = {desc: name for name, desc in self.candidates.items()}
            score_dict = {
                description_to_name[label]: score
                for label, score in zip(labels, scores)
            }
            
            # Sort and filter by threshold
            sorted_scores = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
            filtered_scores = [(k, v) for k, v in sorted_scores if v >= self.threshold]
            
            # Limit to top n results
            return filtered_scores[:self.n] if self.n > 0 else filtered_scores
            
        except Exception as e:
            logging.error(f"Error routing query: {str(e)}")
            return []
    
    def batch_route(self, queries: List[str], batch_size: int = 8, show_progress: bool = True) -> List[List[Tuple[str, float]]]:
        """
        Route multiple queries in batch mode for efficiency.
        
        Args:
            queries: List of queries to route
            batch_size: Number of queries to process at once
            show_progress: Whether to show a progress bar
            
        Returns:
            List of routing results, one per query
        """
        results = []
        
        # Process in batches with optional progress bar
        iterator = tqdm(range(0, len(queries), batch_size), desc="Routing") if show_progress else range(0, len(queries), batch_size)
        
        for i in iterator:
            batch = queries[i:i+batch_size]
            batch_results = [self.route_query(query) for query in batch]
            results.extend(batch_results)
            
        return results
    
    def save_config(self, config_path: str) -> None:
        """
        Save the current configuration to a file.
        
        Args:
            config_path: Path where to save the config
        """
        config = {
            'candidates': self.candidates,
            'classifier_type': self.classifier_type,
            'model_name': self.model_name,
            'pipeline_name': self.pipeline_name,
            'n': self.n,
            'threshold': self.threshold,
        }
        
        path = Path(config_path)
        
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        elif path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        logging.info(f"Configuration saved to {config_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics and information about the router.
        
        Returns:
            Dictionary with router statistics
        """
        return {
            'num_candidates': len(self.candidates),
            'model': self.model_name,
            'pipeline': self.pipeline_name,
            'threshold': self.threshold,
            'device': self.device,
            'top_n': self.n,
        }
    
    def __repr__(self) -> str:
        """String representation of the Router."""
        return (f"Router(candidates={len(self.candidates)}, "
                f"model='{self.model_name}', "
                f"pipeline='{self.pipeline_name}', "
                f"threshold={self.threshold}, "
                f"device={self.device})")
