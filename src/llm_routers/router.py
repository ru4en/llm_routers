import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import os
import yaml
import json
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor

class Router:
    """
    A text routing system that directs queries to handlers using transformer models.
    Supports asynchronous classification to avoid blocking the main thread.
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
        verbose: bool = False,
        async_mode: bool = True,
        max_cache_size: int = 1000,
        max_workers: int = 2
    ) -> None:
        """
        Initialize the Router.
        
        Args:
            candidates: Dictionary of handler names to descriptions
            model_name: Name of the model to use
            pipeline_name: Name of the pipeline to use
            threshold: Minimum confidence score (0.0 to 1.0)
            top_n: Number of top candidates to return
            device: Compute device ("cpu", "cuda", "auto", or GPU index)
            config_path: Path to YAML/JSON configuration file
            verbose: Enable detailed logging
            async_mode: Use asynchronous processing for classification
            max_cache_size: Maximum number of cached results
            max_workers: Maximum number of worker threads
        """
        # Configure logging
        logging_level = logging.DEBUG if verbose else logging.WARNING
        logging.basicConfig(
            level=logging_level,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            force=False  # Don't override existing logger configuration
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging_level)
        
        # Initialize from config file or parameters
        if config_path:
            self._load_config(config_path)
        else:
            self.candidates = candidates or {}
            self.threshold = threshold
            # Ensure top_n is an integer and at least 1
            self.top_n = max(1, int(top_n)) if top_n is not None else 1
            
            # Use provided model config or default
            self.model_name = model_name or self.DEFAULT_MODEL["model"]
            self.pipeline_name = pipeline_name or self.DEFAULT_MODEL["pipeline"]
            
        # Set device
        self.device = self._resolve_device(device)
        
        # Async processing settings
        self.async_mode = async_mode
        self.max_workers = max_workers
        self.max_cache_size = max_cache_size
        
        # Initialize the model, either async or sync
        self.classifier = None
        self._model_initialized = threading.Event()
        
        # Initialize async resources if needed
        if self.async_mode:
            self._thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
            self._request_queue = queue.Queue()
            self._result_cache = {}
            self._shutdown_flag = False
            self._worker_initialized = False
            
            # Start processing thread
            if self.candidates:
                self._init_thread = threading.Thread(target=self._async_init_and_process)
                self._init_thread.daemon = True
                self._init_thread.start()
        
        # Directly initialize model in sync mode
        elif self.candidates:
            self._initialize_model()
            self._model_initialized.set()
    
    def _resolve_device(self, device: Union[int, str]) -> Union[int, str]:
        """Determine the appropriate compute device."""
        if device != "auto":
            return device
            
        try:
            import torch
            return 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            self.logger.warning("PyTorch not found, using CPU")
            return "cpu"
    
    def _load_config(self, config_path: Union[Path, str]) -> None:
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
            # Ensure top_n from config is an integer and at least 1
            top_n_config = config.get('top_n', 1)
            self.top_n = max(1, int(top_n_config)) if top_n_config is not None else 1
            
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid config file format: {e}")
    
    def _initialize_model(self) -> None:
        """Initialize the classification model with fallback options."""
        original_model_name = self.model_name
        
        # Import here to avoid loading transformers until needed
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        except ImportError as e:
            self.logger.error(f"Failed to import required classes from transformers: {e}")
            self.logger.error("Please install transformers with: pip install transformers")
            raise ImportError("The transformers library is required but couldn't be imported. Please install it with: pip install transformers")
        
        # First attempt with user-specified model
        try:
            self.logger.info(f"Initializing model: {self.model_name}")
            
            # Use token if available
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            
            try:
                # First check if tokenizer can be loaded
                tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token)
                
                self.classifier = pipeline(
                    self.pipeline_name,
                    model=self.model_name,
                    device=self.device,
                    token=hf_token,
                    tokenizer=tokenizer
                )
                self.logger.info("Model initialization successful")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load specified model: {e}")
        except Exception as e:
            self.logger.warning(f"Failed to load specified model: {e}")
        
        # Always fall back to the standard Facebook model which is known to work
        try:
            fallback_model = "facebook/bart-large-mnli"
            self.logger.warning(f"Falling back to standard model: {fallback_model}")
            self.model_name = fallback_model
            self.classifier = pipeline(
                self.pipeline_name,
                model=fallback_model,
                device=self.device
            )
            self.logger.info("Fallback model initialization successful")
            return
        except Exception as e:
            error_msg = f"Failed to initialize any model. Last error: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _async_init_and_process(self) -> None:
        """Initialize model and process queue in background thread."""
        try:
            # Initialize model
            self._initialize_model()
            self._worker_initialized = True
            self._model_initialized.set()  # Signal that model is ready
            self.logger.info("Async worker initialized and ready")
            
            # Process queue until shutdown
            while not self._shutdown_flag:
                try:
                    # Get item with timeout to allow checking shutdown flag
                    item = self._request_queue.get(timeout=0.5)
                    query, callback, request_id = item
                    
                    # Check cache first
                    cache_key = f"{query}_{str(self.candidates.keys())}"
                    if cache_key in self._result_cache:
                        result = self._result_cache[cache_key]
                        if callback:
                            callback(result, request_id)
                        self._request_queue.task_done()
                        continue
                    
                    # Process the query
                    result = self._classify_query(query)
                    
                    # Store in cache (with size limit)
                    if len(self._result_cache) >= self.max_cache_size:
                        # Clear some items if cache gets too large
                        # This is a simple approach - could implement LRU more formally
                        for _ in range(100):  # Clear batch of items
                            if self._result_cache:
                                self._result_cache.popitem()
                    
                    self._result_cache[cache_key] = result
                    
                    # Call callback if provided
                    if callback:
                        callback(result, request_id)
                    
                    self._request_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in async worker: {str(e)}")
                    # Mark task as done even on error to avoid blocking
                    try:
                        self._request_queue.task_done()
                    except:
                        pass
                    
        except Exception as e:
            self.logger.error(f"Fatal error in async worker: {str(e)}")
            self._worker_initialized = False

    def _classify_query(self, query: str) -> List[Tuple[str, float]]:
        """Perform the actual classification."""
        if not query.strip() or not self.candidates:
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
            
            # Make sure we're using int slicing
            top_n = min(int(self.top_n), len(scored_handlers))
            return scored_handlers[:top_n]
            
        except Exception as e:
            self.logger.error(f"Classification error: {str(e)}")
            return []

    def route_query(
        self, 
        query: str, 
        callback: Optional[Callable[[List[Tuple[str, float]], Any], None]] = None,
        request_id: Any = None,
        timeout: Optional[float] = 10.0  # Default timeout of 10 seconds
    ) -> List[Tuple[str, float]]:
        """
        Route a query to the most appropriate handler(s).
        
        Args:
            query: The query to route
            callback: Function to call with results (for async mode)
            request_id: Optional ID to track this request (passed to callback)
            timeout: Maximum time to wait for result (sync mode only)
            
        Returns:
            List of (handler_name, confidence_score) tuples above threshold
        """
        if not query.strip():
            self.logger.warning("Empty query received")
            return []
            
        if not self.candidates:
            self.logger.warning("No candidates available")
            return []
        
        # For async mode with callback, add to queue and return immediately
        if self.async_mode and callback:
            self._request_queue.put((query, callback, request_id))
            return []
            
        # For async mode without callback, use thread pool with timeout
        elif self.async_mode:
            # Wait for worker to initialize if needed (with timeout)
            if not self._model_initialized.wait(timeout=timeout):
                self.logger.error("Timed out waiting for model to initialize")
                return []
                
            # Check cache first for direct return
            cache_key = f"{query}_{str(self.candidates.keys())}"
            if cache_key in self._result_cache:
                return self._result_cache[cache_key]
                
            # Submit to thread pool
            future = self._thread_pool.submit(self._classify_query, query)
            try:
                result = future.result(timeout=timeout)
                # Cache the result
                if len(self._result_cache) < self.max_cache_size:
                    self._result_cache[cache_key] = result
                return result
            except TimeoutError:
                self.logger.warning(f"Classification timed out after {timeout} seconds")
                return []
                
        # For sync mode, do classification directly
        else:
            # Wait for model to initialize if needed
            if not self._model_initialized.wait(timeout=timeout):
                self.logger.error("Timed out waiting for model to initialize")
                return []
            return self._classify_query(query)
    
    def batch_route(
        self,
        queries: List[str],
        batch_size: int = 8,
        show_progress: bool = True,
        callback: Optional[Callable[[List[List[Tuple[str, float]]], Any], None]] = None,
        request_id: Any = None
    ) -> List[List[Tuple[str, float]]]:
        """
        Route multiple queries efficiently in batches.
        
        Args:
            queries: List of queries to route
            batch_size: Queries to process per batch
            show_progress: Show progress bar
            callback: Function to call with all results (for async mode)
            request_id: Optional ID to track this request batch
            
        Returns:
            List of routing results per query, or empty list in async mode with callback
        """
        if not queries:
            return []
        
        # For async batch processing with callback
        if self.async_mode and callback:
            def process_batch():
                results = []
                for i in range(0, len(queries), batch_size):
                    batch = queries[i:i+batch_size]
                    # In async mode, we need to process each query individually
                    # and collect results because the model's batch processing
                    # is happening in a separate thread
                    batch_results = []
                    for q in batch:
                        result = self._classify_query(q)
                        batch_results.append(result)
                    results.extend(batch_results)
                callback(results, request_id)
                
            # Start a thread to process the entire batch
            threading.Thread(target=process_batch).start()
            return []
            
        # For synchronous processing
        results = []
        iterator = range(0, len(queries), batch_size)
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Routing queries")
            except ImportError:
                self.logger.warning("tqdm not installed, progress bar disabled")
                
        for i in iterator:
            batch = queries[i:i + batch_size]
            
            # For larger batches, use true batch processing with the model
            if self.async_mode:
                # For async mode without callback, we process each query individually
                batch_results = []
                for q in batch:
                    result = self.route_query(q)
                    batch_results.append(result)
                results.extend(batch_results)
            else:
                # For sync mode, use model's batch capabilities if implemented
                results.extend(self.route_query(q) for q in batch)
                
        return results
    
    def add_candidate(self, name: str, description: str) -> None:
        """Add a new candidate handler."""
        self.candidates[name] = description
        # Clear cache since candidates have changed
        if self.async_mode:
            self._result_cache.clear()
        self.logger.debug(f"Added candidate: {name}")
    
    def remove_candidate(self, name: str) -> bool:
        """Remove a candidate handler."""
        if name in self.candidates:
            del self.candidates[name]
            # Clear cache since candidates have changed
            if self.async_mode:
                self._result_cache.clear()
            self.logger.debug(f"Removed candidate: {name}")
            return True
        return False
    
    def save_config(self, config_path: Union[Path, str]) -> None:
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
    
    def shutdown(self) -> None:
        """Shut down the router and its worker threads."""
        if self.async_mode:
            self._shutdown_flag = True
            try:
                self._thread_pool.shutdown(wait=False)
            except:
                pass
    
    def __del__(self):
        """Ensure resources are cleaned up."""
        self.shutdown()
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(candidates={len(self.candidates)}, "
                f"model='{self.model_name}', "
                f"threshold={self.threshold:.2f}, "
                f"async_mode={self.async_mode})")


