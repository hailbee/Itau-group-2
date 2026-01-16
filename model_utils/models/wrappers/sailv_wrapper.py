import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import warnings

class SAILVModelWrapper:
    """
    Wrapper for SAIL-VL (Sparsity-Aware Intermediate Language Vision) models from ByteDance.
    
    SAIL-VL is a specialized vision-language model. This wrapper provides text embedding
    capabilities compatible with the baseline testing framework.
    """
    
    def __init__(self, model_name="BytedanceDouyinContent/SAIL-VL-1d5-2B", device=None):
        """
        Initialize SAIL-VL wrapper.
        
        Args:
            model_name: Model identifier on HuggingFace
            device: Device to run model on (auto-detected if None)
        """
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.is_fallback = False
        self._load_model()
    
    def _load_model(self):
        """Load SAIL-VL model and tokenizer from HuggingFace."""
        try:
            print(f"Loading SAIL-VL model: {self.model_name}")
            
            # Suppress warnings about architecture mismatch
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Try loading the actual SAIL-VL model
                try:
                    # Import AutoModel here to get the latest version
                    from transformers import AutoModel
                    
                    self.model = AutoModel.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float32,
                        device_map=self.device if isinstance(self.device, str) else None,
                        attn_implementation="flash_attention_2" if "cuda" in str(self.device) else "eager"
                    )
                    if not isinstance(self.device, str):
                        self.model = self.model.to(self.device)
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        use_fast=False
                    )
                    
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    self.is_fallback = False
                    print(f"✓ Successfully loaded SAIL-VL model: {self.model_name}")
                    return
                    
                except Exception as e:
                    print(f"SAIL-VL direct load failed: {str(e)[:100]}")
                    # If direct loading fails, try without device_map
                    from transformers import AutoModel
                    self.model = AutoModel.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float32
                    ).to(self.device)
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        use_fast=False
                    )
                    
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    self.is_fallback = False
                    print(f"✓ Successfully loaded SAIL-VL (without device_map)")
                    return
            
        except Exception as e:
            error_msg = str(e)
            print(f"\n⚠️  SAIL-VL load failed: {error_msg[:100]}\n")
            
            # Fallback to compatible models
            fallback_models = [
                "sentence-transformers/all-MiniLM-L6-v2",
                "bert-base-uncased",
            ]
            
            for fallback_name in fallback_models:
                try:
                    print(f"Loading fallback model: {fallback_name}")
                    from transformers import AutoModel
                    
                    self.model = AutoModel.from_pretrained(
                        fallback_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float32
                    ).to(self.device)
                    
                    self.tokenizer = AutoTokenizer.from_pretrained(fallback_name)
                    
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    self.model_name = fallback_name
                    self.is_fallback = True
                    print(f"✓ Successfully loaded fallback: {fallback_name}")
                    print(f"⚠️  Using {fallback_name} as fallback for SAIL-VL\n")
                    return
                    
                except Exception as fb_error:
                    print(f"Fallback {fallback_name} failed: {str(fb_error)[:50]}")
                    continue
            
            # If all loads failed
            raise RuntimeError(
                f"Failed to load SAIL-VL model and all fallbacks.\n"
                f"Original error: {error_msg}"
            )
    
    def encode_text(self, texts):
        """
        Encode text inputs to embeddings.
        
        Args:
            texts: List of text strings or single string to encode
            
        Returns:
            Normalized embedding tensor of shape [batch_size, embedding_dim]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize inputs
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Extract text embeddings from model output
            if hasattr(outputs, 'text_embeds') and outputs.text_embeds is not None:
                # If model has explicit text_embeds output
                features = outputs.text_embeds
            elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                # Use pooled output if available
                features = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                # Fall back to mean pooling of last hidden state
                features = outputs.last_hidden_state.mean(dim=1)
            elif isinstance(outputs, dict):
                # Try to find a 2D tensor in outputs dict
                features = None
                for key in ['text_embeds', 'embeddings', 'pooled_output']:
                    if key in outputs and outputs[key] is not None:
                        features = outputs[key]
                        break
                
                if features is None:
                    # Find first 2D tensor
                    for v in outputs.values():
                        if v is not None and isinstance(v, torch.Tensor) and len(v.shape) == 2:
                            features = v
                            break
                
                if features is None:
                    raise ValueError(
                        f"Could not extract 2D embeddings from model output.\n"
                        f"Output keys: {outputs.keys()}"
                    )
            else:
                # outputs is likely a tensor directly
                if isinstance(outputs, torch.Tensor):
                    if len(outputs.shape) == 2:
                        features = outputs
                    else:
                        features = outputs.mean(dim=1)
                else:
                    raise ValueError(
                        f"Unexpected output type: {type(outputs)}"
                    )
        
        # Normalize embeddings
        return F.normalize(features, dim=1)

    @property
    def embedding_dim(self):
        """Get the embedding dimension of the text encoder."""
        try:
            if hasattr(self.model, 'config'):
                config = self.model.config
                
                # Try various config attribute names
                for attr in ['hidden_size', 'dim', 'embedding_dim', 'text_config']:
                    if hasattr(config, attr):
                        val = getattr(config, attr)
                        if isinstance(val, int):
                            return val
                        elif hasattr(val, 'hidden_size'):
                            return val.hidden_size
                
                # Check if it's a vision-language model with text_config
                if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
                    return config.text_config.hidden_size
        except Exception as e:
            print(f"Warning: Could not determine embedding_dim: {e}")
        
        # Default fallback
        return 512
    
    def to(self, device):
        """Move model to specified device."""
        self.device = device
        if self.model:
            self.model = self.model.to(device)
        return self
