import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class InternVLModelWrapper:
    """Wrapper for InternVL vision-language models."""
    
    def __init__(self, model_name="OpenGVLab/InternVL2-2B", device=None):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the InternVL model and tokenizer."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model with trust_remote_code for InternVL custom code
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                device_map='auto'
            )
            
            # Move to specified device if not using device_map='auto'
            if self.device.type != 'meta':
                try:
                    self.model = self.model.to(self.device)
                except:
                    # If device_map was used, model might already be distributed
                    pass
                    
        except Exception as e:
            raise RuntimeError(f"Failed to load InternVL model {self.model_name}: {str(e)}")
    
    def encode_text(self, texts):
        """
        Encode texts using InternVL's text encoder.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            torch.Tensor: Normalized text embeddings [batch_size, embedding_dim]
        """
        # Tokenize the texts
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get text embeddings from the model
        with torch.no_grad():
            # InternVL models have a text encoder that can be accessed via the model
            # The model forward pass returns embeddings
            if hasattr(self.model, 'get_text_features'):
                # If model has explicit get_text_features method
                features = self.model.get_text_features(**inputs)
            elif hasattr(self.model, 'encode_text'):
                # If model has encode_text method
                features = self.model.encode_text(**inputs)
            else:
                # Standard transformer output - use last hidden state mean pooling
                outputs = self.model.text_model(**inputs) if hasattr(self.model, 'text_model') else self.model(**inputs)
                
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    features = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                    # Mean pooling over sequence dimension
                    features = outputs.last_hidden_state.mean(dim=1)
                else:
                    # Fallback: find first 2D tensor
                    features = None
                    if isinstance(outputs, dict):
                        for v in outputs.values():
                            if v is not None and len(v.shape) == 2:
                                features = v
                                break
                    if features is None:
                        raise ValueError(f"Could not extract features from InternVL model output")
        
        # Normalize embeddings
        return F.normalize(features, dim=1)

    @property
    def embedding_dim(self):
        """Get the embedding dimension of the model."""
        try:
            # Try to get hidden size from text model config
            if hasattr(self.model, 'text_config'):
                return self.model.text_config.hidden_size
            elif hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
                return self.model.config.hidden_size
            elif hasattr(self.model, 'config') and hasattr(self.model.config, 'text_config'):
                return self.model.config.text_config.hidden_size
            else:
                # Fallback: encode a dummy text to get embedding dimension
                with torch.no_grad():
                    dummy_embeddings = self.encode_text(["dummy"])
                    return dummy_embeddings.shape[1]
        except Exception as e:
            raise RuntimeError(f"Could not determine embedding dimension: {str(e)}")
    
    def to(self, device):
        """Move model to specified device."""
        self.device = device
        if self.model:
            try:
                self.model = self.model.to(device)
            except:
                # Model might already be distributed or on device
                pass
        return self
