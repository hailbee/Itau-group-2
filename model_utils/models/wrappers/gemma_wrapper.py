import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class GemmaModelWrapper:
    """Wrapper for Google Gemma language models.
    
    Gemma is Google's open-source multimodal foundation model.
    This wrapper extracts text features for similarity matching.
    """
    
    def __init__(self, model_name="google/gemma-2b", device=None):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load Gemma model and tokenizer from Hugging Face."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation="eager"  # Use eager attention for stability
        )
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        # Set to eval mode
        self.model.eval()
    
    def encode_text(self, texts):
        """
        Encode a batch of text strings to embeddings.
        
        Args:
            texts: List of text strings or a single string
            
        Returns:
            torch.FloatTensor: Text embeddings of shape (batch_size, embedding_dim)
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize texts
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            # Gemma outputs hidden states
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Extract text features from the last hidden state
            last_hidden = outputs.hidden_states[-1]
            
            # Mask out padding tokens
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
            masked_hidden = last_hidden * attention_mask
            
            # Mean pooling
            features = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
        
        # Normalize
        return F.normalize(features, dim=1)

    @property
    def embedding_dim(self):
        """Get the embedding dimension of the text encoder."""
        # Gemma-2b uses 2048 dimensions, Gemma-7b uses 3072
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'hidden_size'):
                return self.model.config.hidden_size
        # Default based on model size
        if '2b' in self.model_name:
            return 2048
        elif '7b' in self.model_name:
            return 3072
        return 2048
    
    def to(self, device):
        """Move model to specified device."""
        self.device = device
        if self.model:
            self.model = self.model.to(device)
        return self
    
    def eval(self):
        """Set model to evaluation mode."""
        if self.model:
            self.model.eval()
        return self
    
    def train(self):
        """Set model to training mode."""
        if self.model:
            self.model.train()
        return self
