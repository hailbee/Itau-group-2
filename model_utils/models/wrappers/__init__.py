"""
Model wrappers for different vision-language models.
Each wrapper provides a consistent interface for encoding text.
"""

from .clip_wrapper import CLIPModelWrapper
from .coca_wrapper import CoCaModelWrapper
from .flava_wrapper import FLAVAModelWrapper
from .siglip_wrapper import SigLIPModelWrapper
from .cogvlm_wrapper import CogVLMModelWrapper
from .qwenvlm_wrapper import QwenVLMModelWrapper
from .gemma_wrapper import GemmaModelWrapper

__all__ = [
    'CLIPModelWrapper',
    'CoCaModelWrapper', 
    'FLAVAModelWrapper',
    'SigLIPModelWrapper',
    'CogVLMModelWrapper',
    'QwenVLMModelWrapper',
    'GemmaModelWrapper',
] 