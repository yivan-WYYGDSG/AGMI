from .basic_fusion_head import BaseFusionHead
from .tcnn_fusion_head import TcnnFusionHead
from .attn_head import GatedAttention
from .agmi_head import AGMIFusionHead

__all__ = ['BaseFusionHead', 'TcnnFusionHead', 'GatedAttention', 'AGMIFusionHead']