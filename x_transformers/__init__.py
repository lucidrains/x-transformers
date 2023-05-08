import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse('2.0.0'):
    from einops._torch_specific import allow_ops_in_compiled_graph
    allow_ops_in_compiled_graph()

from x_transformers.x_transformers import XTransformer, Encoder, Decoder, CrossAttender, Attention, TransformerWrapper, ViTransformerWrapper, ContinuousTransformerWrapper

from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
from x_transformers.nonautoregressive_wrapper import NonAutoregressiveWrapper
from x_transformers.continuous_autoregressive_wrapper import ContinuousAutoregressiveWrapper
from x_transformers.xl_autoregressive_wrapper import XLAutoregressiveWrapper
