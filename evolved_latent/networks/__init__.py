"""
This module contains the code for the networks used in the project.
"""

from .autoencoder_baseline import BaselineAutoencoder
from .autoencoder_resnet_norm import ResNetNormAutoencoder
from .autoencoder_resnet import ResNetAutoencoder
from .autoencoder_res_attn import ResNetAttentionAutoencoder
from .autoencoder_res_attn_qk import ResNetAttentionQKAutoencoder

# Seq2Seq
from .seq2seq_transformer import Seq2SeqTransformer
