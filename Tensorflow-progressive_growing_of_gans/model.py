import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
from keras.layers.merge import _Merge
from keras import activations
import numpy as np
from layers import *

def Generator(
    num_channels        = 1,
    resolution          = 32,
    label_size          = 0,
    fmap_base           = 4096,
    fmap_decay          = 1.0,
    fmap_max            = 256,
    latent_size         = None,
    normalize_latents   = True,
    use_wscale          = True,
    use_pixelnorm       = True,
    use_leakyrelu       = True,
    use_batchnorm       = False,
    tanh_at_end         = None,
    **kwargs):
    pass


def Discriminator(
    num_channels    = 1,        # Overridden based on dataset.
    resolution      = 32,       # Overridden based on dataset.
    label_size      = 0,        # Overridden based on dataset.
    fmap_base       = 4096,
    fmap_decay      = 1.0,
    fmap_max        = 256,
    mbstat_func     = 'Tstdeps',
    mbstat_avg      = 'all',
    mbdisc_kernels  = None,
    use_wscale      = True,
    use_gdrop       = True,
    use_layernorm   = False,
    **kwargs):   
    pass