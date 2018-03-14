import tensorflow as tf
import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
from keras.layers.merge import _Merge
from keras.layers import *
from keras import activations
from keras import initializers
from keras.models import Model
import numpy as np
from layers import *

linear, linear_init = activations.linear,       initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal')
relu,   relu_init = activations.relu,         initializers.he_normal()
lrelu,  lrelu_init = lambda x: K.relu(x, 0.2),  initializers.he_normal()


def vlrelu(x): return K.relu(x, 0.3)


def G_convblock(net,
        num_filter,
        filter_size,
        actv,
        init,
        pad='same',
        use_wscale=True,
        use_pixelnorm=True,
        use_batchnorm=False,
        name=None):
    if pad == 'full':
        pad = filter_size - 1
    Pad = ZeroPadding2D(pad, name=name + 'Pad')
    net = Pad(net)
    Conv = Conv2D(num_filter, filter_size, padding='same',
                  activation=actv, kernel_initializer=init, name=name)
    net = Conv(net)
    if use_wscale:
        Wslayer = WScaleLayer(Conv, name=name + 'WS')
        net = Wslayer(net)
    if use_batchnorm:
        Bslayer = BatchNormalization(name=name + 'BN')
        net = Bslayer(net)
    if use_pixelnorm:
        Pixnorm = PixelNormLayer(name=name + 'PN')
        net = Pixnorm(net)
    return net


def NINblock(net,
        num_channels,
        actv,
        init,
        use_wscale=True,
        name=None):
    NINlayer = Conv2D(num_channels, 1, padding='same',
                      activation=actv, kernel_initializer=init, name=name + 'NIN')
    net = NINlayer(net)
    if use_wscale:
        Wslayer = WScaleLayer(NINlayer, name=name + 'NINWS')
        net = Wslayer(net)
    return net


def Generator(num_channels=1,
        resolution=32,
        label_size=0,
        fmap_base=4096,
        fmap_decay=1.0,
        fmap_max=256,
        latent_size=None,
        normalize_latents=True,
        use_wscale=True,
        use_pixelnorm=True,
        use_leakyrelu=True,
        use_batchnorm=False,
        tanh_at_end=None,
        **kwargs):
    R = int(np.log2(resolution))
    assert resolution == 2 ** R and resolution >= 4
    cur_lod = K.variable(np.float32(0.0), dtype='float32', name='cur_lod')

    def numf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    if latent_size is None:
        latent_size = numf(0)
    (act, act_init) = (lrelu, lrelu_init) if use_leakyrelu else (relu, relu_init)

    inputs = [Input(shape=[None, latent_size], name='Glatents')]
    net = inputs[-1]
    if normalize_latents:
        net = PixelNormLayer(name='Gnorm')(net)
    if label_size:
        inputs += [Input(shape=[None, label_size], name='Glabels')]
        net = Concatenate(name='Gina')([net, inputs[-1]])
    net = Reshape((None, -1, 1, 1), name='G1nb')(net)

    net = G_convblock(net, numf(1), 4, act, act_init, pad='full', use_wscale=use_wscale,
                      use_batchnorm=use_batchnorm, use_pixelnorm=use_pixelnorm, name='G1a')
    net = G_convblock(net, numf(1), 3, act, act_init, pad=1, use_wscale=use_wscale,
                      use_batchnorm=use_batchnorm, use_pixelnorm=use_pixelnorm, name='G1b')
    lods = [net]
    for I in range(2, R):
        net = UpSampling2D(2, name='G%dup' % I)(net)
        net = G_convblock(net, numf(I), 3, act, act_init, pad=1, use_wscale=use_wscale,
                          use_batchnorm=use_batchnorm, use_pixelnorm=use_pixelnorm, name='G%da' % I)
        net = G_convblock(net, numf(I), 3, act, act_init, pad=1, use_wscale=use_wscale,
                          use_batchnorm=use_batchnorm, use_pixelnorm=use_pixelnorm, name='G%db' % I)
        lods += [net]

    lods = [NINblock(l, num_channels, linear, linear_init, use_wscale=use_wscale,
                     name='Glod%d' % i) for i, l in enumerate(reversed(lods))]
    output = LODSelectLayer(cur_lod, name='Glod')(lods)
    if tanh_at_end is not None:
        output = Activation('tanh', name='Gtanh')(output)
        if tanh_at_end != 1.0:
            output = Lambda(lambda x: x * tanh_at_end, name='Gtanhs')

    model = Model(inputs=inputs, outputs=[output])


def Discriminator(num_channels=1,        # Overridden based on dataset.
        resolution=32,       # Overridden based on dataset.
        label_size=0,        # Overridden based on dataset.
        fmap_base=4096,
        fmap_decay=1.0,
        fmap_max=256,
        mbstat_func='Tstdeps',
        mbstat_avg='all',
        mbdisc_kernels=None,
        use_wscale=True,
        use_gdrop=True,
        use_layernorm=False,
        **kwargs):

    epsilon = 0.01
    R = int(np.log2(resolution))
    assert resolution == 2 ** R and resolution >= 4
    cur_lod = K.variable(np.float(0.0), dtype='float32', name='cur_lod')
    gdrop_strength = K.variable(np.float(0.0), dtype='float32', name='gdrop_strength')

    def numf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

    def GD(incoming):
        return GDropLayer(name='gd', mode='prop', strength=gdrop_strength)(incoming) if use_gdrop else incoming

    def WS(layer):
        return WScaleLayer(layer, name=layer.name + 'ws') if use_wscale else layer

    def NINBlock(net,
            num_channels,
            actv,
            init,
            name=None):
        layer = Conv2D(num_channels, 1, activation=actv,
                       kernel_initializer=init, pad='same', name=name + 'NIN')
        net = layer(net)
        if use_wscale:
            layer = WScaleLayer(layer, name=name + 'NINWS')
            net = layer(net)
        return net

    def Downscale2DLayer(incoming, scale_factor, **kwargs):
        return AveragePooling2D(pool_size=scale_factor, **kwargs)(incoming)

    def ConvBlock(net,
            num_filter,
            filter_size,
            actv,
            init,
            pad,
            name=None):
        net = GD(net)
        layer = Conv2D(num_filter, filter_size, activation=actv,
                       kernel_initializer=init, padding=pad, name=name)
        net = layer(net)
        if use_wscale:
            layer = WScaleLayer(layer, name=name + 'ws')
            net = layer(net)
        if use_layernorm:
            layer = LayerNormLayer(layer, epsilon, name=name + 'ln')
        return net

    inputs = Input(shape=[None, num_channels, 2 ** R, 2 ** R], name='Dimages')
    net = NINBlock(inputs, numf(R - 1), lrelu, lrelu_init, name='D%dx' % (R - 1))
    for i in range(R - 1, 1, -1):
        net = ConvBlock(net, numf(i), 3, lrelu, lrelu_init, 1, name='D%db' % i)
        net = ConvBlock(net, numf(i - 1), 3, lrelu,
                        lrelu_init, 1, name='D%da' % i)
        net = Downscale2DLayer(net, name='D%ddn' % i, scale_factor=2)
        lod = Downscale2DLayer(inputs, name='D%dxs' % (i - 1), scale_factor=2 ** (R - i))
        lod = NINBlock(lod, numf(i - 1), lrelu, relu_init, name='D%dx' % (i - 1))
        net = LODSelectLayer(cur_lod, name='D%dlod' % (i - 1), first_incoming_lod=R - i - 1)([net, lod])
    if mbstat_avg is not None:
        net = MinibatchStatConcatLayer(num_kernels=mbdisc_kernels, name='Dmd')(net)

    output_layers = [WS(Dense(1, activation=linear, kernel_initializer=linear_init))(net)]
    if label_size:
        output_layers += [WS(Dense(label_size, activation=linear,
                                   kernel_initializer=linear_init, name='Dlabels')(net))]

    model = Model(inputs=[inputs], outputs=output_layers)
