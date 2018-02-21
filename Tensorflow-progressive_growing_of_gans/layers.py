import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
import numpy as np


#----------------------------------------------------------------------------
# Resize activation tensor 'inputs' of shape 'si' to match shape 'so'.
# TODO(naykun): Test this part.
class ACTVResizeLayer(Layer):
    def __init__(self,si,so,**kwargs):
        self.si = si
        self.so = so
        super(ACTVResizeLayer,self).__init__(**kwargs)
    def call(self, v, **kwargs):
        assert len(self.si) == len(self.so) and self.si[0] == self.so[0]

        # Decrease feature maps.  Attention: channels last
        if self.si[-1] > self.so[-1]:
            v = v[...,:so[-1]]

        # Increase feature maps.  Attention:channels last
        if self.si[-1] < self.so[-1]:
            z = K.zeros((self.so[:-1] + (self.so[-1] - self.si[-1])),dtype=v.dtype)
            v = K.concatenate([v,z])
        
        # Shrink spatial axis
        if len(self.si) == 4 and (self.si[1] > self.so[1] or self.si[2] > self.so[2]):
            assert self.si[1] % self.so[1] == 0 and self.si[2] % self.so[2] == 0
            pool_size = (self.si[1] / self.so[1],self.si[2] / self.so[2])
            strides = pool_size
            v = K.pool2d(v,pool_size=pool_size,strides=strides,padding='same',data_format='channels_last',pool_mode='avg')

        #Extend spatial axis
        for i in range(1,len(self.si) - 1):
            if self.si[i] < self.so[i]:
                assert self.so[i] % self.si[i] == 0
                v = K.repeat_elements(v,rep=2,axis=i)

        return v
    def compute_output_shape(self, input_shape):
        return self.so


#----------------------------------------------------------------------------
# Resolution selector for fading in new layers during progressive growing.
class LODSelectLayer(Layer):
    def __init__(self,output_dim,**kwargs):
        self.output_dim = output_dim
        super(LODSelectLayer,self).__init__(**kwargs)
    def build(self,input_shape):
        # TODO(naykun): Create a trainable weight variable for this layer.
        pass
    def call(self, inputs, **kwargs):
        # TODO(naykun): Implementation layer function
        pass
    def compute_output_shape(self, input_shape):
        # TODO(naykun): Help Keras can do automatic shape inference.
        pass



#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.
class PixelNormLayer(Layer):
    def __init__(self,output_dim,**kwargs):
        self.output_dim = output_dim
        super(PixelNormLayer,self).__init__(**kwargs)
    def build(self,input_shape):
        # TODO(naykun): Create a trainable weight variable for this layer.
        pass
    def call(self, inputs, **kwargs):
        # TODO(naykun): Implementation layer function
        pass
    def compute_output_shape(self, input_shape):
        # TODO(naykun): Help Keras can do automatic shape inference.
        pass




#----------------------------------------------------------------------------
# Applies equalized learning rate to the preceding layer.
class WScaleLayer(Layer):
    def __init__(self,output_dim,**kwargs):
        self.output_dim = output_dim
        super(WScaleLayer,self).__init__(**kwargs)
    def build(self,input_shape):
        # TODO(naykun): Create a trainable weight variable for this layer.
        pass
    def call(self, inputs, **kwargs):
        # TODO(naykun): Implementation layer function
        pass
    def compute_output_shape(self, input_shape):
        # TODO(naykun): Help Keras can do automatic shape inference.
        pass


#----------------------------------------------------------------------------
# Minibatch stat concatenation layer.
# - func is the function to use for the activations across minibatch
# - averaging tells how much averaging to use ('all', 'spatial', 'none')
class MinibatchStatConcatLayer(Layer):
    def __init__(self,output_dim,**kwargs):
        self.output_dim = output_dim
        super(MinibatchStatConcatLayer,self).__init__(**kwargs)
    def build(self,input_shape):
        # TODO(naykun): Create a trainable weight variable for this layer.
        pass
    def call(self, inputs, **kwargs):
        # TODO(naykun): Implementation layer function
        pass
    def compute_output_shape(self, input_shape):
        # TODO(naykun): Help Keras can do automatic shape inference.
        pass


#----------------------------------------------------------------------------
# Generalized dropout layer.  Supports arbitrary subsets of axes and different
# modes.  Mainly used to inject multiplicative Gaussian noise in the network.
class GDropLayer(Layer):
    def __init__(self,output_dim,**kwargs):
        self.output_dim = output_dim
        super(GDropLayer,self).__init__(**kwargs)
    def build(self,input_shape):
        # TODO(naykun): Create a trainable weight variable for this layer.
        pass
    def call(self, inputs, **kwargs):
        # TODO(naykun): Implementation layer function
        pass
    def compute_output_shape(self, input_shape):
        # TODO(naykun): Help Keras can do automatic shape inference.
        pass


#----------------------------------------------------------------------------
# Layer normalization.  Custom reimplementation based on the paper:
# https://arxiv.org/abs/1607.06450
class LayerNormLayer(Layer):
    def __init__(self,output_dim,**kwargs):
        self.output_dim = output_dim
        super(LayerNormLayer,self).__init__(**kwargs)
    def build(self,input_shape):
        # TODO(naykun): Create a trainable weight variable for this layer.
        pass
    def call(self, inputs, **kwargs):
        # TODO(naykun): Implementation layer function
        pass
    def compute_output_shape(self, input_shape):
        # TODO(naykun): Help Keras can do automatic shape inference.
        pass