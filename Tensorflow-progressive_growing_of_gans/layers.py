import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
import numpy as np


#----------------------------------------------------------------------------
# Resize activation tensor 'v' of shape 'si' to match shape 'so'.

class ACTVResizeLayer(Layer):
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
# Generalized dropout layer. Supports arbitrary subsets of axes and different
# modes. Mainly used to inject multiplicative Gaussian noise in the network.

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
# Layer normalization. Custom reimplementation based on the paper:
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