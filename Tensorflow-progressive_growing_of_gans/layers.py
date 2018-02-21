import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
import numpy as np


#----------------------------------------------------------------------------
# Resize activation tensor 'inputs' of shape 'si' to match shape 'so'.
class ACTVResizeLayer(Layer):
    def __init__(self,si,so,**kwargs):
        self.si = si
        self.so = so
        super(ACTVResizeLayer,self).__init__(**kwargs)
    def call(self, inputs, **kwargs):
        # TODO(naykun): Implementation layer function
        assert len(si) == len(so) and si[0] == so[0]

        # Decrease feature maps.  Attention: channels last
        if si[-1] > so[-1]:
            inputs = inputs[...,:so[-1]]

        # Increase feature maps Attention:channels last
        if si[-1] < so[-1]:
            z = K.zeros((so[:-1]+(so[-1]-si[-1])),dtype=inputs.dtype)
            inputs = K.concatenate([inputs,z])
        
        # Shrink spatial axis 
        if len(si) == 4 and (si[1] > so[1] or si[2] > so[2]):
            assert si[1] % so[1] == 0 and si[2] % so[2] == 0
            pool_size = (si[1] / so[1],si[2] / so[2])
            strides = pool_size
            outputs = K.pool2d(inputs,pool_size=pool_size,strides=strides,padding='same',data_format='channels_last',pool_mode='avg')

        #Extend spatial axis
        for i in range(1,len(si)-1):
            if si[i] < so[i]:
                assert so[i]%si[i] == 0
                outputs = K.repeat_elements(inputs,rep=2,axis=i)

        return outputs
    def compute_output_shape(self, input_shape):
        # TODO(naykun): Help Keras do automatic shape inference.
        return so


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