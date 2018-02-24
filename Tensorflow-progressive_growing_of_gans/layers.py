import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
from keras.layers.merge import _Merge
from keras import activations
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
class LODSelectLayer(_Merge):
    def __init__(self,cur_lod,first_incoming_lod=0,ref_idx=0, min_lod=None, max_lod=None,**kwargs):
        super(LODSelectLayer,self).__init__(**kwargs)
        self.cur_lod = cur_lod
        self.first_incoming_lod = first_incoming_lod
        self.ref_idx = ref_idx
        self.min_lod = min_lod
        self.max_lod = max_lod

    def _merge_function(self, inputs):
        self.input_shapes = [K.int_shape(input) for input in inputs]
        v = [ACTVResizeLayer(K.int_shape(input), self.input_shapes[self.ref_idx])(input) for input in inputs]
        lo = np.clip(int(np.floor(self.min_lod - self.first_incoming_lod)), 0, len(v)-1) if self.min_lod is not None else 0
        hi = np.clip(int(np.ceil(self.max_lod - self.first_incoming_lod)), lo, len(v)-1) if self.max_lod is not None else len(v)-1
        t = self.cur_lod - self.first_incoming_lod
        r = v[hi]
        for i in range(hi-1, lo-1, -1): # i = hi-1, hi-2, ..., lo
            r = K.switch(K.less(t, i+1), v[i] * ((i+1)-t) + v[i+1] * (t-i), r)
        if lo < hi:
            r = K.switch(K.less_equal(t, lo), v[lo], r)
        return r
    def compute_output_shape(self, input_shape):
        return self.input_shapes[self.ref_idx]



#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.
class PixelNormLayer(Layer):
    def __init__(self,**kwargs):
        super(PixelNormLayer,self).__init__(**kwargs)
    def call(self, inputs, **kwargs):
        return inputs / K.sqrt(K.mean(v**2, axis=1, keepdims=True) + 1.0e-8)
    def compute_output_shape(self, input_shape):
        return input_shape




#----------------------------------------------------------------------------
# Applies equalized learning rate to the preceding layer.
class WScaleLayer(Layer):
    def __init__(self,incoming,activation = None,**kwargs):
        self.incoming = incoming
        self.activation = activations.get(activation)
        super(WScaleLayer,self).__init__(**kwargs)
    def build(self,input_shape):
        kernel = K.get_value(self.incoming.kernel)
        scale = np.sqrt(np.mean(kernel ** 2))
        K.set_value(self.incoming.kernel,kernel/scale)
        self.scale=self.add_weight(name = 'scale',shape = scale.shape,trainable=False)
        K.set_value(self.scale,scale)
        if  hasattr(self.incoming, 'bias') and self.incoming.bias is not None:
            bias = K.get_value(self.incoming.bias)
            self.bias=self.add_weight(name = 'bias',shape = bias.shape)
            del self.incoming.trainable_weights[self.incoming.bias]
            self.incoming.bias = None
        
    def call(self, input, **kwargs):
        input = input * self.scale
        if self.bias is not None:
            pattern = ['x'] + ['x'] * (K.ndim(input) - 2)+[0]
            input = input + K.permute_dimensions(self.bias,*pattern)
        return self.activation(v)
    def compute_output_shape(self, input_shape):
        return input_shape


#----------------------------------------------------------------------------
# Minibatch stat concatenation layer.
# - func is the function to use for the activations across minibatch
# - averaging tells how much averaging to use ('all', 'spatial', 'none')
class MinibatchStatConcatLayer(Layer):
    def __init__(self,averaging = 'all',**kwargs):
        self.averaging = averaging.lower()
        super(MinibatchStatConcatLayer,self).__init__(**kwargs)
        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in ['all', 'flat', 'spatial', 'none', 'gpool'], 'Invalid averaging mode'%self.averaging
        self.adjusted_std = lambda x, **kwargs: K.sqrt(K.mean((x - K.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)
    def call(self, input, **kwargs):
        s = list(K.int_shape(input))
        vals = self.adjusted_std(input,axis=0,keepdims=True)                # per activation, over minibatch dim
        if self.averaging == 'all':                                 # average everything --> 1 value per minibatch
            vals = K.mean(vals,keepdims=True)
            reps = s; reps[-1]=1
            vals = K.tile(vals,reps)
        elif self.averaging == 'spatial':                           # average spatial locations
            if len(s) == 4:
                vals = K.mean(vals,axis=(1,2),keepdims=True)
            reps = s; reps[-1]=1
            vals = K.tile(vals,reps)
        elif self.averaging == 'none':                              # no averaging, pass on all information
            vals = K.repeat_elements(vals,rep=s[0],axis=0)
        elif self.averaging == 'gpool':                             # EXPERIMENTAL: compute variance (func) over minibatch AND spatial locations.
            if len(s) == 4:
                vals = self.adjusted_std(input,axis=(0,1,2),keepdims=True)
            reps = s; reps[-1]=1
            vals = K.tile(vals,reps)
        elif self.averaging == 'flat':
            vals = self.adjusted_std(input,keepdims=True)                   # variance of ALL activations --> 1 value per minibatch
            reps = s; reps[-1]=1
            vals = K.tile(vals,reps)
        elif self.averaging.startswith('group'):                    # average everything over n groups of feature maps --> n values per minibatch
            n = int(self.averaging[len('group'):])
            vals = vals.reshape((1, s[1], s[2], n,s[3]/n))
            vals = K.mean(vals, axis=(1,2,4), keepdims=True)
            vals = vals.reshape((1, 1, 1,n))
            reps = s; reps[-1] = 1
            vals = K.tile(vals, reps)
        else:
            raise ValueError('Invalid averaging mode', self.averaging)
        return K.concatenate([input, vals], axis=1)
    def compute_output_shape(self, input_shape):
        s = list(input_shape)
        if self.averaging == 'all': s[-1] += 1
        elif self.averaging == 'flat': s[-1] += 1
        elif self.averaging.startswith('group'): s[-1] += int(self.averaging[len('group'):])
        else: s[-1] *= 2
        return tuple(s)


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