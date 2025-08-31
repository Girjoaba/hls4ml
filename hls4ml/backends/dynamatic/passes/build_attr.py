# Typing imports
from __future__ import annotations # makes all annotations into strings
from typing import Literal, Optional, Callable, TYPE_CHECKING
from numpy.typing import NDArray
if TYPE_CHECKING:
    from hls4ml.model.graph import ModelGraph
    from hls4ml.model.layers import Layer


from hls4ml.model.optimizer import OptimizerPass

from functools import wraps
import numpy as np
from fxpmath import Fxp


class DynamaticAttrBuilder:
    """A helper class that sets Dynamatic specific attributes for the layers of the original ModelGraph.
    In doing so, we simplify the process of creating new optimization passes 
    and constructing the writer class. 
    The new attributes must be accessed with '.get_attr(...)'

    New attributes:
        write_weights (bool): the layer contains weights that should be explicitly defined in the project file
        write_dims (bool):    the layer dimensions should be explicitly written in the project file
        write_func (bool):    the layer has a corresponding function call that should be explicitly written
                              as part of the NN architecture in the project file
        func_call (str):      the corresponding layer C function call 

        in_dim_key, out_dim_key (str): the variable name containing the layer dimensions (that goes in and out the layer)
        in_dim_val, out_dim_val (int): the value of each layer dimension (that goes in and out the layer)

        fxp_weights (np.ndarray): already quantized weight matrix
        fxp_bias (np.ndarray):    already quantized bias vector

        in_width, in_frac (str): parameters used for fixed point computation
                                 the parameters of the input vector 
                                 bit width, fraction
        out_width, out_frac (str): parameters used for fixed point computation
                                   the parameters of the output vector 
                                   bit width, fraction

        in_type, out_type (str): the associated inferred types in the C frontend code

    Args:
        node (Layer): A layer of the model graph
    """

    def __init__(self, node) -> None:
        self.node = node

    @staticmethod
    def attach_to_node(attr_name: Optional[str] = None) :
        """A decorator-factory to easily chain 'set_attr' commands to the node.
        It calls the provided function. This eliminates a lot of boiler plate code.
        All the added attributes can be chained in one call since the wrapped function returns self.
        """
        def decorator(fn) -> Callable:
            name = attr_name or fn.__name__
            @wraps(fn)
            def wrapped(self, *args, **kwargs):
                val = fn(self, *args, **kwargs)
                self.node.set_attr(name, val)
                return self
            return wrapped
        return decorator
    
    @attach_to_node()
    def write_weights(self) -> bool:
        return self.node.class_name in ['Dense']

    @attach_to_node()
    def write_dims(self) -> bool:
        return self.node.class_name in ['Input', 'Dense']
    
    @attach_to_node()
    def write_func(self) -> bool:
        return self.node.class_name in ['Dense', 'Activation', 'Softmax']
    
    
    @attach_to_node()
    def in_dim_key(self, v: str) -> str:
        return v
    
    @attach_to_node()
    def in_dim_val(self, v: int) -> int:
        return v
    
    @attach_to_node()
    def out_dim_key(self, v: str) -> str:
        return v
    
    @attach_to_node()
    def out_dim_val(self, v: int) -> int:
        return v
    
    @attach_to_node()
    def fxp_weights(self, weights, out_dim: int, in_dim: int) -> NDArray[NDArray[np.int_]]:
        #TODO: check which element in the precision array should we take Currently we assume the precision of weights is the first elem.
        # has weights
        if len(weights) >= 1:
            width = self.node.get_attr('in_width')
            frac = self.node.get_attr('in_frac')

            mat = np.array(list(list(weights)[0])).reshape(in_dim, out_dim)
            mat_T = mat.T   # in Keras the weights are transposed
            fxp_w: NDArray[NDArray[np.int_]] = Fxp(mat_T, signed=True, n_word=width, n_frac=frac).raw()
            return fxp_w 
        return np.array([])
    
    @attach_to_node()
    def fxp_bias(self, weights) -> NDArray[np.int_]:
        #TODO: check which element in the precision array should we take Currently we assume the precision of weights is the first elem.
        # has bias
        if len(weights) >= 2:
            width = self.node.get_attr('in_width')
            frac = self.node.get_attr('in_frac')

            fxp_b: NDArray[np.int_] = Fxp(list(list(weights)[1]), signed=True, n_word=width, n_frac=frac).raw()
            return fxp_b
        return np.array([])
    
    @attach_to_node()
    def in_width(self, prev_layer_precision: dict | None) -> str: # TODO: right now we only care about the first defined type in the list
        if prev_layer_precision:
            for _, type_var in prev_layer_precision.items():
                return int(type_var.precision.width)
        return ''
    
    @attach_to_node()
    def in_frac(self, prev_layer_precision: dict | None) -> str:
        if prev_layer_precision:
            for _, type_var in prev_layer_precision.items():
                return int(type_var.precision.width - type_var.precision.integer)
        return ''
    
    @attach_to_node()
    def out_width(self, layer_precision: dict) -> str:
        if layer_precision.get('result_t', False):
            width = layer_precision['result_t'].precision.width
            return int(width)
        for _, type_var in layer_precision.items():
            return int(type_var.precision.width)
        return ''

    @attach_to_node()
    def out_frac(self, layer_precision) -> str:
        if layer_precision.get('result_t', False):
            width = layer_precision['result_t'].precision.width
            integer = layer_precision['result_t'].precision.integer
            return int(width - integer)
        for _, type_var in layer_precision.items():
            return int(type_var.precision.width - type_var.precision.integer)
        return ''
    
    @attach_to_node()
    def in_type(self) -> str:
        return f'default_t'
    
    @attach_to_node()
    def out_type(self) -> str:
        return f'default_t'

    @attach_to_node()
    def func_call(self) -> str:
        func_call_str = ''
        if self.node.class_name == 'Dense':
            func_call_str = f'DENSE_LAYER'
        
        elif self.node.class_name == 'Activation':
            func_call_str = f''

        elif self.node.class_name == 'Softmax':
            implementation = dict(self.node.attributes).get('implementation', 'stable')
            if implementation == 'stable':
                #TODO: implement an approximation with look-up tables for softmax
                func_call_str = f'ARGMAX'
            elif implementation == 'latency':
                table_size = dict(self.node.attributes)['table_size']
                func_call_str = f'ARGMAX'
            elif implementation == 'argmax':
                func_call_str = f'ARGMAX'
        return func_call_str
    
    
class BuildAttr(OptimizerPass):
    """Builds the Dynamatic specific attributes for all layers.
    """

    def match(self, node: Layer) -> bool:
        if node.class_name == 'Input':
            return True
        return False

    def transform(self, model: ModelGraph, node: Layer) -> Literal[False]:        
        prev_out_dim_key = ''
        prev_out_dim_val = -1
        prev_layer_precision = None

        for layer in model.get_layers():
            curr_out_dim_key: str = list(layer.get_output_variable().get_shape())[0][0]
            curr_out_dim_val: int = list(layer.get_output_variable().get_shape())[0][1]
            curr_weights = layer.get_weights()
            curr_prec: dict = layer.get_layer_precision()

            # uses the builder to add all the attributes
            b = DynamaticAttrBuilder(layer)
            (b
                .write_dims()
                .write_weights()
                .write_func()
                .in_dim_key(prev_out_dim_key)
                .in_dim_val(prev_out_dim_val)
                .out_dim_key(curr_out_dim_key)
                .out_dim_val(curr_out_dim_val)
                .in_width(prev_layer_precision)
                .in_frac(prev_layer_precision)
                .out_width(curr_prec)
                .out_frac(curr_prec)
                .in_type()
                .out_type()
                .fxp_weights(curr_weights, out_dim=curr_out_dim_val, in_dim=prev_out_dim_val)
                .fxp_bias(curr_weights)
                .func_call()

            )

            prev_out_dim_key = curr_out_dim_key
            prev_out_dim_val = curr_out_dim_val
            prev_layer_precision = curr_prec

        return False