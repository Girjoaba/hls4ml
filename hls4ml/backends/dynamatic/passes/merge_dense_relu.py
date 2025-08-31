# Typing imports
from __future__ import annotations # makes all annotations into strings
from typing import List, Literal, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from hls4ml.model.graph import ModelGraph
    from hls4ml.model.layers import Layer

from hls4ml.model.optimizer import OptimizerPass

    
class MergeDenseRelu(OptimizerPass):
    """Merges a dense layer followed by a relu layer in one layer by
    applying the relu function immediately after each dot product. 

    This optimization is useful because it removes a RAW dependecy in between the Dense and ReLU layers.
    If we appy ReLU right after the dot product in the
    Matrix-Vector Multiplication, we do not require an additional store.
    """

    def match(self, node: Layer) -> bool:
        """Match any dense layers.
        
        Args:
            node (Layer): each layer of the neural networked will be called.
        Returns:
            bool: True if it matches a dense layer, False otherwise.
        """
        if node.class_name == 'Dense':
            return True
        return False

    def transform(self, model: ModelGraph, node: Layer) -> Literal[False]:        
        """If any matched layer is followed by ReLU, apply the activation function during the 
        MatVec itself by merging the layers.

        Args:
            model (ModelGraph): contains the neural network information.
            node (Layer): the matched layer.
        Returns:
            bool: False because we do not change the network architecture.
        """
        layers: list[Layer] = list(model.get_layers())
        for i, layer in enumerate(layers[:-1]):
            next_layer = layers[i + 1]
            if layer == node and next_layer.class_name == 'Activation':
                new_func_call = f'DENSE_RELU_LAYER'
                layer.set_attr('func_call', new_func_call)
                next_layer.set_attr('write_func', False)

        return False