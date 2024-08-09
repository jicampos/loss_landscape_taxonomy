"""
Compute Bit Operations (BOPs) for PyTorch and HAWQ layers.
"""
import math
import logging
from typing import Any, Optional
import numpy as np
import torch
import torch.nn as nn
from hawq.utils.quantization_utils.quant_modules import QuantLinear, QuantConv2d, QuantAct, QuantBnConv2d


class BitOperations:

    def __init__(self, 
        model : nn.Module, 
        input_shape : list, 
        input_precision : Optional[int] = 32
    ) -> None:
        self.model = model 
        self.input_shape = input_shape
        self.input_precision = input_precision
        self.hooks = list()
        self.register_layers = list()
        self.layer_total = dict()
        self.layer_alive = dict()

        self._setup_model()

    def _setup_model(self) -> None:
        """Collect input/output shapes & flag quantized layers"""
        self.hooks = self.register_forward_hooks(self.model, 'model', self.hooks)
        # forward pass to trigger hooks
        self.model(torch.randn(self.input_shape))
        self.remove_forward_hooks()
        self.print_registered_layers()

    def count_nonzero_weights(self) -> None:
        for module_name in self.register_layers:
            module = self._get_layer(module_name, self.model)
            nonzero = torch.count_nonzero(module.weight)
            total_params = torch.prod(torch.prod(torch.tensor(module.weight.shape)))
            print(f"{module_name+'.weight':40} | nonzeros = {nonzero:7} / {total_params:7} ({100 * nonzero / total_params:6.2f}%)| total_pruned = {total_params - nonzero :7} | shape = {module.weight.shape}")
            
            self.layer_total[module_name+".weight"] = total_params
            self.layer_alive[module_name+".weight"] = nonzero

            if hasattr(module, "bias"):
                nonzero = torch.count_nonzero(module.bias)
                total_params = torch.prod(torch.prod(torch.tensor(module.bias.shape)))
                print(f"{module_name+'.bias':40} | nonzeros = {nonzero:7} / {total_params:7} ({100 * nonzero / total_params:6.2f}%)| total_pruned = {total_params - nonzero :7} | shape = {module.weight.shape}")
                
                self.layer_total[module_name+".bias"] = total_params
                self.layer_alive[module_name+".bias"] = nonzero

    def compute(self, weight_bitwidth : int) -> float:
        """Compute BOPs. Strong assumptions: 1) weight bitwidth is constant and 2) activations are weight_bitwidth+3"""
        b_w = weight_bitwidth              # weight bitwidth 
        b_a = self.input_precision  # activation bitwidth 
        total_bops = 0
        self.count_nonzero_weights()

        for module_name in self.register_layers:
            module = self._get_layer(module_name, self.model)
            b_w = weight_bitwidth if module.is_quantized else 32
            if isinstance(module, (nn.Linear, QuantLinear)):
                n = module.in_features
                m = module.out_features
                k = 1
            elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, QuantConv2d, QuantBnConv2d)):
                n = np.prod(module.input_shape)
                m = np.prod(module.output_shape)
                k = np.prod(module.kernel_size) if len(module.kernel_size) > 1 else module.kernel_size**2
            
            total = self.layer_total[module_name+".weight"] + self.layer_total[module_name+".bias"]
            alive = self.layer_alive[module_name+".weight"] + self.layer_alive[module_name+".bias"]
            p = 1 - ((total-alive) / total)  # fraction remaining 

            module_bops = m * n * k * k * (p * b_a * b_w + b_a + b_w + math.log2(n * k * k))
            print("{} BOPS: {} = {}*{}*{}({}*{}*{} + {} + {} + {})".format(module_name,module_bops,m,n,k*k,p,b_a,b_w,b_a,b_w,math.log2(n*k*k)))
            total_bops += module_bops

            b_a = b_w + 3 if module.is_quantized else 32
        print(f"Total BOPs: {total_bops}")
        return total_bops

    def _register_layer(self, name : str) -> None:
        if name not in self.register_layers:
            self.register_layers.append(name)

    @staticmethod
    def forward_hook(module, input, output) -> None:
        """Store the input and output shape"""
        if len(output) == 2:
            output, _ = output
        print(f"[{module.__class__.__name__}] Forward hook: Input shape: {input[0].shape}, Output shape: {output.shape}")
        module.input_shape = input[0].shape
        module.output_shape = output.shape

    def register_forward_hooks(self, model : nn.Module, module_name : str, hooks : list) -> list:
        """Register a forward hook on each layer to collect layer information"""
        for name, module in model.named_children():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):  # nn.MaxPool2d, nn.Flatten
                module.is_quantized = False
                hooks.append(module.register_forward_hook(self.forward_hook))
                self._register_layer(module_name+"."+name)
            elif isinstance(module, (QuantLinear, QuantConv2d, QuantBnConv2d)):
                module.is_quantized = True
                hooks.append(module.register_forward_hook(self.forward_hook))
                self._register_layer(module_name+"."+name)
            elif isinstance(module, (QuantAct)):
                continue
            elif isinstance(module, (nn.Sequential, nn.Module)):  # For custom modules and nested sequential models 
                hooks += self.register_forward_hooks(model=module, module_name=module_name+"."+name, hooks=hooks)
        return hooks 

    def remove_forward_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()

    def _get_layer(self, layer_name : str, module : nn.Module) -> nn.Module:
        for name in layer_name.split('.')[1:]:
            module = getattr(module, name)
        return module
    
    def profile(self) -> None:
        if len(self.layer_alive) == 0:
            print('Need to compute bit operations before profile!')
            return 
        
        for layer_name in self.layer_alive.keys():
            nonzero = self.layer_alive[layer_name]
            total_params = self.layer_total[layer_name]
            print(f"{layer_name:40} | nonzeros = {nonzero:7} / {total_params:7} ({100 * nonzero / total_params:6.2f}%)| total_pruned = {total_params - nonzero :7}")

    def print_registered_layers(self) -> None:
        """Sanity check - print all registered layers"""
        print('Registered Layers:')
        for module_name in self.register_layers:
            print(f"\t{module_name}")
        
    def __del__(self) -> None:
        self._cleanup()
    
    def _cleanup(self) -> None:
        """Remove added attributes/flags from setup procedure"""
        for module_name in self.register_layers:
            module = self._get_layer(module_name, self.model)
            delattr(module, "is_quantized")
            delattr(module, "input_shape")
            delattr(module, "output_shape")


if __name__ == "__main__":
    model = nn.Sequential(
        nn.Linear(16, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 5),
        nn.Softmax()
    )

    bops = BitOperations(model, input_shape=[1, 16], input_precision=32)
    bops.compute(weight_bitwidth=8)
    bops._cleanup()  # Optional: removes added attributes/flags 
    print('======================================================================')
    bops.profile()
