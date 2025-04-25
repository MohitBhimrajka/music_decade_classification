# src/models.py (REVISED for Normalization Layer Flexibility)

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# --- Constants ---
N_FEATURES = 90
N_CLASSES = 10 # 10 decades (0-9)

# --- Activation Function Mapping (Keep from previous revision) ---
ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'gelu': nn.GELU,
    'selu': nn.SELU,
}


class BaseModel(nn.Module):
    """A base class for easier model summary printing (optional)."""
    def __init__(self):
        super().__init__()

    def __str__(self):
        model_str = f"Model Architecture: {self.__class__.__name__}\n"
        try:
            model_str += f"  Input Size: {getattr(self, 'input_size', 'N/A')}\n"
            model_str += f"  Num Classes: {getattr(self, 'num_classes', 'N/A')}\n"
            model_str += f"  Activation: {getattr(self, 'activation_name', 'N/A')}\n"
            model_str += f"  Normalization: {getattr(self, 'norm_layer_type', 'None')}\n" # Added Norm Type
            total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            model_str += f"Total Trainable Parameters: {total_params:,}\n"
            # Print layer structure (optional, can be verbose)
            # for name, layer in self.named_children():
            #     model_str += f"  ({name}): {layer}\n"
        except Exception as e:
            logging.warning(f"Could not generate full model string representation: {e}")
            model_str += "  (Could not retrieve detailed attributes)\n"
        return model_str

# Helper function to create norm layer
def get_norm_layer(norm_type, num_features):
    if norm_type == 'batch':
        return nn.BatchNorm1d(num_features)
    elif norm_type == 'layer':
        return nn.LayerNorm(num_features)
    elif norm_type is None or norm_type == 'none':
        return nn.Identity() # No normalization
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")

# --- Architecture 2: Wider Layers (Revised for Normalization) ---
class Model_2(BaseModel):
    def __init__(self, input_size=N_FEATURES, num_classes=N_CLASSES,
                 activation_fn=nn.ReLU, activation_name='ReLU',
                 norm_layer_type=None): # Add norm_layer_type argument
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.activation_name = activation_name
        self.norm_layer_type = norm_layer_type

        # Layer 1
        self.layer_1 = nn.Linear(input_size, 256)
        self.norm_1 = get_norm_layer(norm_layer_type, 256) # Norm after linear
        self.activation_1 = activation_fn()               # Activation after norm

        # Layer 2
        self.layer_2 = nn.Linear(256, 256)
        self.norm_2 = get_norm_layer(norm_layer_type, 256) # Norm after linear
        self.activation_2 = activation_fn()               # Activation after norm

        # Output Layer (typically no norm/activation before final output for CrossEntropyLoss)
        self.output_layer = nn.Linear(256, num_classes)

    def forward(self, x):
        # Layer 1 block
        x = self.layer_1(x)
        x = self.norm_1(x)
        x = self.activation_1(x)

        # Layer 2 block
        x = self.layer_2(x)
        x = self.norm_2(x)
        x = self.activation_2(x)

        # Output
        x = self.output_layer(x)
        return x


# --- Example Usage (Update if needed, Model_1/Model_3 not revised here) ---
if __name__ == '__main__':
    print("--- Example Instantiations (Model_2 Revised) ---")

    model_relu_nonorm = Model_2(activation_fn=nn.ReLU, activation_name='ReLU', norm_layer_type=None)
    print(model_relu_nonorm)

    model_gelu_batchnorm = Model_2(activation_fn=nn.GELU, activation_name='GELU', norm_layer_type='batch')
    print(model_gelu_batchnorm)

    model_gelu_layernorm = Model_2(activation_fn=nn.GELU, activation_name='GELU', norm_layer_type='layer')
    print(model_gelu_layernorm)

    # Test forward pass
    dummy_input = torch.randn(64, N_FEATURES)
    output_bn = model_gelu_batchnorm(dummy_input)
    output_ln = model_gelu_layernorm(dummy_input)
    print(f"Output shape Model 2 (GELU, BatchNorm): {output_bn.shape}")
    print(f"Output shape Model 2 (GELU, LayerNorm): {output_ln.shape}")