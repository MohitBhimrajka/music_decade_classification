# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Constants
N_FEATURES = 90
N_CLASSES = 10 # 10 decades (0-9)

class BaseModel(nn.Module):
    """A base class for easier model summary printing (optional)."""
    def __init__(self):
        super().__init__()

    def __str__(self):
        """Prints model architecture."""
        model_str = f"Model Architecture: {self.__class__.__name__}\n"
        total_params = 0
        for name, layer in self.named_children():
            model_str += f"  ({name}): {layer}\n"
            for param in layer.parameters():
                total_params += param.numel()
        model_str += f"Total Trainable Parameters: {total_params:,}\n"
        return model_str

# --- Architecture 1: Moderate, Consistent ---
class Model_1(BaseModel):
    def __init__(self, input_size=N_FEATURES, num_classes=N_CLASSES):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, 128)
        self.relu_1 = nn.ReLU()
        self.layer_2 = nn.Linear(128, 128)
        self.relu_2 = nn.ReLU()
        self.output_layer = nn.Linear(128, num_classes)
        # Note: Softmax is typically applied *implicitly* by the loss function (CrossEntropyLoss)
        # So we don't need a nn.Softmax() layer here.

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu_1(x)
        x = self.layer_2(x)
        x = self.relu_2(x)
        x = self.output_layer(x)
        return x

# --- Architecture 2: Wider Layers ---
class Model_2(BaseModel):
    def __init__(self, input_size=N_FEATURES, num_classes=N_CLASSES):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, 256)
        self.relu_1 = nn.ReLU()
        self.layer_2 = nn.Linear(256, 256)
        self.relu_2 = nn.ReLU()
        self.output_layer = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu_1(x)
        x = self.layer_2(x)
        x = self.relu_2(x)
        x = self.output_layer(x)
        return x

# --- Architecture 3: Bottleneck/Mixed ---
class Model_3(BaseModel):
    def __init__(self, input_size=N_FEATURES, num_classes=N_CLASSES):
        super().__init__()
        self.layer_1 = nn.Linear(input_size, 256)
        self.relu_1 = nn.ReLU()
        self.layer_2 = nn.Linear(256, 128)
        self.relu_2 = nn.ReLU()
        self.layer_3 = nn.Linear(128, 64)
        self.relu_3 = nn.ReLU()
        self.output_layer = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu_1(x)
        x = self.layer_2(x)
        x = self.relu_2(x)
        x = self.layer_3(x)
        x = self.relu_3(x)
        x = self.output_layer(x)
        return x

if __name__ == '__main__':
    # Example of instantiating and printing models
    model1 = Model_1()
    print(model1)

    model2 = Model_2()
    print(model2)

    model3 = Model_3()
    print(model3)

    # Test forward pass with dummy data
    dummy_input = torch.randn(64, N_FEATURES) # Batch size 64
    output1 = model1(dummy_input)
    output2 = model2(dummy_input)
    output3 = model3(dummy_input)
    print(f"Output shape Model 1: {output1.shape}") # Should be [64, 10]
    print(f"Output shape Model 2: {output2.shape}") # Should be [64, 10]
    print(f"Output shape Model 3: {output3.shape}") # Should be [64, 10]