from model import Net
import torch
from torchviz import make_dot

model = Net(num_classes=2)
x = torch.randn(1, 3, 224, 224)  # Assuming the input size is 3x224x224
y = model(x)

# Generate a detailed visualization similar to the provided CNN architecture image
selected_layers = [
    "encoder.layers.encoder_layer_10", 
    "encoder.layers.encoder_layer_11", 
    "heads.head.weight", 
    "heads.head.bias"
]
dot = make_dot(y, params=dict((name, param) for name, param in model.named_parameters() if any(layer in name for layer in selected_layers)), show_attrs=True, show_saved=True)
dot.format = 'png'
dot.render('detailed_model_architecture', view=False)
print("Detailed model architecture diagram saved as 'detailed_model_architecture.png'")
