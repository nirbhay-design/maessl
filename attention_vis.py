import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# --- 1. Define the Hook ---
attention_weights = []

def get_attention_hook(module, input, output):
    # output shape depends on the exact Attention implementation.
    # Usually it's: [batch_size, num_heads, num_tokens, num_tokens]
    # We detach it and move it to CPU.
    attention_weights.append(output.detach().cpu())

# --- 2. Setup Model and Attach Hook ---
# Assuming `model` is your instantiated and loaded MAEEncoder
# model = MAEEncoder(...)
# model.load_state_dict(torch.load('your_checkpoint.pth'))
model.eval()

# Attach the hook to the LAST transformer block's attention dropout layer.
# Note: Check your specific `Block` implementation. It is usually `attn.attn_drop`.
# If it's different, you just need to point it to the layer right after the softmax.
hook_handle = model.blocks[-1].attn.attn_drop.register_forward_hook(get_attention_hook)

# --- 3. Prepare the Image ---
# Load an image and apply standard ViT transforms
img_path = "your_test_image.jpg"
img_pil = Image.open(img_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = transform(img_pil).unsqueeze(0) # [1, 3, 224, 224]

# --- 4. Forward Pass ---
attention_weights.clear() # Clear any previous weights
with torch.no_grad():
    # Make sure mask_ratio=0.0 so we see attention over the whole image!
    _ = model(img_tensor.to(next(model.parameters()).device), mask_ratio=0.0)

# --- 5. Process the Attention Map ---
# Get the weights from our hook list
attn = attention_weights[0] # Shape: [1, num_heads, num_patches+1, num_patches+1]

# We want the attention from the cls_token (index 0) to all other spatial patches (index 1 to end)
# We take the first image in batch [0], all heads [:], cls_token query [0], all patch keys [1:]
cls_attention = attn[0, :, 0, 1:] # Shape: [num_heads, num_patches]

# Average the attention across all attention heads
cls_attention = torch.mean(cls_attention, dim=0) # Shape: [num_patches]

# Reshape the 1D patch list back into a 2D grid
# For a 224x224 image and 16x16 patch size, num_patches is 196. sqrt(196) = 14.
grid_size = int(np.sqrt(cls_attention.shape[0]))
attention_grid = cls_attention.reshape(grid_size, grid_size).numpy()

# Normalize the attention map for visualization (min-max scaling to [0, 1])
attention_grid = (attention_grid - attention_grid.min()) / (attention_grid.max() - attention_grid.min())

# --- 6. Visualize and Overlay ---
# Resize the 14x14 attention map back to 224x224
attention_resized = cv2.resize(attention_grid, (224, 224), interpolation=cv2.INTER_CUBIC)

# Prepare original image for plotting (undo normalization)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_show = img_tensor[0].permute(1, 2, 0).numpy()
img_show = std * img_show + mean
img_show = np.clip(img_show, 0, 1)

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.imshow(img_show)
ax1.set_title("Original Image")
ax1.axis('off')

ax2.imshow(attention_resized, cmap='jet')
ax2.set_title("Attention Map")
ax2.axis('off')

# Overlay
ax3.imshow(img_show)
ax3.imshow(attention_resized, cmap='jet', alpha=0.5) # Alpha controls transparency
ax3.set_title("Overlay")
ax3.axis('off')

plt.tight_layout()
plt.show()

# Don't forget to remove the hook when you are done!
hook_handle.remove()