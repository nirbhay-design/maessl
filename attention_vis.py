import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from src.mae import MAEEncoder
from train_utils import yaml_loader
from functools import partial
import argparse, os

def get_args():
    parser = argparse.ArgumentParser(description="Training script for linear probing")

    # basic experiment settings
    parser.add_argument("--saved_path", type=str, default="model.pth", required=True, help="path for pretrained model")
    parser.add_argument("--image", type=str, default="test_image.jpg", required=True, help="image to visualize attention maps")
    parser.add_argument("--gpu", type=int, default = 0, help="gpu_id")
    args = parser.parse_args()
    return args

attention_weights = []

# We create a closure so the hook knows the specific dimensions of the timm attention block
def get_qkv_hook(attn_module):
    def hook(module, input, output):
        # input[0] shape: [Batch, Tokens, Dim]
        # output shape: [Batch, Tokens, 3 * Dim]
        B, N, _ = input[0].shape
        
        # 1. Reshape the output exactly how timm does it
        qkv = output.reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        
        # 2. Account for newer timm versions that normalize q and k
        if hasattr(attn_module, 'q_norm'):
            q = attn_module.q_norm(q)
            k = attn_module.k_norm(k)
        
        # 3. Recreate the attention math manually
        q = q * attn_module.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        
        # 4. Save the final probabilities
        attention_weights.append(attn.detach().cpu())
        
    return hook

if __name__ == "__main__":
    args = get_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # --- 2. Setup Model and Attach Hook ---
    config = yaml_loader("configs/test.yaml")
    model_params = config["mae_model_params"]
    model = MAEEncoder(img_size=model_params["img_size"], patch_size=model_params["patch_size"], in_chans=model_params["in_chans"],
                embed_dim=model_params["embed_dim"], depth=model_params["depth"], num_heads=model_params["num_heads"],
                mlp_ratio=model_params["mlp_ratio"], norm_layer=partial(nn.LayerNorm, eps=1e-6))
    print(model.load_state_dict(torch.load(args.saved_path, map_location=device)))
    model = model.to(device)
    model.eval()

    # print(model)

    target_attn = model.blocks[-1].attn
    hook_handle = target_attn.qkv.register_forward_hook(get_qkv_hook(target_attn))

    # to normalize image 
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    print(f"using image: {args.image}")
    img_path = args.image
    img_pil = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    img_tensor = transform(img_pil).unsqueeze(0) # [1, 3, 224, 224]

    attention_weights.clear() # Clear any previous weights
    with torch.no_grad():
        _ = model(img_tensor.to(device)) # default mask ratio is 0.0

    # Get the weights from our hook list
    attn = attention_weights[0] # Shape: [1, num_heads, num_patches+1, num_patches+1]
    # print(attn.shape)
    cls_attention = attn[0, :, 0, 1:] # Shape: [num_heads, num_patches]
    cls_attention_head = torch.mean(cls_attention, dim=0) # Shape: [num_patches]
    # head_idx = 4
    # cls_attention_head = cls_attention[head_idx] # Shape: [num_patches]

    grid_size = int(np.sqrt(cls_attention_head.shape[0]))
    attention_grid = cls_attention_head.reshape(grid_size, grid_size).numpy()

    attention_grid = (attention_grid - attention_grid.min()) / (attention_grid.max() - attention_grid.min())

    threshold = 0.3
    attention_grid[attention_grid < threshold] = 0.0

    attention_img = Image.fromarray(attention_grid)
    attention_resized_img = attention_img.resize((224, 224), resample=Image.Resampling.BICUBIC)
    attention_resized = np.array(attention_resized_img)

    # Prepare original image for plotting (undo normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_show = img_tensor[0].permute(1, 2, 0).numpy()
    img_show = std * img_show + mean
    img_show = np.clip(img_show, 0, 1)

    model_name = ".".join(args.saved_path.split("/")[-1].split(".")[:-1])
    save_path = os.path.join("attention_maps", args.image.split("/")[-1].split(".")[0] + "." + model_name + '.png')
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(img_show)
    # ax1.set_title("Original Image")
    ax1.axis('off')

    cmap = "inferno"

    ax2.imshow(attention_resized, cmap=cmap)
    # ax2.set_title("Attention Map")
    ax2.axis('off')

    # Overlay
    ax3.imshow(img_show)
    ax3.imshow(attention_resized, cmap=cmap, alpha=0.5) # Alpha controls transparency
    # ax3.set_title("Overlay")
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    # Don't forget to remove the hook when you are done!
    hook_handle.remove()