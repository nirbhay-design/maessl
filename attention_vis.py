import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils

from src.mae import MAEEncoder
from train_utils import yaml_loader, get_vit_config_test
from functools import partial

def get_args():
    parser = argparse.ArgumentParser(description="Training script for linear probing")
    parser.add_argument("--saved_path", type=str, default="model.pth", required=True, help="path for pretrained model")
    parser.add_argument("--model", type=str, default="vit", help="vit/vit_t_8/vit_s16")
    parser.add_argument("--image", type=str, default="test_image.jpg", required=True, help="image to visualize attention maps")
    parser.add_argument("--gpu", type=int, default=0, help="gpu_id")
    parser.add_argument("--output_dir", type=str, default="attention_maps", help="directory to save outputs")
    parser.add_argument("--threshold", type=float, default=0.6, help="Keep xx% of the mass for the DINO masks.")
    args = parser.parse_args()
    return args

attention_weights = []

def get_qkv_hook(attn_module):
    def hook(module, input, output):
        B, N, _ = input[0].shape
        qkv = output.reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        
        if hasattr(attn_module, 'q_norm'):
            q = attn_module.q_norm(q)
            k = attn_module.k_norm(k)
        
        q = q * attn_module.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attention_weights.append(attn.detach().cpu())
        
    return hook

if __name__ == "__main__":
    args = get_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Setup Directory
    model_name = ".".join(os.path.basename(args.saved_path).split(".")[:-1])
    img_name = os.path.basename(args.image).split(".")[0]
    save_dir = os.path.join(args.output_dir, f"{img_name}_{model_name}")
    print(f"saving to: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Setup Model
    config = yaml_loader("configs/test.yaml")
    model_params = config["mae_model_params"]
    model_params = get_vit_config_test(args.model, model_params, "img100")
    patch_size = model_params["patch_size"]
    model = MAEEncoder(img_size=model_params["img_size"], patch_size=patch_size, in_chans=model_params["in_chans"],
                embed_dim=model_params["embed_dim"], depth=model_params["depth"], num_heads=model_params["num_heads"],
                mlp_ratio=model_params["mlp_ratio"], norm_layer=partial(nn.LayerNorm, eps=1e-6))
    
    print(model.load_state_dict(torch.load(args.saved_path, map_location=device)))
    model = model.to(device)
    model.eval()

    # Attach hook
    target_attn = model.blocks[-1].attn
    hook_handle = target_attn.qkv.register_forward_hook(get_qkv_hook(target_attn))

    # 2. Process Image
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    print(f"Using image: {args.image}")
    img_pil = Image.open(args.image).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    img_tensor = transform(img_pil).unsqueeze(0) # [1, 3, 224, 224]

    # 3. Forward Pass
    attention_weights.clear()
    with torch.no_grad():
        _ = model(img_tensor.to(device)) 

    hook_handle.remove()

    # 4. Extract Attention Maps
    attn = attention_weights[0] # Shape: [1, num_heads, num_patches+1, num_patches+1]
    cls_attention = attn[0, :, 0, 1:] # Shape: [num_heads, num_patches]
    
    nh = cls_attention.shape[0]
    w_featmap = h_featmap = int(np.sqrt(cls_attention.shape[1]))

    # 5. DINO Mass Thresholding Logic
    if args.threshold is not None:
        val, idx = torch.sort(cls_attention)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        
        # Interpolate threshold masks
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    # Interpolate standard heatmaps
    attentions = cls_attention.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    # 6. Prepare Original Image for Output
    mean_np = np.array([0.485, 0.456, 0.406])
    std_np = np.array([0.229, 0.224, 0.225])
    img_show = img_tensor[0].permute(1, 2, 0).numpy()
    img_show = std_np * img_show + mean_np
    img_show = np.clip(img_show, 0, 1)
    
    # Save original image
    # vutils.save_image(vutils.make_grid(img_tensor, normalize=True, scale_each=True), os.path.join(save_dir, "img.png"))
    
    # Convert img_show to uint8 for DINO's polygon plotting
    img_show_uint8 = (img_show * 255).astype(np.uint8)

    # 7. Generate Outputs (Per Head)
    for j in range(nh):
        # Save standard Heatmap
        # fname_heatmap = os.path.join(save_dir, f"attn-head{j}.png")
        # plt.imsave(fname=fname_heatmap, arr=attentions[j], cmap='inferno')
        # print(f"{fname_heatmap} saved.")
        head_to_plot = j  # You can change this to visualize different heads (e.g., 0 through 15)
    
        fig, axes = plt.subplots(1, 2, figsize=(4, 4))

        # Plot 1: Original Image
        axes[0].imshow(img_show)
        # axes[0].set_title("Original Image", fontsize=14)
        axes[0].axis('off')

        # Plot 2: Attention Heatmap (for the selected head)
        axes[1].imshow(attentions[head_to_plot], cmap='inferno')
        # axes[1].set_title(f"Attention Map (Head {head_to_plot})", fontsize=14)
        axes[1].axis('off')

        # Plot 3: Overlay (Original + Heatmap)
        # axes[2].imshow(img_show)
        # axes[2].imshow(attentions[head_to_plot], cmap='inferno', alpha=0.5) # Alpha controls transparency
        # # axes[2].set_title(f"Overlay (Head {head_to_plot})", fontsize=14)
        # axes[2].axis('off')

        plt.tight_layout()
        
        # Save and display the combined plot
        combined_save_path = os.path.join(save_dir, f"combined_plot_head{head_to_plot}.png")
        plt.savefig(combined_save_path, bbox_inches='tight', dpi=150)
        print(f"Combined side-by-side plot saved to: {combined_save_path}")
        
        plt.show()

    print(f"All outputs successfully saved in: {save_dir}")


    
