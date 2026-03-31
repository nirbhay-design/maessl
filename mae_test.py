from src.mae import * 

model = mae_vit_base_patch16()

print(model)

img = torch.rand(4,3,224,224)
model = model.to(0)
img = img.to(0)

loss, pred, mask, latent = model(img, mask_ratio=0.8)

print(loss)
print(pred.shape)
print(mask.shape)
print(mask.sum(dim = -1))
print(latent.shape)