from src.mae import * 

model = mae_vit_base_patch16()

print(model)

img = torch.rand(4,3,224,224)
model = model.to(0)
img = img.to(0)

loss, pred, mask = model(img)

print(loss)
print(pred.shape)
print(mask.shape)