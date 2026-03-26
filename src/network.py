import math
import torch
import torchvision 
import torch.nn as nn 
import torch.nn.functional as F
from torchdiffeq import odeint
from .ssl import proj_dict, pred_dict
import warnings; warnings.filterwarnings("ignore")

class MLP(nn.Module): # MLP for linear protocol
    def __init__(self, in_features, num_classes, mlp_type="linear"):
        super().__init__()
        if mlp_type == "linear":
            print("===> using linear mlp")
            self.mlp = nn.Linear(in_features, num_classes)
            # standard initialization trick 
            self.mlp.weight.data.normal_(mean=0.0, std=0.01)
            self.mlp.bias.data.zero_()
        else:
            print("===> using hiddin mlp")
            self.mlp = nn.Sequential(
                nn.Linear(in_features, in_features),
                nn.ReLU(),
                nn.Linear(in_features, num_classes)
            )

    def forward(self, x):
        return self.mlp(x)
    
# Energy network regularization
class EnergyScoreNet(nn.Module):
    def __init__(self, z_dim, eta = 1e-4, steps = 30, sigma = 1e-3, delta = 0.1, net_type = "score"):
        """
        net_type: score / energy
        """
        super().__init__()
        hidden = z_dim * 2
        self.snet = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Linear(z_dim, hidden)),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.1),
            nn.utils.parametrizations.spectral_norm(nn.Linear(hidden, hidden)),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.1)
        )

        self.net_type = net_type 
        self.sigma = sigma 
        self.delta = delta 

        if self.net_type == "score":
            self.snet.append(nn.utils.parametrizations.spectral_norm(nn.Linear(hidden, z_dim)))
        else:
            self.snet.append(nn.utils.parametrizations.spectral_norm(nn.Linear(hidden, 1)))

        # parameters for langevin sampling 
        self.eta = eta
        self.steps = steps 

    def forward(self, z):
        return self.snet(z)

    def langevin_sampling(self, z = None):
        self.eval()
        if z is None:
            z = torch.randn_like(z)
        else:
            z = z.clone().detach()
        z.requires_grad_(True)
        for _ in range(self.steps):
            if self.net_type == "score":
                grad = self(z)
            else:
                e = self(z).squeeze().sum()
                grad = -torch.autograd.grad(e, z, create_graph=False)[0] # -\nabla_{z} E(z)
            z = z + self.eta * torch.clamp(grad, -self.delta, self.delta) + math.sqrt(2 * self.eta) * torch.randn_like(z) # langevin dynamics
            z = z.detach()
            z.requires_grad_(True)
        self.train()
        return z.detach()
    
    def dsm_loss(self, z):
        epsilon = torch.randn_like(z)
        z_hat = z + self.sigma * epsilon
        if self.net_type == "score":
            s = self(z_hat)
            loss = 0.5 * (self.sigma * s + epsilon).pow(2).sum(dim = -1).mean()
        elif self.net_type == "energy":
            z_hat.requires_grad_(True)
            e = self(z_hat).sum()
            s = torch.autograd.grad(e, z_hat, create_graph=True)[0]
            loss = 0.5 * (self.sigma * s - epsilon).pow(2).sum(dim = -1).mean()
        return loss 

# proj_dim = 128, ode_steps = 10, algo_type="nodel", carl_hidden = 4096, byol_hidden=4096, pred_dim = 512, barlow_hidden = 8192, vae_out = 256

class BaseEncoder(nn.Module):
    def __init__(self, model_name = "resnet18", pretrained = False):
        super().__init__()
        if model_name == 'resnet50':
            model = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None)
        elif model_name  == 'resnet18':
            model = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None)
        else:
            print(f"{model_name} model type not supported")
            model = None

        # for smaller image size
        module_keys = list(model._modules.keys())
        self.feat_extractor = nn.Sequential()
        for key in module_keys[:-1]:
            if key == "maxpool": # don't add maxpool layer
                continue
            module_key = model._modules.get(key, nn.Identity())
            self.feat_extractor.add_module(key, module_key)

        if not pretrained:
            in_feat = self.feat_extractor.conv1.in_channels
            out_feat = self.feat_extractor.conv1.out_channels
            self.feat_extractor.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False)

        self.classifier_infeatures = model._modules.get(module_keys[-1], nn.Identity()).in_features

    def forward(self, x):
        return self.feat_extractor(x).flatten(1)

class Network(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.base_encoder = BaseEncoder(model_name = kwargs.get("model_name", "resnet18"), pretrained = kwargs.get("pretrained", False))
        self.ci = self.base_encoder.classifier_infeatures
        self.algo_type = kwargs.get("algo_type", "-1")
        assert self.algo_type != "-1", "Please specify algo_type for the network"

        # so far general feature extractor ($h_{\theta}$)
        proj_dim = kwargs.get("proj_dim", 128)
        self.proj_args = {
            "simclr": (self.ci, self.ci, proj_dim),
            "scalre": (self.ci, self.ci, proj_dim),
            "byol": (self.ci, kwargs.get("byol_hidden", 4096), proj_dim),
            "simsiam": (self.ci,),
            "bt": (self.ci, kwargs.get("barlow_hidden", 8192), proj_dim),
            "vicreg": (self.ci, kwargs.get("barlow_hidden", 8192), proj_dim),
            "simsiam-sc": (self.ci,),
            "bt-sc": (self.ci, kwargs.get("barlow_hidden", 8192), proj_dim),
            "vicreg-sc": (self.ci, kwargs.get("barlow_hidden", 8192), proj_dim),
            "byol-sc": (self.ci, kwargs.get("byol_hidden", 4096), proj_dim)
        }

        self.pred_args = {"byol": (kwargs.get("pred_dim", 256), kwargs.get("byol_hidden", 4096), proj_dim),
                          "simsiam": (self.ci, kwargs.get("pred_dim", 512)),
                          "byol-sc": (kwargs.get("pred_dim", 256), kwargs.get("byol_hidden", 4096), proj_dim),
                          "simsiam-sc": (self.ci, kwargs.get("pred_dim", 512))}

        self.proj = proj_dict[self.algo_type](*self.proj_args[self.algo_type])
        self.pred = pred_dict.get(self.algo_type, None)
        if self.pred is not None:
            self.pred = self.pred(*self.pred_args[self.algo_type])

    def forward(self, x, t = None, test=None):
        features = self.base_encoder(x)
        if test:
            return {"features": features}
        
        proj = self.proj(features)
        if self.pred is not None:
            pred = self.pred(proj)
            return {"features": features, "proj_features": proj, "pred_features": pred}
        else:
            return {"features": features, "proj_features": proj}


if __name__ == "__main__":
    device=torch.device('cuda:0')
    byol_params = {
        "model_name": 'resnet18',
        "pretrained": False,
        "proj_dim": 256,
        "algo_type": "byol",
        "byol_hidden": 4096,
        "pred_dim": 256
    }

    byol_net = Network(**byol_params).to(device)

    print(byol_net.base_encoder)
    print(byol_net.proj)