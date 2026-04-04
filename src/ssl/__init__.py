from .simclr import *
from .vicreg import *
from .barlow_twins import *
from .bt_clr import * 
from .vicreg_clr import * 

loss_dict = {
    "simclr": SimCLR,
    "vicreg": VICRegLoss,
    "bt": BarlowTwinLoss,
}

proj_dict = {
    "bt": bt_proj,
    "vicreg": vicreg_proj,
    "simclr": BYOL_mlp
}

pretrain_algo = {
    # "lema": train_lema
    "vicreg_clr": train_vicregclr,
    "bt_clr": train_btclr,
    "bt": train_bt
}