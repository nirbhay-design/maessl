from .simclr import *
from .vicreg import *
from .barlow_twins import *

loss_dict = {
    "simclr": SimCLR,
    "vicreg": VICRegLoss,
    "bt": BarlowTwinLoss,
    "scalre": SimCLR,
    "vicreg-sc": VICRegLoss,
    "bt-sc": BarlowTwinLoss,
    "lema": SimCLR
}

proj_dict = {
    "bt": bt_proj,
    "vicreg": vicreg_proj,
    "bt-sc": bt_proj,
    "vicreg-sc": vicreg_proj,
}

pretrain_algo = {
    "simclr": train_simclr,
    "vicreg": train_vicreg,
    "bt": train_bt,
    "scalre": train_scalre,
    "vicreg-sc": train_vicreg_sc,
    "bt-sc": train_bt_sc,
    # "lema": train_lema
}