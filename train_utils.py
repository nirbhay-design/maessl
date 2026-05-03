from src.ssl import loss_dict
from src.network import Network, MLP
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torch.optim as optim 
import yaml, sys, random, numpy as np
from yaml.loader import SafeLoader
from src.lars import LARS
import math
import copy, os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score,\
                            adjusted_rand_score, \
                            normalized_mutual_info_score, \
                            davies_bouldin_score
import umap 
import torch.distributed as dist 

def yaml_loader(yaml_file):
    with open(yaml_file,'r') as f:
        config_data = yaml.load(f,Loader=SafeLoader)
    
    return config_data

def progress(current, total, **kwargs):
    progress_percent = (current * 50 / total)
    progress_percent_int = int(progress_percent)
    data_ = ""
    for meter, data in kwargs.items():
        data_ += f"{meter}: {round(data,2)}|"
    print(f" |{chr(9608)* progress_percent_int}{' '*(50-progress_percent_int)}|{current}/{total}|{data_}",end='\r')
    if (current == total):
        print()

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"

def save_model(model, epochs, path):
    # path format: dir/model.pth
    cur_path = os.path.join(path.split('/')[0], '.'.join(path.split('/')[-1].split(".")[:-1]) + f".ec{epochs}.pth") 
    final_model = model.module if dist.is_initialized() else model 
    torch.save(final_model.base_encoder.state_dict(), cur_path)
    print(f"Model saved at: {cur_path}")

def get_features_labels(model, loader, device, return_logs = False):
    model = model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        loader_len = len(loader)
        for idx,(x,y) in enumerate(loader):
            x = x.to(device)
            # y = y.to(device)

            feats = model(x)

            all_features.append(feats.detach().cpu())
            all_labels.append(y)

            if return_logs:
                progress(idx+1,loader_len)

    features = F.normalize(torch.vstack(all_features), dim = -1).detach().cpu().numpy()
    labels = torch.hstack(all_labels).detach().cpu().numpy()

    return {"features": features, "labels": labels}

def make_tsne_for_dataset(model, loader, device, return_logs = False, tsne_name = None):
    
    output = get_features_labels(model, loader, device, return_logs)
    features = output["features"]
    labels = output["labels"]
    make_tsne_plot(features, labels, name = tsne_name)

def get_tsne_knn_logreg(model, train_loader, test_loader, device, return_logs = False, umap = True, tsne = True, knn = True, log_reg = True, cmet=True, tsne_name = None):
    train_output = get_features_labels(model, train_loader, device, return_logs)
    test_output = get_features_labels(model, test_loader, device, return_logs)
    
    x_train, y_train = train_output["features"], train_output["labels"]
    x_test, y_test = test_output["features"], test_output["labels"]

    outputs = {}

    if tsne:
        print("TSNE on Train set")
        make_tsne_plot(x_train, y_train, name = f"tsne.{tsne_name}")

    if umap:
        print("UMAP on Train set")
        make_umap_plot(x_train, y_train, name = f"umap.{tsne_name}")

    if knn:
        print("knn evalution")
        knnc = KNeighborsClassifier(n_neighbors=200)
        knnc.fit(x_train, y_train)
        y_test_pred = knnc.predict(x_test)
        knn_acc = accuracy_score(y_test, y_test_pred)
        outputs["knn_acc"] = knn_acc

    if log_reg:
        print("logistic regression evalution")
        lreg = LogisticRegression(random_state=42) # Example hyperparameters
        lreg.fit(x_train, y_train)
        # Make predictions
        y_test_pred = lreg.predict(x_test)
        lreg_acc = accuracy_score(y_test, y_test_pred)
        outputs["lreg_acc"] = lreg_acc

    if cmet:
        N_CLASSES = len(np.unique(y_test))
        print("clustering metrics evalution")
        kmeans = KMeans(n_clusters=N_CLASSES, random_state=42)
        cluster_labels = kmeans.fit_predict(x_train)
        ari = adjusted_rand_score(y_train, cluster_labels)
        nmi = normalized_mutual_info_score(y_train, cluster_labels)
        ss = silhouette_score(x_train, cluster_labels)
        dbs = davies_bouldin_score(x_train, cluster_labels)
        outputs["ari"] = ari
        outputs["nmi"] = nmi
        outputs["ss"] = ss
        outputs["dbs"] = dbs 

    return outputs 

def loss_function(loss_type = 'scalre', **kwargs):
    print(f"loss function: {loss_type}")
    return loss_dict[loss_type](**kwargs)
    
def model_optimizer(model, opt_name, model2 = None, **opt_params):
    print(f"using optimizer: {opt_name}")

    if model2 is None:
        params = model.parameters()
    else:
        params = list(model.parameters()) + list(model2.parameters())

    if opt_name == "SGD":
        return optim.SGD(params, **opt_params, fused = True)
    elif opt_name == "ADAM":
        return optim.Adam(params, **opt_params, fused = True)
    elif opt_name == "AdamW":
        return optim.AdamW(params, **opt_params, fused = True)
    elif opt_name == "LARS":
        return LARS(params, **opt_params)
    else:
        print("{opt_name} not available")
        return None

def load_dataset(dataset_name, **kwargs):
    aug = kwargs.get("aug", "v1")
    if aug == "v1":
        import src.data as data_module
    else:
        import src.data_v2 as data_module
    
    if dataset_name == "cifar10":
        return data_module.Cifar10DataLoader(**kwargs)
    if dataset_name == 'cifar100':
        return data_module.Cifar100DataLoader(**kwargs)
    if dataset_name == "timg":
        return data_module.tinyimagenet_dataloader(**kwargs)
    if dataset_name == "img100":
        return data_module.imagenet100_dataloader(**kwargs)
    else:
        print(f"{dataset_name} is not supported")
        return None

def make_tsne_plot(X, y, name):
    print(f"TSNE name: {name}")
    if len(np.unique(y)) > 10:

        mask = np.isin(y, range(10)) 
        X = X[mask]
        y = y[mask]

    # tsne = umap.UMAP()
    tsne = TSNE(n_components=2, random_state=0)
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s = 16, alpha = 0.8, cmap='turbo', edgecolors="white", linewidths=0.3)  # Color by labels
    plt.box(False)
    plt.xticks([]) 
    plt.yticks([])
    plt.title("t-SNE")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.colorbar(label="Labels")

    plt.savefig(f"plots/{name}")
    plt.close()

def make_umap_plot(X, y, name):
    tsne = umap.UMAP()
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s = 4, alpha = 0.8, cmap='turbo', edgecolors="white", linewidths=0.3)  # Color by labels
    plt.box(False)
    plt.xticks([]) 
    plt.yticks([])
    plt.title("t-SNE")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.colorbar(label="Labels")
    plt.savefig(f"plots/{name}")
    plt.close()