from pykeops.torch import Vi, Vj
import torch
from dataclasses import dataclass

@dataclass
class KNN_KeOps:
    K: int
    metric: str = "euclidean"

    def fit(self, x_train: torch.Tensor):
        # Setup the K-NN estimator:
        # Encoding as KeOps LazyTensors:
        D = x_train.shape[1]
        X_i = Vi(0, D)  # Purely symbolic "i" variable, without any data array
        X_j = Vj(1, D)  # Purely symbolic "j" variable, without any data array

        # Symbolic distance matrix:
        if self.metric == "euclidean":
            D_ij = ((X_i - X_j) ** 2).sum(-1)
        elif self.metric == "manhattan":
            D_ij = (X_i - X_j).abs().sum(-1)
        elif self.metric == "angular":
            D_ij = -(X_i | X_j)
        elif self.metric == "hyperbolic":
            D_ij = ((X_i - X_j) ** 2).sum(-1) / (X_i[0] * X_j[0])
        else:
            raise NotImplementedError(f"The '{self.metric}' distance is not supported.")

        # K-NN query operator:
        self.KNN_fun = D_ij.argKmin(self.K, dim=1)
        self.KNN_dist = D_ij.Kmin(self.K, dim=1)
        
        self.x_train = x_train

    def __call__(self,  x_test: torch.Tensor):
        """
            TODO: verify if for each query we should return the point or only its neighbors...
        """
        # Actual K-NN query:
        indices = self.KNN_fun(x_test, self.x_train)
        dists = self.KNN_dist(x_test, self.x_train)
        return indices, dists
