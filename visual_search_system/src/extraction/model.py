import warnings
import torch
import click
import re

from numpy import dot
from numpy.linalg import norm
from PIL import Image

import torch
from torch import nn
from torchvision import models

warnings.simplefilter("ignore")
torchDevice = torch.device(
    'mps') if torch.backends.mps.is_available() else torch.device('cpu')


class Identity(nn.Module):
    """
    Identity layer.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return x


class Dense(nn.Module):
    """
    Fully-connected layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        use_batchnorm=False,
    ):
        super().__init__()

        self.use_bias = use_bias
        self.use_batchnorm = use_batchnorm

        self.dense = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=self.use_bias and not self.use_batchnorm,
        )

        if self.use_batchnorm:
            self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        if self.use_batchnorm:
            x = self.bn(x)
        return x


class ProjectionHead(nn.Module):
    """
    Projection head;
    converts extracted features to the embedding space.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        head_type="nonlinear",
    ):
        """
        Initial;
        :param in_features: number of input feature;
        :param hidden_features: number of hidden features;
        :param out_features: number of output features;
        :param head_type: linear -- one dense layer,
        non-linear -- two dense layers with ReLU activation function;
        """
        super().__init__()

        if head_type == "linear":
            self.layers = Dense(in_features, out_features, False, True)
        elif head_type == "nonlinear":
            self.layers = nn.Sequential(
                Dense(in_features, hidden_features, True, True),
                nn.ReLU(),
                Dense(hidden_features, out_features, False, True),
            )
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x


class SimCLR(nn.Module):
    """
    Contrastive model.
    """

    def __init__(self, architecture):
        super().__init__()

        # Configure base model
        if architecture == "resnet_018":
            self.encoder = models.resnet18(pretrained=True)
        elif architecture == "resnet_034":
            self.encoder = models.resnet34(pretrained=True)
        self.encoder.maxpool = Identity()
        self.encoder.fc = Identity()

        # Unfreeze parameters
        for p in self.encoder.parameters():
            p.requires_grad = True

        self.projector = ProjectionHead(512, 256, 256)
        self.encoder.to(torchDevice)
        self.projector.to(torchDevice)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torchDevice)
        out = self.encoder(x)
        out = self.projector(out)
        return out


class Model:

    def __init__(self, model_path):

        architecture = re.search("resnet_[0-9]*", model_path).group(0)
        self.model = SimCLR(architecture=architecture)

        self.model.load_state_dict(torch.load(
            model_path, map_location=torchDevice))

        self.model.eval()

    def extract(self, image):
        return self.model(image.unsqueeze(0))[0].detach().cpu().numpy()


if __name__ == "__main__":
    from preprocess import Preprocessing
    print(torchDevice)

    image_size = 224
    architecture = 'resnet_034'

    preprocessing = Preprocessing(image_size=image_size)
    image_path = "simclr_resources/probe/Aaron_Sorkin/Aaron_Sorkin_0002.jpg"
    probe = Image.open(image_path)
    probe = preprocessing.process(probe)

    model = Model(
        f"simclr_resources/model_size_{image_size:03}_{architecture}.pth")

    print(model.extract(probe).shape)
