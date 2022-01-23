import torch
from torch import nn
import torch.nn.functional as F


def build_extractor():
    return nn.Sequential(
        nn.Linear(769, 600),
        nn.ReLU(),
        nn.Linear(600, 400),
        nn.ReLU(),
        nn.Linear(400, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
        nn.ReLU(),
    )


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()

        self.extractor1 = build_extractor()
        self.extractor2 = build_extractor()

        self.connector = nn.Sequential(
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
        )

    def forward(self, x1, x2):
        f1 = self.extractor1(x1)
        f2 = self.extractor1(x2)

        f = torch.cat([f1, f2], dim=1)
        f = f.view(f.size(0), -1)
        return self.connector(f)
