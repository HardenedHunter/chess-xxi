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

        # try one layer?
        self.top_fc1 = nn.Linear(769, 600)
        self.top_fc2 = nn.Linear(600, 400)
        self.top_fc3 = nn.Linear(400, 200)
        self.top_fc4 = nn.Linear(200, 100)

        self.bot_fc1 = nn.Linear(769, 600)
        self.bot_fc2 = nn.Linear(600, 400)
        self.bot_fc3 = nn.Linear(400, 200)
        self.bot_fc4 = nn.Linear(200, 100)

        # self.extractor1 = build_extractor()
        # self.extractor2 = build_extractor()

        self.fc1 = nn.Linear(200, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 2)
        #     nn.ReLU(),
        #     nn.Linear(400, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 2),

        # self.connector = nn.Sequential(
        #     nn.Linear(200, 400),
        #     nn.ReLU(),
        #     nn.Linear(400, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 2),
        #     # nn.Softmax(2)
        # )

    def forward(self, x1, x2):
        f1 = F.relu(self.top_fc1(x1))
        f1 = F.relu(self.top_fc2(f1))
        f1 = F.relu(self.top_fc3(f1))
        f1 = F.relu(self.top_fc4(f1))

        f2 = F.relu(self.bot_fc1(x2))
        f2 = F.relu(self.bot_fc2(f2))
        f2 = F.relu(self.bot_fc3(f2))
        f2 = F.relu(self.bot_fc4(f2))
        # f1 = self.extractor1(x1)
        # f2 = self.extractor1(x2)

        f = torch.cat([f1, f2], dim=1)
        f = f.view(f.size(0), -1)
        f = F.relu(self.fc1(f))
        f = F.relu(self.fc2(f))
        f = F.relu(self.fc3(f))
        f = self.fc4(f)

        return f

        # return self.connector(torch.cat([f1, f2], dim=-1))
