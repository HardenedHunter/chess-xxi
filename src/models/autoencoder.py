from torch import nn


# class Autoencoder(nn.Module):
#     def __init__(self):
#         super(Autoencoder, self).__init__()

#         self.encoder = nn.Sequential(
#             nn.Linear(769, 600),
#             nn.ReLU(),
#             nn.Linear(600, 400),
#             nn.ReLU(),
#             nn.Linear(400, 200),
#             nn.ReLU(),
#             nn.Linear(200, 100),
#         )

#         self.decoder = nn.Sequential(
#             nn.Linear(100, 200),
#             nn.ReLU(),
#             nn.Linear(200, 400),
#             nn.ReLU(),
#             nn.Linear(400, 600),
#             nn.ReLU(),
#             nn.Linear(600, 769),
#             nn.ReLU(),
#         )

#     def encode(self, x):
#         return self.encoder(x)

#     def decode(self, x):
#         return self.decoder(x)

#     def forward(self, x):
#         encoded = self.encode(x)
#         decoded = self.decode(encoded)
#         return decoded

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(769, 600),
            nn.Tanh(),
            nn.Linear(600, 400),
            nn.Tanh(),
            nn.Linear(400, 200),
            nn.Tanh(),
            nn.Linear(200, 100),
            nn.Tanh(),
            nn.Linear(100, 200),
            nn.Tanh(),
            nn.Linear(200, 400),
            nn.Tanh(),
            nn.Linear(400, 600),
            nn.Tanh(),
            nn.Linear(600, 769),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.encoder(x)
