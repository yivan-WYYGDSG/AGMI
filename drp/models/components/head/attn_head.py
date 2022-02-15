from drp.models.registry import COMPONENTS
import torch.nn as nn
import torch
import torch.nn.functional as F


@COMPONENTS.register_module()
class GatedAttention(nn.Module):
    def __init__(self, L, D, K):
        super(GatedAttention, self).__init__()
        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.squeeze(0)
        # H = self.feature_extractor_encoder(x)
        H = x
        H = H.transpose(1, 2) # 18498*3
        # H = self.feature_extractor_linear(H)  # NxL

        A_V = self.attention_V(H)  # NxD 18498*64
        A_U = self.attention_U(H)  # NxD 18498*64
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK 18498*1
        A = torch.transpose(A, 2, 1)  # KxN 1*18498
        A = F.softmax(A, dim=2)  # softmax over N
        M = torch.mul(x, A)
        # print(f'results shape:{M.shape}')

        return M, A

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
