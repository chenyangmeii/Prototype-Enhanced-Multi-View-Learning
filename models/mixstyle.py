import torch
import torch.nn as nn


class MixStyle(nn.Module):

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix="random"):

        super().__init__()
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        assert mix in ["random", "crossdomain"]
        self.mix = mix
        self.beta = torch.distributions.Beta(self.alpha, self.alpha)

    def forward(self, x):

        if (not self.training) or (self.p <= 0):
            return x
        if torch.rand(1).item() > self.p:
            return x

        B = x.size(0)
        if B <= 1:
            return x


        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        x_normed = (x - mu) / sig


        lam = self.beta.sample((B, 1, 1, 1)).to(x.device)

        # 选择配对方式
        if self.mix == "random":
            perm = torch.randperm(B, device=x.device)
        else:

            perm = torch.arange(B, device=x.device)
            half = B // 2
            perm[:half], perm[half:2*half] = perm[half:2*half], perm[:half]
            if B % 2 == 1:
                perm[-1] = perm[-1]  # 最后一个不动

        mu2, sig2 = mu[perm], sig[perm]


        mu_mix = mu * lam + mu2 * (1 - lam)
        sig_mix = sig * lam + sig2 * (1 - lam)

        return x_normed * sig_mix + mu_mix
