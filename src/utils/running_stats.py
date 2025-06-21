import math


class OnlineStatsCalculator:
    def __init__(self):
        self.n = 0  # Number of data points
        self.mean = 0.0  # Mean
        self.sqmean = 0.0  # Variance accumulator

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        sqdelta = x**2 - self.sqmean
        self.sqmean += sqdelta / self.n

    def get_mean(self):
        return self.mean.mean()

    def get_variance(self):
        n_total = self.n * math.prod(self.sqmean.shape)
        return (self.sqmean.mean() - self.get_mean()**2) * n_total / (n_total - 1)

    def get_std(self):
        return self.get_variance().sqrt()


if __name__ == "__main__":
    import time

    import torch

    a = torch.rand(1000, 80, 208)
    print("Traditional:")
    t0 = time.time()
    mean, std = a.mean().item(), a.std().item()
    t1 = time.time()
    print(f"mean: {a.mean().item():.3f}\tstd: {a.std().item():.3f}")
    print(f"Time elapsed: {t1 - t0:.5f}s")

    print()

    print("Running:")
    t0 = time.time()
    stats = OnlineStatsCalculator()
    for elem in a:
        stats.update(elem)
    t1 = time.time()
    print(f"mean: {stats.get_mean().item():.3f}\tstd: {stats.get_std().item():.3f}")
    print(f"Time elapsed: {t1 - t0:.5f}s")
