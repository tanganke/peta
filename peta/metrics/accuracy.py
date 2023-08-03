from torch import Tensor


class Accuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, preds: Tensor, targets: Tensor):
        self.correct += (preds == targets).sum().item()
        self.total += len(preds)

    def compute(self):
        return self.correct / self.total
