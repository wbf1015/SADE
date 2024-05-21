import torch

class LossManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LossManager, cls).__new__(cls)
            cls._instance.losses = []
            cls._instance.weights = []
        return cls._instance

    def add_loss(self, loss, weight):
        # 确保权重是一个tensor
        assert isinstance(weight, torch.Tensor), "Weight must be a PyTorch Tensor."
        assert isinstance(loss, torch.Tensor), "Loss must be a PyTorch Tensor."
        self.losses.append(loss)
        self.weights.append(weight)

    def get_weighted_loss(self):
        # 计算权重损失和
        total_loss = sum(loss * weight for loss, weight in zip(self.losses, self.weights))
        return total_loss

    def clear_loss(self):
        self.losses = []
        self.weights = []
