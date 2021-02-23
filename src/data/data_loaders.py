from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


_default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])


def get_MNIST_training_set(batch_size: int) -> DataLoader:
    mnist_dataset = MNIST(root="../data/mnist", train=True, download=True, transform=_default_transform)
    mnist_train_loader = DataLoader(dataset=mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                    pin_memory=True, drop_last=False)
    return mnist_train_loader


def get_MNIST_test_set(batch_size: int) -> DataLoader:
    mnist_dataset= MNIST(root="../data/mnist", train=False, download=True, transform=_default_transform)
    mnist_test_loader = DataLoader(dataset=mnist_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                   pin_memory=True, drop_last=False)
    return mnist_test_loader
