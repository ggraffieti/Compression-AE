from models.cnn import MNIST_CNN
from models.simple_AE import SimpleAE
from data.data_loaders import get_MNIST_training_set, get_MNIST_test_set
import torch
from torch.utils.data import TensorDataset


def train(epoch: int) -> None:
    coded_mnist = torch.load("../data/mnist/codes_simple_ae.pt")
    coded_mnist_labels = torch.load("../data/mnist/labels_simple_ae.pt")
    train_ds = TensorDataset(coded_mnist, coded_mnist_labels)
    train_loader = torch.utils.data.dataloader.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2,
                                                          pin_memory=True, drop_last=True)
    ae.eval()
    optimizer = torch.optim.SGD(cnn.parameters(), lr=0.02, momentum=0.9)
    cnn.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            data = ae.decode(data)
        optimizer.zero_grad()
        output = cnn(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test() -> None:
    test_loader = get_MNIST_test_set(batch_size=128)
    cnn.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = cnn(data)
            test_loss += loss_function(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.view(pred.size())).sum()
    test_loss /= len(test_loader)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = MNIST_CNN()
    cnn.to(device)
    ae = SimpleAE(latent_space_dimension=5)
    ae.load_state_dict(torch.load("../data/models/simple_ae.pth"))
    ae.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    epochs = 10
    for ep in range(epochs):
        train(ep)
        test()
