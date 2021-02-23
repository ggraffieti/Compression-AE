from models.simple_AE import SimpleAE
from data.data_loaders import get_MNIST_training_set, get_MNIST_test_set
import torch
from utils import to_img
from torchvision.utils import save_image


def train(epoch: int) -> None:
    train_loader = get_MNIST_training_set(batch_size=128)
    simple_ae.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = simple_ae(data)
        loss = 0.5 * loss_function(output, data)
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(epoch: int) -> None:
    test_loader = get_MNIST_test_set(batch_size=128)
    simple_ae.eval()
    test_loss = 0
    save_imgs = True
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = simple_ae(data)
            test_loss += loss_function(output, data).item()
            if save_imgs:
                images = to_img(torch.cat((data, output), dim=0))
                save_image(images, "pic_ep_{}.png".format(epoch), nrow=16)
                save_imgs = False
    test_loss /= len(test_loader)
    print('\nTest set: Avg. loss: {:.4f}\n'.format(
        test_loss))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    simple_ae = SimpleAE(latent_space_dimension=5)
    simple_ae.to(device)
    loss_function = torch.nn.MSELoss()  # try L1 loss
    optimizer = torch.optim.Adam(simple_ae.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
    epochs = 30
    for ep in range(epochs):
        train(ep)
        test(ep)
        scheduler.step()
    torch.save(simple_ae.state_dict(), "../data/models/simple_ae.pth")
