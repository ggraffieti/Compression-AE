from models.simple_AE import SimpleAE
from data.data_loaders import get_MNIST_training_set
import torch


def encode_mnist():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    simple_ae = SimpleAE(latent_space_dimension=5)
    simple_ae.to(device)
    simple_ae.load_state_dict(torch.load("../data/models/simple_ae.pth"))
    train_loader = get_MNIST_training_set(batch_size=128)
    for itr, (data, label) in enumerate(train_loader):
        data = data.to(device)
        code = simple_ae.encode(data)
        codes.append(code.detach().cpu())
        labels.append(label)
        if itr % 20 == 0:
            print('[{}/{} ({:.0f}%)]'.format(
                itr * len(data), len(train_loader.dataset),
                100. * itr / len(train_loader)))


if __name__ == "__main__":
    codes = []
    labels = []
    encode_mnist()
    tensor_codes = torch.cat(codes, dim=0)
    tensor_labels = torch.cat(labels, dim=0)
    print(tensor_codes.shape)
    torch.save(tensor_codes, "../data/mnist/codes_simple_ae.pt")
    torch.save(tensor_labels, "../data/mnist/labels_simple_ae.pt")
