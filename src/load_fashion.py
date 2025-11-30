from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

def get_fashion_mnist(batch_size=1):
    """
    Download and return Fashion-MNIST DataLoader.
    Data will be stored in ./data folder.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),     # Convert image to tensor (0~1)
    ])

    train_dataset = FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Fashion MNIST data loaded successfully!")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    return train_loader, test_loader


if __name__ == "__main__":
    get_fashion_mnist()
