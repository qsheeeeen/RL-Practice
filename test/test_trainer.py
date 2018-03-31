import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from self_driving_car.trainer import Trainer


def main():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data',
                       train=True,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size,
        shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data',
                       train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size,
        shuffle=True, **kwargs)

    del env

    trainer = Trainer(CNNPolicy, inputs, outputs, './agent/data.hdf5')

    trainer.fit(batch_size=32, epochs=10)

    trainer.save()


if __name__ == '__main__':
    main()
