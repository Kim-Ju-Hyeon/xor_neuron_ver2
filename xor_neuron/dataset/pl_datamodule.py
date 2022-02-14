from pytorch_lightning import LightningDataModule
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

from typing import Optional

class PL_DataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset_conf = config.dataset

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset_conf.name == 'mnist':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

            train_dataset = datasets.MNIST(root=self.dataset_conf.data_path,
                                           train=True,
                                           transform=transform,
                                           download=True)

            test_dataset = datasets.MNIST(root=self.dataset_conf.data_path,
                                          train=False,
                                          transform=transform,
                                          download=False)

            self.train_dataset, self.val_dataset = random_split(train_dataset, [50000, 10000])
            self.test_dataset = test_dataset

        elif self.dataset_conf.name == 'cifar10':
            if self.dataset_conf.augmentation:
                transform = transforms.Compose(
                    [transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])])
            else:
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])])

            self.train_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path,
                                             train=True,
                                             transform=transform,
                                             download=True)

            self.val_dataset = datasets.CIFAR10(root=self.dataset_conf.data_path,
                                           train=False,
                                           transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2471, 0.2435, 0.2616])]),
                                           download=True)

            self.test_dataset = self.val_dataset


        elif self.dataset_conf.name == 'cifar100':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            train_dataset = datasets.CIFAR100(root=self.dataset_conf.data_path,
                                              train=True,
                                              transform=transform,
                                              download=True)

            test_dataset = datasets.CIFAR100(root=self.dataset_conf.data_path,
                                             train=False,
                                             transform=transform,
                                             download=False)
            self.train_dataset, self.val_dataset = random_split(train_dataset, [40000, 10000])
            self.test_dataset = test_dataset

        else:
            raise ValueError("Non-supported dataset!")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.train.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.train.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.train.batch_size)

