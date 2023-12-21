import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import glob
from sklearn.model_selection import train_test_split

def pre_process_data(folder_path):
    files = glob.glob(f'{folder_path}*.jpg')
    labels = [1 if 'dog' in x.split('/')[-1] else 0  for x in files]
    return list(zip(files, labels))

def split_datasets(data, split_size):
    train, test = train_test_split(data, test_size=split_size)
    train, val = train_test_split(train, test_size=split_size)
    return train, val, test

class GetDataset(Dataset):
    def __init__(self, data, img_size=32):
        self.data = data
        self.image_paths = [x[0] for x in data]
        self.labels = [x[1] for x in data]
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = torchvision.io.read_file(self.image_paths[index])
        image = torchvision.io.decode_jpeg(image)
        image = torchvision.transforms.functional.resize(image, (self.img_size, self.img_size))
        image = image / 255.0
        return image, torch.Tensor([label]).float()

def get_loader(data, img_size = 32, batch_size = 32, num_workers=3, shuffle=True):
    dataset = GetDataset(data, img_size=img_size)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return loader

