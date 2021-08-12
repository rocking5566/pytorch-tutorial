import PIL.Image as Image
import torch.onnx
import os
import time
import torch
import torch.nn as nn
import torchvision


class MnistImageDataset(torch.utils.data.Dataset):
  def __init__(self, img_dir, transform=None):
    self.img_dir = img_dir
    self.img_names_label = []
    for cls in range(10):
      dir = os.path.join(self.img_dir, str(cls))
      for i in os.listdir(dir):
        self.img_names_label.append((i, cls))

    self.transform = transform

  def __len__(self):
    return len(self.img_names_label)

  def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir,
                            str(self.img_names_label[idx][1]), self.img_names_label[idx][0])
    image = Image.open(img_path, mode='r')
    if self.transform:
      image = self.transform(image)

    image = torchvision.transforms.ToTensor()(image)
    label = self.img_names_label[idx][1]
    return image, label


def GetTrainTestDataLoader(batch_size):
  train_dataset = MnistImageDataset(
      img_dir='/Data/dataset_zoo/mnist/train')
  test_dataset = MnistImageDataset(img_dir='/Data/dataset_zoo/mnist/test')
  # Official dataset
  # train_dataset = torchvision.datasets.MNIST(root='./data/',
  #                                            train=True,
  #                                            transform=torchvision.transforms.ToTensor(),
  #                                            download=True)

  # test_dataset = torchvision.datasets.MNIST(root='./data/',
  #                                           train=False,
  #                                           transform=torchvision.transforms.ToTensor())

  train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True, num_workers=2, 
                                             pin_memory=False, prefetch_factor=2, 
                                             persistent_workers=False)

  test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False, num_workers=2, 
                                            pin_memory=False, prefetch_factor=2, 
                                            persistent_workers=False)
  return train_loader, test_loader


class ConvNet(nn.Module):
  # Convolutional neural network (two convolutional layers)
  def __init__(self, num_classes=10):
    super(ConvNet, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2))
    self.layer2 = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2))
    self.fc = nn.Linear(7*7*32, num_classes)

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)
    return out


def test(model, device, test_loader):
  model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
      images = images.to(device)
      labels = labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(
        100 * correct / total))


def train(model, device, train_loader, num_epochs=1, learning_rate=0.0001):
  model.train()
  # Loss and optimizer
  loss_func = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  total_step = len(train_loader)

  t = time.time()
  for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
      # Copy data to target device
      images = images.to(device)
      labels = labels.to(device)
      # Forward pass
      outputs = model(images)
      loss = loss_func(outputs, labels)

      # Backward and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if i % 50 == 0:
        print('Epoch [{}/{}], Time: {:.4f}, Step [{}/{}], Loss: {:.4f}'
              .format(epoch+1, num_epochs, time.time() - t, i, total_step, loss.item()))
        t = time.time()


if __name__ == '__main__':
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # Hyper parameters
  num_classes = 10
  batch_size = 100
  num_epochs = 1
  learning_rate = 0.0001

  # Dataset
  train_loader, test_loader = GetTrainTestDataLoader(batch_size)

  model = ConvNet(num_classes).to(device)
  # If you do not want to train from scratch, you can load previous weight.
  # model.load_state_dict(torch.load('model.ckpt'))

  train(model, device, train_loader, num_epochs, learning_rate)
  test(model, device, test_loader)

  # Save the model checkpoint
  torch.save(model.state_dict(), 'model.ckpt')
