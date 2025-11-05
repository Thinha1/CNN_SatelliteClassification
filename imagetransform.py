from lib import *

class ImageTransform():
  def __init__(self, resize, mean, std):
      self.data_transform = {
        'train': transforms.Compose([ #data augmentation
            transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),

        'val': transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        
        'test': transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
      }
  def __call__(self, img, phase ='train'):
    return self.data_transform[phase](img)