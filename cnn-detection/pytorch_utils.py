import pandas as pd
import numpy as np

from skimage import io, transform

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import transforms, utils

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt
import matplotlib.patches as patches


pref = "../../Training_Data/CNN_labeled_training_data/train/images"

train = pd.read_csv("../../Training_Data/CNN_labeled_training_data/train/_annotations.csv")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2  # 1 class (smoke) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

class SmokeDataset(Dataset):

    def __init__(self, dataframe, image_dir, transform=None):
        super().__init__()

        self.image_ids = dataframe['filename'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['filename'] == image_id]

        image = io.imread(f'{self.image_dir}/{image_id}')

        boxes = records[['xmin', 'ymin', 'xmax', 'ymax']].values
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class
        labels = torch.ones((records.shape[0],), dtype=torch.int64)
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)
        
        target = {}
        target['image'] = image
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        
        if self.transform:
            target = self.transform(target)
            target['labels'] = labels
        


        return target

    def __len__(self) -> int:
        return self.image_ids.shape[0]
    
    
def show_box(image, boxes, axis):
    """Show image with landmarks"""
    plt.imshow(image)
    
    rect = patches.Rectangle((boxes[:, 0], boxes[:, 1]), boxes[:, 2]-boxes[:, 0], boxes[:, 3]-boxes[:, 1], linewidth=1, edgecolor='r', facecolor='none')

    axis.add_patch(rect)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
def display_dataset(dataset):
    for i in range(len(dataset)):
        sample = dataset[i]

        ax = plt.subplot(2, 2, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_box(sample["image"], sample['boxes'], ax)


        if i == 3:
            plt.show()
            break

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, boxes = sample['image'], sample['boxes']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'boxes': torch.from_numpy(boxes)}
    

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, boxes = sample['image'], sample['boxes']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        boxes = np.multiply(boxes, [new_w / w, new_h / h, new_w / w, new_h / h])

        return {'image': img, 'boxes': boxes, 'image_id' : sample["image_id"]}
    

final_dataset = SmokeDataset(train, pref, transform=transforms.Compose([Rescale((640, 480)), ToTensor()]))


dataloader = DataLoader(final_dataset, batch_size=4,
                        shuffle=True, num_workers=0)

# Helper function to show a batch
def show_boxes_batch(sample_batched):
    images_batch, boxes_batch, labels_batch = \
            sample_batched['image'], sample_batched['boxes'], sample_batched['labels']
    batch_size = len(images_batch)
    im_size = images_batch.size(3)
    
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    print([labels_batch[i, :] for i in range(len(labels_batch))])

    for i in range(batch_size):
        axis = plt.subplot()
        rect = patches.Rectangle((boxes_batch[i, :, 0] + i * im_size + (i + 1) * grid_border_size, boxes_batch[i, :, 1] + grid_border_size), boxes_batch[i, :, 2]-boxes_batch[i, :, 0], boxes_batch[i, :, 3]-boxes_batch[i, :, 1], linewidth=1, edgecolor='r', facecolor='none')

        axis.add_patch(rect)

        plt.title('Batch from dataloader')

# if you are using Windows, uncomment the next line and indent the for loop.
# you might need to go back and change "num_workers" to 0.

# if __name__ == '__main__':
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['boxes'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_boxes_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break

