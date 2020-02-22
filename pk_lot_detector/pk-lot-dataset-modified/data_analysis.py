import pandas as pd
import matplotlib.pyplot as plt
import random
from matplotlib import patches

# read the csv file using read_csv function of pandas
train = pd.read_csv('training/training.csv')
train.head()

print('Number of images:', train['img_path'].nunique())

# Number of classes
print('Class count: ', train['occupancy'].value_counts())

print(train.shape)

index = random.randint(0, train.shape[0])
sample_img = train['img_path'][index]
image = plt.imread(sample_img)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
plt.imshow(image)

# iterating over the image for different objects
for _, row in train[train.img_path == sample_img].iterrows():

    occupancy = row.occupancy
    xmin = row.xmin
    ymin = row.ymin
    xmax = row.xmax
    ymax = row.ymax

    print(sample_img, occupancy, xmin, ymin, xmax, ymax)

    if occupancy == 'free':
        edgecolor = 'b'
    else:
        edgecolor = 'r'

    width = xmax - xmin
    height = ymax - ymin

    # add axes to the image
    ax = fig.add_axes([0, 0, 1, 1])

    # add bounding boxes to the image
    rect = patches.Rectangle((xmin, ymin), width, height, edgecolor=edgecolor, facecolor='none')
    ax.add_patch(rect)

plt.show()
