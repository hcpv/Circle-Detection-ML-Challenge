
# Circle Detection ML Challenge

The goal of this challenge is to develop a circle detector that can find the location of a circle in an image with arbitrary noise. Your goal is to produce a deep learning model that can take an image as input and produce as output the location of the circle’s center (x,y) and its radius.


## Solution

My solution works in two stages: Data Preparation and Modelling.

### Data Preparation

Run this script to prepare the data.

```
$ python prepare_data.py
```

This script takes following arguments as inputs :
img_size - Image size of the dataset. Default value is 64.
noise_level - Noise level that gets added to image. Default value is 0.1.
min_radius - Minimum radius. Default value is img_size/10.
max_radius - Maximum radius. Default value is img_size/2.
n_train - Number of training samples. Defalut value is 30000.
n_val - Number of validation samples. Defalut value is 10000.
n_testing - Number of testing samples. Defalut value is 10000.
filepath - Directory where prepared data would be saved. Default value is None.

Here is how this script works:
- Firstly it prepares train, val and test datasets according to the arguments passed. 
- For this it uses Custom Torch Dataset defined in [dataset.py](dataset.py). 
- The custom dataset just takes number of samples and other arguments. It generates images and circle parameters and wraps it in a torch dataset.
- Once datasets are ready finally, it gets saved.

### Modelling

Run this script for model training and testing. This script can be executed directly without running the data preparation script as it handles the case where dataset was not already prepared.


```
$ python run.py
```

This script takes following arguments as inputs :
img_size - Image size of the dataset. Default value is 64.
noise_level - Noise level that gets added to image. Default value is 0.1.
min_radius - Minimum radius. Default value is img_size/10.
max_radius - Maximum radius. Default value is img_size/2.
n_train - Number of training samples. Defalut value is 30000.
n_val - Number of validation samples. Defalut value is 10000.
n_testing - Number of testing samples. Defalut value is 10000.
filepath - Directory where prepared data would be saved. Default value is None.
savepath - Directory where model would be saved. Default value is None.
batch_size - Batch Size. Default value is  64.
epochs - Number of epochs. Default value is 1000.
threshold - Threshold to get the accuracy. Default value is 0.9.
early_stop_thresh - Number of epochs after which early stop training. Default value is 10.
lr - learning rate. Default value is 1e-4.

Here is how this script works:

- Loads the prepared datasets. If not already prepared, runs the data preparation.
- Prepares the dataloaders.
- Runs the training iterations. After each epoch tests train and val accuracy.
- if the accuracy has not improved for early_stop_thresh, then stops training.


## Discussion

To get the data ready, I picked a picture size of 64. For dividing the data into training, validation, and testing sets, I went with a ratio of 60:20:20. At first, I made only 3000 examples for training, but that didn't work well for the model. So, I increased the training size to 30000, and it worked much better. I couldn't increase the image size beacuse of disk space issues.

For modelling, I employed three layers of convolution with max-pooling and ReLU activation. Following the convolution, there are three fully connected layers. The final layer consists of three output neurons, each corresponding to the x-coordinate, y-coordinate, and radius of the circle. This model has around 9 million learnable parameters. Initially, I began with a two-layered CNN model with 1 million parameters. However, the model's performance was subpar, so I opted for the model discussed above. I used Mean Squared Error (MSE) for loss function.

To evaluate the model, I used a threshold of 0.9, meaning that instances where the "intersection over union" (IOU) metric between the true location of the circle in the image and the predicted location is greater than 0.9 are classified as correct.

Here are the results:

| noise level   | val accuracy | test accuracy |
| ------------- | ------------ | ------------- |
| 0.1           | 95.7         | 96.2          |
| 0.2           | 94.5         | 94.4          |
| 0.3           | 95.7         | 96.2          |
