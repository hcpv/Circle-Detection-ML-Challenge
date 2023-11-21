
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
```
img_size - Image size of the dataset. Default value is 64.
noise_level - Noise level that gets added to image. Default value is 0.1.
min_radius - Minimum radius. Default value is img_size/10.
max_radius - Maximum radius. Default value is img_size/2.
n_train - Number of training samples. Default value is 30000.
n_val - Number of validation samples. Default value is 10000.
n_testing - Number of testing samples. Default value is 10000.
filepath - Directory where prepared data would be saved. Default value is None.
```

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
```
img_size - Image size of the dataset. Default value is 64.
noise_level - Noise level that gets added to image. Default value is 0.1.
min_radius - Minimum radius. Default value is img_size/10.
max_radius - Maximum radius. Default value is img_size/2.
n_train - Number of training samples. Default value is 30000.
n_val - Number of validation samples. Default value is 10000.
n_testing - Number of testing samples. Default value is 10000.
filepath - Directory where prepared datasets would be saved. Default value is None.
savepath - Directory where model would be saved. Default value is None.
batch_size - Batch Size. Default value is  64.
epochs - Number of epochs. Default value is 200.
threshold - Threshold to get the accuracy. Default value is 0.9.
early_stop_thresh - Number of epochs after which early stop training. Default value is 10.
lr - learning rate. Default value is 1e-4.
```

Here is how this script works:

- Loads the prepared datasets. If not already prepared, runs the data preparation.
- Prepares the dataloaders.
- Runs the training iterations. After each epoch evaluates train and val accuracy.
- If the accuracy has not improved for early_stop_thresh epochs, then stops training.


## Discussion

All the training was done UW Madison CS labs' GPU machines (12 GB Nvidia 2080 TI).

To get the data ready, I picked a picture size of 64. For data split into training, validation, and testing sets, I went with a ratio of 60:20:20. At first, I made only 3000 examples for training, but that didn't work well for the model. So, I increased the training size to 30000, and it worked much better.

For modelling, I employed three layers of convolution with max-pooling and ReLU activation. Following the convolution, there are three fully connected layers. The final layer consists of three output neurons, each corresponding to the x-coordinate, y-coordinate, and radius of the circle. This model has around 9 million learnable parameters. Initially, I began with a two-layered CNN model with 1 million parameters. However, the model's performance was subpar, so I opted for the model discussed above. I used Mean Squared Error (MSE) for loss function. For Optimizer, I used Adam with learning rate of 1e-4. I trained the model for maximum of 200 epochs. However, in all the cases the training stopped early under 80 epochs.

To evaluate the model, I used a threshold of 0.9, meaning that instances where the "intersection over union" (IOU) metric between the true location of the circle in the image and the predicted location is greater than 0.9 are classified as correct.

I could only run experiments with different noise levels.
Here are the results:

| noise level   | val accuracy | test accuracy |
| ------------- | ------------ | ------------- |
| 0.1           | 95.7         | 96.2          |
| 0.2           | 94.5         | 94.4          |
| 0.3           | 89.7         | 89.4          |
| 0.4           | 77.2         | 76.8          |
| 0.5           | 59.3         | 59.2          |

For lower noise levels, the model was able to provide good test accuracies. However, when the noise level exceeded 0.3, the test accuracy decreased significantly. However, the training accuracy (numbers not mentioned above) was consistently above 90 percent.

So, the design of the model was effective for lower noise levels, but for higher noise levels, it began to overfit. To handle higher noise levels, a larger model or more data, or both, might have been helpful. The same might apply to a higher threshold in the IOU metric (I have only used 0.9; it is not expected to perform well with higher thresholds).
