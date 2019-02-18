# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

#### 1. Required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* writeup.md summarizing the results
* model.h5 containing a trained convolution neural network for track 1
* run1.mp4 is a recorded video shows one lap of driving on track 1
* model_track2.h5 containing a trained convolution neural network for track 2
* run2.mp4 is a recorded video shows one lap of driving on track 2

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

run1.mp4 is a recorded video shows one lap of driving on track.

The drive.py was modified to convert the image to BGR color space (code line 66-67), which made the model works more effective.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. 

The code in model.py uses a Python generator to generate data for training rather than storing the training data in memory. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network introduced by Nvida in the lecture,  with three 5x5 filter sizes and depths as 24/36/48, two 3x3  filter sizes and depths as 64,  and 4 fully connected layers(model.py lines 71-80) . 

The model includes RELU layers to introduce nonlinearity (code line 71-75), and the data is normalized in the model using a Keras lambda layer (code line 69). In order to save memory, a cropping layer is also included (code line 70).

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 83-95). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I also tried adding a dropout layer after first CNN or first fully connected layer, the result didn't change a lot, so I removed it. Then I used the early stop to reduce the overfitting (code line 87-90), a call back was registered with val_loss as monitor. The early stop works very well, following is a chart shows how the loss and val_loss converged and stopped training.

![history_1](\images\history_1.png)

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 83).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the sample data provided by the lecture, which including a combination of center lane driving, left/right cameras, and recovering from the left and right sides of the road.

I flip images to get some right turn images (code line 42, 46, 48 and 50). I also used left and right cameras with an adjusted angle 0.2 (code line 47 and 49).

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the established LeNet in the previous lecture. I thought this model might be appropriate because it's proven to be a good image classifier.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a high mean squared error on the training set and the validation set, which shows an underfitting. And the car drives not very well at the left turn, such as diving into water, and so on.

Then I changed to the Nvidia network introduced in the lecture, which performs much better with both loss/val_loss at 0.0xx level. But it still didn't drive well in the simulator.

There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I added images captured by left and right cameras, and flip all images, I met memory issue here, so changed the code to use a generator to feed the data (code line 25 - 55).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 68-80) consisted of a convolution neural network with the following layers and layer sizes.

| Layer           | Desc                                     |
| --------------- | ---------------------------------------- |
| Input           | 160x320 3 channels image                 |
| Lambda          | Normalizing                              |
| Crop            |                                          |
| Convolution 5x5 | 2x2 stride, output depth 24, activation relu |
| Convolution 5x5 | 2x2 stride, output depth 36, activation relu |
| Convolution 5x5 | 2x2 stride, output depth 48, activation relu |
| Convolution 3x3 | output depth 64, activation relu         |
| Convolution 3x3 | output depth 64, activation relu         |
| Flatten         |                                          |
| Fully connected | Output 100                               |
| Fully connected | Output 50                                |
| Fully connected | Output 10                                |
| Fully connected | Output 1                                 |



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the data provided by the lecture. Here is an example image of center lane driving:

![center_2017_12_15_15_48_49_916](\images\center_2017_12_15_15_48_49_916.jpg)

I checked the image of the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to adjust the direction. There're many kinds of recovery images in the data set, following is a sample:

![recovery_left_2016_12_01_13_32_46_689](\images\recovery_left_2016_12_01_13_32_46_689.jpg)

There're total 8036 data points in the data set.

To augment the data sat, I also flipped images and angles horizontally. For example, here is an image that has been flipped:

![flip_after_center_2016_12_01_13_31_13_279](\images\flip_after_center_2016_12_01_13_31_13_279.jpg)

![flip_before_center_2016_12_01_13_31_13_279](\images\flip_before_center_2016_12_01_13_31_13_279.jpg)



I also used images from left and right images with steering angle adjust 0.2 (code line 47 and 49). There're 48216 data points finally.

A crop layer was added (code line 70), which cropped top 50 and bottom 20 line of pixels.

I finally randomly shuffled the data set and put 20% of the data into a validation set (code line 17). 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the change of loss and valid_loss, which didn't change a lot after epoch 10. I used an adam optimizer so that manually training the learning rate wasn't necessary.



#### 4. Track Two

I drove 3 laps on the track 2 manually and recorded data. The track 2 is a little difficult so that some data shall be removed, such as car fully stop, and so on. Finally I got 31632 raw data points, including center, left and right cameras.

Since an early stop and saving model after each epoch is employed (code line 87 - 94), I changed the epoch to a very large number 200. Firstly I set min_delta as 0.01 and  patience as 6, the result is not good. Finally I set min_delta as 0.001 and  patience as 30, the training stopped at epoch 100. 

The run2.mp4 show the car can stay on the road for one lap.

On the other hand, the car often cross the center lane line, which is normal since the training data I recorded includes many such cases. I tried my best to keep the car at right lane when driving it manually, but it's not that easy. I think the enhancement can be split the whole lap to many small pieces, and try to keep at right lane in each piece.
