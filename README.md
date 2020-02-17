A Fire Inside
Project 3
By: Ash Ranu, Jeneia Mullins, and Zach Steindam

Lets Stop Wildfires Hackathon


AI for Mankind launched a Let’s Stop Wildfires Hackathon with the goal to come up with ideas to help solve California wildfires crisis. 

Images captured from the High Performance Wireless Research and Education Network (HPWREN) Cameras was collated into labeled (smoke/no smoke) set of images. 

![camera_network](\HPWREN.jpg)

![dataset_image1](\dataset_image1.png)

![dataset_image2](\dataset_image2.png)

![dataset_image3](\dataset_image3.png)

![bar_chart2](\bar_chart2.png)


The task is to build a wildfire smoke classifier to predict if there is smoke given a small grid image.


We built a Convolutional Neural Network using TensorFlow in order to model the training set of images to predict the class of the valadiation set of images. We started with a simple custom built model.  


The role of the CNN is  to reduce the images into a form which is easier to process, without losing features which are critical for getting a good prediction but is also is scalable to massive datasets. To this end the CNN takes the RGB image and separates it by the three color planes — Red, Green, and Blue.

![rbg_panels](\rbg_panels.png)

Convolutional networks use filters (or a documented set of features) that classify details about the image. The first layer captures high-level features (such as edges, gradient, and color). The filter (or kernal) will slide across the image and document a convolved feature. The filter repeats this process until it has scanned across the entire image. Matrix multiplication calculates the bias weights to provide a robust Convoluted Feature Output.

![conv_matrix](\conv_matrix.gif)


In the custom model we included a "max pooling" layer. A pooling layer reduces the size of the convoluted feature in order to decrease the computational power required to process the data. Max pooling is preferred because it acts as a noise suppressant in feature extraction. These features are then fed into a deep neural net to measure predictions.

Also, we used three fully connected layers as follows:
Fully connected input layer (flatten) takes the output of the previous layers, “flattens” them and turns them into a single vector that can be an input for the next stage.

The first fully connected layer takes the inputs from the feature analysis and applies weights to predict the correct label.

Fully connected output layer gives the final probabilities for each label.

![fully_connected](\full_connected.jpeg)


We then used transfer learning to apply a pre-trained model to our dataset. Data scientists often utilize transfer learning when attempting to train large datasets that can take hours and significant GPU to process on a personal computer. We chose VGG-19 to apply to our dataset of images. VGG-19 is a convolutional neural network that is trained on millions of images from the ImageNet database. The network is 19 layers deep and can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals.

![vgg19](\vgg19.jpeg)


Lastly we used a process called fine tuning to help further tune the transfer learing VGG19 model to our dataset. We unfroze  the last 3 layers of VGG19 and the retrained it using a lower learning rate. Applying fine-tuning allows us to utilize pre-trained networks to recognize classes they were not originally trained on. This method led to higher accuracy than transfer learning via feature extraction.


We were able to deploy Tensorboard to visualize our results. TensorBoard is a tool for providing the measurements and visualizations needed during the machine learning workflow. It enables tracking experiment metrics like loss and accuracy, visualizing the model graph, projecting embeddings to a lower dimensional space, and much more.

![tensorboard](\tensorboard.jpg)




