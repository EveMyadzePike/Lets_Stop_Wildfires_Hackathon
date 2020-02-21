[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/WIVIV/Project_3_AFI/blob/master/Proj_3_WF_GRID_TL_with_tensorboard.ipynb)


# A Fire Inside
## Project 3
### By: Ash Ranu, Jeneia Mullins, and Zach Steindam

## Lets Stop Wildfires Hackathon


AI for Mankind launched a Let’s Stop Wildfires Hackathon with the goal to come up with ideas to help solve California wildfires crisis. 

Images captured from the High Performance Wireless Research and Education Network (HPWREN) Cameras was collated into labeled (smoke/no smoke) set of images. The task is to build a wildfire smoke classifier to predict if there is smoke given a small grid image.

Camera network is as follows:
<br/>
<br/>
<div style="text-align:center"><img src="Images\HPWREN_t.jpg" /></div>
<br/>
<br/>
<div style="text-align:center"><img src="Images\Sample_Images.png" /></div>
<br/>
<br/>
Number of training and validation images are as follows:
<div style="text-align:center"><img src="Images\fig_distribution.png" /></div>
<br/>
<br/>

### Convolutional Neural Network
We built a Convolutional Neural Network (CovNets) using TensorFlow in order to model the training set of images to predict the class of the validation set of images. We started with a simple custom built model.  


The role of the CNN is  to reduce the images into a form which is easier to process, without losing features which are critical for getting a good prediction but is also is scalable to massive datasets. To this end the CNN takes the RGB image and separates it by the three color planes — Red, Green, and Blue.



CovNets use filters (or a documented set of features) that classify details about the image. The first layer captures high-level features (such as edges, gradient, and color). The filter (or kernal) will slide across the image and document a convolved feature. The filter repeats this process until it has scanned across the entire image. Matrix multiplication calculates the bias weights to provide a robust Convoluted Feature Output. The main components of CNNs are summarized below:
<div style="text-align:center"><img src="Images\CNN_SUMMARY_1.png" /></div>
<br/>
<br/>
<div style="text-align:center"><img src="Images\CNN_SUMMARY_2.png" /></div>



Initially we built a custom CovNet model with 3 convolution layers, with each followed by a max pooling layer. 

We used three fully connected layers as follows:
* Fully connected input layer (flatten) takes the output of the previous layers, “flattens” them and turns them into a single vector that can be an input for the next stage.
* The first fully connected layer takes the inputs from the feature analysis and applies weights to predict the correct label.
* Fully connected output layer gives the final probabilities for each label.

Custom Model structure, parameters and results are as follows:
<br/>
<br/>
<div style="text-align:center"><img src="Images\fig_base_model_summary.png" /></div>
<br/>


### Transfer Learning 
We then used transfer learning to apply a pre-trained model to our dataset. Data scientists often utilize transfer learning when attempting to train large datasets that can take hours and significant GPU to process on a personal computer. We chose VGG-19 to apply to our dataset of images. VGG-19 is a convolutional neural network that is trained on millions of images from the ImageNet database. The network is 19 layers deep and can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. VGG19 is structured as follows:
<br/>
<br/>

<div style="text-align:center"><img src="Images\VGG19.png" /></div>
<br/>
<br/>

We froze the FC layers and replaced the prediction head of the model as follows:
<br/>
<br/>
<div style="text-align:center"><img src="Images\VGG_19_Diagram_TL.png" /></div>


Transfer Learning Model structure, parameters and results are as follows:
<br/>
<br/>
<div style="text-align:center"><img src="Images\fig_tl_model_summary.png" /></div>

### Fine Tuning 
Lastly we used a process called fine tuning to help further tune the VGG19 model to our dataset. We unfroze  the last 5 layers of VGG19 and the retrained it using a lower learning rate. This method led to higher accuracy than transfer learning via feature extraction.
images\VGG19.png
<div style="text-align:center"><img src="Images\VGG_19_Diagram_FT.png" /></div>
<br/>
<br/>
Fine Tuning Model structure, parameters and results are as follows:
<div style="text-align:center"><img src="Images\fig_ft_model_summary.png" /></div>
<br/>
<br/>

### Tensorboard
We used Tensorboard to interactively visualize our results. TensorBoard is a tool for providing the measurements and visualizations needed during the machine learning workflow. It enables tracking experiment metrics like loss and accuracy, visualizing the model graph, projecting embeddings to a lower dimensional space, and much more.

<div style="text-align:center"><img src="Images\tensorboard.gif" /></div>
<br/>
<br/>

### Colab
We used Colab for all coding and collaboration. Colab allowed us to use a cloud compute resource (including GPU) for free and sync our notebooks with GitHub. It was a huge time saver.
<br/>
<br/>
<div style="text-align:center"><img src="Images\Colab.png" /></div>



