# Principal Component Analysis - Generative Adversarial Network
![Walk1](https://s8.gifyu.com/images/walk1.gif)
![Walk2](https://s8.gifyu.com/images/walk2.gif)
![Walk3](https://s8.gifyu.com/images/walk3.gif)
</br>
Using Deep Convolutional GANs and Unsupervised Learning (Principal Component Analysis) to Generate Cats.

# How To Use This Repository
* ## Requirements
  * Python 3
  * Keras (I use ```2.3.1```)
  * Tensorflow (I use ```1.14.0```)
  * Sklearn
  * Scipy
  * Numpy
  * Matplotlib
  * PIL
  * [Keract](https://github.com/philipperemy/keract) (for Model Visualization)
* ## Dataset
  * [Download from Kaggle](https://www.kaggle.com/spandan2/cats-faces-64x64-for-generative-models)
* ## Documentation
  * ## [PCA GAN Training](https://nbviewer.jupyter.org/github/vee-upatising/PCA-GAN/blob/main/PCA%20GAN%20Training.ipynb)
    * This script is used to define the DCGAN class, train the Generative Adversarial Network, generate samples, and save the model at every epoch interval.
    * The Generator and Discriminator models were designed to be trained on an 8 GB GPU. If you have a less powerful GPU then decrease the conv_filter and kernel parameters accordingly.
    
    * ### User Specified Parameters:
        * ```data_path```: File path pointing to folder containing dataset.
        * ```img_dimensions```: Tuple representing the dimensions of the images inside the dataset.
        * ```model_path```: File path pointing to folder where you want to save to model as well as generated samples.
        * ```interval```: Integer representing how many epochs between saving your model.
        * ```epochs```: Integer representing how many epochs to train the model.
        * ```batch```: Integer representing how many images to train at one time. If batch size is less than 8, alter the save_img function to plot less images. Ideally this number would be a factor of the size of your dataset.
        * ```conv_filters```: Integer representing how many convolutional filters are used in each convolutional layer of the Generator and the Discrminator.
        * ```kernel```: Tuple representing the size of the kernels used in the convolutional layers.
        * ```png```: Boolean flag, set to True if the data has PNGs to remove alpha layer from images.
        
     * ### DCGAN Class:
        * ```__init__(self)```: The class is initialized by defining the dimensions of the input vector as well as the output image. The Generator and Discriminator models get initialized using ```build_generator()``` and ```build_discriminator()```.
        * ```build_generator(self)```: Defines Generator model. There are 5 convolutional layers, upsampling from ```8x8x8``` to ```64x64x3```. Gets called when the DCGAN class is initialized.
        * ```build_discriminator(self)```: Defines Discriminator model. There are 5 convolutional layers, downsampling from ```64x64x3``` to ```1``` scalar prediction. Gets called when the DCGAN class is initialized.
        * ```load_data(self)```: Loads data from user specified file path, ```data_path```. Uses PCA to project the image dataset onto a lower dimension as X_Train dataset. Processes image dataset and reshape to 4 dimensions for Y_Train dataset. Gets called in the ```train()``` method.
        * ```train(self, epochs, batch_size, save_interval)```: Trains the Generative Adversarial Network. Each epoch trains the model using the entire dataset split up into chunks defined by ```batch_size```. If epoch is at ```save_interval```, then the method calls ```save_imgs()``` to generate samples and saves the model of the current epoch.
        * ```save_imgs(self, epoch, gen_imgs, y_points)```: Saves the model and generates prediction samples for a given epoch at the user specified path, ```model_path```. Each sample contains 8 generated predictions and 8 training samples. If the batch size is less than 8 then this function needs to be modified.
    
  * ## [PCA GAN Inference](https://nbviewer.jupyter.org/github/vee-upatising/PCA-GAN/blob/main/PCA%20GAN%20Inference.ipynb)
    * This script is used to perform inference on Generator models trained by the ```PCA GAN Training``` script and interpolate points in the latent space of the Generator model input.
    * The pretrained model provided, ```model.h5```, can be used with this notebook.
    * The interpolation of points can be used to make GIFs of walking through the latent space of the Generator model input such as the GIF in this README.
    
    * ### User Specified Parameters:
        * ```data_path```: File path pointing to folder containing dataset.
        * ```img_dimensions```: Tuple representing the dimensions of the images inside the dataset.
        * ```model_path```: File path pointing to folder where you want to save to model as well as generated samples.
        * ```save_path```: File path pointing to folder where you want to save generated predictions of the trained model.
        * ```png```: Boolean flag, set to True if the data has PNGs to remove alpha layer from images.
        
  * ## [Model Visualization](https://nbviewer.jupyter.org/github/vee-upatising/PCA-GAN/blob/main/Model%20Visualization.ipynb)
    * This script is used to visualize the convolutional filters inside the Generator models trained by the ```PCA GAN Training``` script.
    * The pretrained model provided, ```model.h5```, can be used with this notebook.
    * This script uses the [Keract](https://github.com/philipperemy/keract) library to visualize what is happening at each convolutional filter when performing inference on a model.
    
    * ### User Specified Parameters:
        * ```data_path```: File path pointing to folder containing dataset.
        * ```img_dimensions```: Tuple representing the dimensions of the images inside the dataset.
        * ```model_path```: File path pointing to folder where you want to save to model as well as generated samples.
        * ```png```: Boolean flag, set to True if the data has PNGs to remove alpha layer from images.
 
* ## Generated Training Sample
![Training](https://i.imgur.com/qfXMsYm.jpg)

* ## Model Visualization
![Keract](https://s8.gifyu.com/images/Keract.gif)

* ## Generator Model Architecture
  * Using (5,5) Convolutional Kernels </br>
![Generator](https://i.imgur.com/toVb4MD.png)

* ## Discriminator Model Architecture
  * Using (5,5) Convolutional Kernels </br>
![Discriminator](https://i.imgur.com/MkgHCUt.png)
