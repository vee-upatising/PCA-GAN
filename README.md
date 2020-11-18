# Principal Component Analysis - Generative Adversarial Network
![walk](https://thumbs.gfycat.com/OffbeatImpracticalCirriped-size_restricted.gif)
</br>
Using Deep Convolutional GANs and Unsupervised Learning (Principal Component Analysis) to Generate Cats.

# How To Use This Repository
* ## Jupyter Notebook
  * ## PCA GAN Training.ipynb
    * This notebook is used to define the DCGAN class, train the Generative Adversarial Network, generate samples, and save the model at every epoch interval.
    * The Generator and Discriminator models were designed to be trained on an 8 GB GPU. If you have a less powerful GPU then decrease the conv_filter and kernel parameters accordingly.
    
    * ### DCGAN Class:
        * ```__init__(self)```: The class is initialized by defining the dimensions of the input vector as well as the output image. The Generator and Discriminator models get initialized using ```build_generator()``` and ```build_discriminator()```.
        * ```build_generator(self)```: Define Generator model. There are 5 convolutional filters, upsampling from ```8x8x8``` to ```64x64x3```. Gets called when DCGAN class is initialized.
        * ```build_discriminator(self)```: Define Discriminator model. There are 5 convolutional filters, downsampling from ```64x64x3``` to ```1``` scalar prediction. Gets called when DCGAN class is initialized.
        * ```load_data(self)```: Load data from user specified file path, ```data_path```. Use PCA to project image dataset onto a lower dimension as X_Train dataset. Process image dataset and reshape to 4 dimensions for Y_Train dataset. Gets called in the ```train()``` method.
        * ```train(self, epochs, batch_size, save_interval)```: Train the Generative Adversarial Network. Each epoch trains the model using the entire dataset split up into chunks defined by ```batch_size```. If epoch is at ```save_interval```, call ```save_imgs()``` to generate samples and save model of current epoch.
        * ```save_imgs(self, epoch, gen_imgs, y_points)```: Save the model and generate prediction samples for a given epoch at the user specified path, ```model_path```. Each sample contains 8 generated predictions and 8 training samples. If the batch size is less than 8 then this function needs to be modified.
    
    * ### User Specified Parameters:
        * ```data_path```: File path pointing to folder containing dataset.
        * ```model_path```: File path pointing to folder where you want to save to model as well as generated samples.
        * ```interval```: How many epochs between saving your model.
        * ```epochs```: How many epochs to train the model.
        * ```batch```: How many images to train at one time. If batch size is less than 8, alter the save_img function to plot less images. Ideally this number would be a factor of the size of your dataset.
        * ```conv_filters```: How many convolutional filters for each convolutional layer of the Generator and the Discrminator.
        * ```kernel```: Size of kernel used in the convolutional layers.
        * ```png```: Boolean flag, set to True if the data has PNGs to remove alpha layer from images.
    
  * ## PCA GAN Inference.ipynb
    * This notebook is used to perform inference on Generator models trained by the ```PCA GAN Training``` script and interpolate points in the latent space of the Generator model input.
    * The pretrained model provided, ```model.h5```, can be used with this notebook.
    * The interpolation of points can be used to make GIFs of walking through the latent space of the Generator model input such as the GIF in this README.
    
    * ### User Specified Parameters:
        * ```data_path```: File path pointing to folder containing dataset.
        * ```model_path```: File path pointing to folder where you want to save to model as well as generated samples.
        * ```save_path```: File path pointing to folder where you want to save generated predictions of the trained model.
        * ```png```: Boolean flag, set to True if the data has PNGs to remove alpha layer from images.
        
