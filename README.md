# DCGAN Art Generation


## Description


This is the main module for the image generation project based on Deep Convolutional 
Generative Adversarial Network (DCGAN). This module is responsible for pretrained 
models management, interface, generation and upscaling images.

The creation of this project was motivated by the theoretical potential to reduce the 
human resources needed to create art. The main goal of this project is to simplify and 
algorithmize the process of image creation.

Additionally, this project was a form of self-founded experience to understand how 
Generative Models work, are created and trained. Specifically was chosen Generative 
Adversarial Networks (GANs) because they are relatively easy and fast to train. 

This project can help humans to make the process of art creation faster by providing 
some form of inspiration or even partially replacing human-created art.

In the end, it became clear that GANs are not that easy to train, and they are pretty 
unstable during training. The main problem for the training process was 
['mode collapse'](https://arxiv.org/abs/1406.2661). To battle this issue was implemented 
a lot of different modifications. The most successful was 
['WGAN-GP'](https://arxiv.org/abs/1704.00028) which helps with 'mode collapse' but at 
the cost of training time. In general, the FID score for the 'WGAN-GP' model was 128.259, 
which can be classified as a satisfactory result.


## Installation


Implementation was written using Python 3.9 

In file [requirements.txt](requirements.txt) are listed main libraries which are necessary 
for this project. For installing requirements run this command (inside project folder): 

	pip install -r requirements.txt

In folder [models](models) are all possible pretrained generators which can be chosen 
from interface to generate images. Models can be replaced for new pretrained models. 

In folder [real-esrgan](real-esrgan) are existing implementation of 
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN). This code already precompiled
and can run using GPU. Main purpose of this algorithm is to allow image upscaling.
   

## Usage


1. To start main app run [`main.py`](main.py)
2. Using interface choose generator type, random seed, images quantity and output size
3. Press 'Run' button to start generation process
4. Wait until end of generation (progress can be seen in integrated console)
5. Result will be saved in [output](output) folder
    

## Credits


* [Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., 
Ozair, S., Courville, A., & Bengio, Y. (2014, June 10). Generative Adversarial 
Networks](https://arxiv.org/abs/1406.2661)
* [Goodfellow, I. (2017, April 3). NIPS 2016 tutorial: Generative Adversarial 
Networks](https://arxiv.org/abs/1701.00160)
* [Brownlee, J. (2019, July 19). A gentle introduction to generative adversarial 
networks (Gans)](
https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/)
* [Radford, A., Metz, L., & Chintala, S. (2016, January 7). Unsupervised
representation learning with deep convolutional generative Adversarial 
Networks](https://arxiv.org/abs/1511.06434)
* [Liao, P., Li, X., Liu, X., & Keutzer, K. (2022, June 22). The ARTBENCH 
dataset: Benchmarking generative models with artworks](
https://arxiv.org/abs/2206.11404)
* [Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. 
(2018, January 12). Gans trained by a two time-scale update rule converge 
to a local Nash equilibrium](https://arxiv.org/abs/1706.08500)
* [Arjovsky, M., Chintala, S., & Bottou, L. (2017, December 6). Wasserstein 
Gan](https://arxiv.org/abs/1701.07875)
* [Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. 
(2017, December 25). Improved training of Wasserstein Gans](
https://arxiv.org/abs/1704.00028)
* [Wang, X., Xie, L., Dong, C., & Shan, Y. (2021, August 17). Real-ESRGAN:
Training real-world blind super-resolution with pure synthetic data](
https://arxiv.org/abs/2107.10833)
* [Wang, X., Xie, L., Dong, C., & Shan, Y. (2021, August 17). Real-ESRGAN 
aims at developing practical algorithms for General Image/video 
restoration](https://github.com/xinntao/Real-ESRGAN)


## License


Licensed under the [MIT](LICENSE) license.
