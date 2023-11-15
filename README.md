# tdt05-miniproject

Repo containing the code for a TDT05 project to showcase the use of SSL in handling noisy images. The dataset used is the fashion_mnist dataset.
The experiment evaluates the use of an autoencoder, trained using SSL on unlabeled data, to remove noise from images and increase the performance of a simple MLP-classifier model on noisy images.

## Structure
- figures
- src
    - keras (saved keras models to reuse)
    - autoencoder.py (architecture and training code for the autoencoder for denoising)
    - data.py (fetching and processing training and test data)
    - models.py (architecture and training code for the classification task)
    - run_experiment.py (script to run the experiment)
    - visualizations.py (visualizations of images, noised, encoded/decoded)


## Running code
Run the ```run_experiment.py``` script from the src-folder to train an autoencoder on the unlabeled dataset, and test the effects of doing image denoising with the autoencoder on the fashion_mnist classification task.
