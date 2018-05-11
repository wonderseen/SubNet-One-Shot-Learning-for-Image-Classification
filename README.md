# SubNet: One-Shot-Learning-for-Image-Classification
A complete project for image classification with an interface provided for ROC test.  <br />
In my testset, after training for 600 iteration, I achieve an mean-accuracy of 98.21% and the AUC approximately equals to 1.

# Environment Configuration
1. Tensorflow-cpu 1.3 for Linux
2. opencv-python-dev, numpy and some common 3rd party modules for python

# Project Structure
>*SubNet*  <br />
>*|*  <br />
>*|--SubNet.py: the definition of SubNet and important operations*  <br />
>*|--train_gray.py: script for training and testing*  <br />
>*|*  <br />
>*|__dataset: dictionary of your image dataset*  <br />
>*|__model: dictionary to place the saved-model*  <br />
>*|__tools:*  <br />
>     *|--Augement.py*  <br />
>     *|--Draw_ROC_iteration.py*  <br />
>     *|--ReadData.py*  <br />

# Scripts Description
1. ReadData.py: Load gray images of shape [h,w,1] (types supported: jpg/jpeg/png)
2. Augement.py: Contain image augement operators.
3. SubNet.py:  Main class of the SubNet model.
4. Draw_ROC_iteration.py: a function for computing multi-iteration ROC results

# Classify Your Dataset
1. Modify ReadData.py to fit your dataset
2. Modify the input shape to match your image dimension
3. Warn that do not the output prediction's shape
4. Run 'train_gray.py' to start the training process.
5. Mark RGB_net.train() as comment out, uncomment RGB_net.test_single_threshold() and rerun 'train_gray.py' to start the test process.

# Contact
Feel free to share your doubts with me.
