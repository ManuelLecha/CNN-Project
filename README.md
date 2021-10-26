# CNN-Project

## Experiments and comments

### test:

The confusion matrix (in .out file) is size 28x28, but should be 29x29. 

Reason found: Both "Woven fabric" directiories (below Train and Valid) are empty. 

That because of typo in .csv file. Instead of "Woven fabric" the "Woven fabric " was found.

### 0:

The same code was run with correct dataset (15 epochs).

Both plots (accuracy & loss) show the improving performance of the model. 

We decided to keep last saved weights (Validation accuracy: 0.4187) and continue training from this point. 

NOTE: wrong confusion matrix was genereted - because of shuffle=True when using ImageDataGenerator for validation dataset. 

### 1:

Parameters from previous training was not changed. 

This time we set the number of epochs to 50.

We started training from last saved weights during experiment 0.

At the end of trainig both (accuracy and loss) plots can indicate to overfitting. 


### 1_Batch_Normalization:
Add Batch Norm after each of the conv layers

Lot of overfitting 

The model stops after 33 epochs because valid acc is decreasing (Early stopping patiente 5)

Our best model achieves 64%

Decision: Try another kind of extra regularization ---> L2 reg


### 1_Batch_Norm_L2:


Our best model achieves 67.22%

The model stills overfit too much after aprox 33 epochs but gets better accuracy (Early stopping patiente 5)

L2 to 128 up conv layers and 256 Dense layers

Decision: Add strong reg --> Dropout (We will try with 20% and 10%)


### 1_Batch_Norm_L2_Dropout_20:

Dropout to 258 up



