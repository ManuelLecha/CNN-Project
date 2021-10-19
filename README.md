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
