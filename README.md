# Machine Learning Project 1 - Group lineardepression

## Project Overview 

Cardiovascular diseases, particularly heart diseases, are a leading cause of death worldwide. This project aims to predict heart attacks based on various health indicators obtained by the Behavioral Risk Factor Surveillance System (BRFSS). The BRFSS is an ongoing health surveillance system for adults in the United States. 
This repository implements a machine learning-based approach for predicting heart attacks using data from the Behavioral Risk Factor Surveillance System (BRFSS)
After extensive data processing and model selection, a F1 score of 0.429 and an accuracy of 0.866 were achieved.

## Repository structure

```
data                   <- project data files   
costs.py               <- contains all the required cost functions and a function to compute the F1 score
implementations.py     <- contains all the required functions used for the differents models
data_processing .py    <- all the functions that process the data - in order to apply it to the models
helpers.py             <- helpers functions
run.py                 <- our result python file - create a submission file with predicitions based on the x_test data set
```   


- **implementations.py**: Contains all required functions for each model used.
  
- **helpers.py**: Includes various helper functions, such as:
    - `load_csv`: Modified to remove headers.
    - `cross_validation`: Performs cross-validation, providing accuracy and F1 scores for each model.
  
  The `cross_validation` function helps select the best parameters for the models.

- **data_processing.py**: Includes functions for:
    - Data cleaning.
    - Handling missing (`NaN`) values.
    - Selecting the most correlated features to create a feature subset with the highest information.



## Results

By running the file `run.py`, it creates a submission file with predicitions based on the x_test data set

A file `final_submission.csv` should be created

By submitting it on AiCrowd, we obtained the following performance : 

- **F1 score**: 0.429
- **Accuracy**: 0.866


## Contributors

  Levin Hertrich, 
  Data Science

  Eugénie Cyrot, 
  Mechanical Engineering
  
  Gabriel Marival, 
  Applied Mathematics
