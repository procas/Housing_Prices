# Aim: To predict the housing prices in a given locality

### Technologies used: Pandas, Matplotlib, Standard Scaler, Label encoder, Random Forest Regressor, Imputer

### Data gathering
The dataset was given with necessary attributes such as housing price in locality, living area (sq ft.), tax information, waterfront, number of bedrooms and bathrooms, etc.

### Data cleaning
Imputer was used to handle the data cleaning, in case of missing or NaN values in the dataset, or other discrepancies in the data prior to processing. The imputer was made to use 'mean' strategy for missing data replacement.

### Data preparation
The given dataset was split into training and test sets, followed by encoding the categorical values into numerical for the use of analysis.

### The prediction
The model chosen for prediction in our case is the "Random Forest Regressor" algorithm. The advantage of this algorithm is that it is able to handle both classification and regression problems, this makes sense from our perspective since we need to essentially classify the houses according to their respective price levels, as well as make a fair prediction about the pricing which should be assigned to a target house. Moreover, the variance of the model is lower than a single regressor or a single decision tree model. The cleaned training data is fit into this model for the prediction. 

### Visualisation
The data visualisation is handled with a pyplot from matplotlib with the predicted values against the test set.
