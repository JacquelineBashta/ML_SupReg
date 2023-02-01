# Dimensionality issue
### Reason
1. Data with high dimension (text, images and/or sound)
2. using OneHotEncoder to convert categories to numerical features ( sparse features)
### Effect
1. High computing cost
2. High training time
3. model sensitive to correlated features
4. find good solutions in sparser spaces (curse of dimensionality)
    - same observation divided over higher dimesnsions introduce alot of empty space (low data denisty): the dataset has become sparser. 
    A new instance is likely to fall further away from any other observation. 
    Therefore, predictions will be based on much larger extrapolations. 
    It will be more difficult to find well defined patterns supported by a large amount of observations, 
    and noise is more likely to play a bigger role in the model, thus creating a bigger risk of overfitting.

### Solution Dimension reduction
Feature selection : kick out some of the worst features ( based on its predictive value) , losing the information within these features
Feature engineering : features are re-engineered, compressed to smaller amount of features that hold as much information as possible.

Dimension reduction is Feature engineering.
#### Find best n_components

1.  PCA(n_component=None)
tbd

2.  PCA(n_component=0.999) 
might actually improve performace
Assuming the information was all reserved
PCA had only removed the redundent Features, which could have introduced noise in the modeling
hence removing these redundency might lead to increase of the performance

3. PCA(n_componenet=0.95)

might or might not improve the performance
Since we accepted losing some of the information variance, there will be a compromise between how much redundency we removed and how much information we lost.
If the information loss has bigger impact , then the new features will have lower model performance
If the redundency removal has bigger impact, then the new features will have higher model performance

4. PCA(n_componenet=int)
Avoid using it that way unless 
  a. you don't care about performance rather the low dimensionality of your model
  b. you search for that value via SearchCV and got the best number for your model


#### Consider

1. use StanderdScaler()  before PCA
You need to normalize the features or their variances will not be comparable. 
Think of a feature where the variance is a ratio to the range. 
A larger range produces a larger variance. 
You don't want the PCA to focus on variables with larger ranges.

2. Generated feature are not readable anymore

3. PCA and Categorical Data
https://towardsdatascience.com/famd-how-to-generalize-pca-to-categorical-and-numerical-data-2ddbeb2b9210

PCA can not be used directly with Categorical data,as the variation in Categiorical data can't be easily correlated with numerical features

Solution : FAMD (Factorial Analysis of Mixed Data)

```python
from prince import FAMD
famd = FAMD(n_components =2, n_iter = 3, random_state = 101)
famd.fit(df)
famd.transform(df)
famd.plot_row_coordinates(df,figsize=(15, 10))
```