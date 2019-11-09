# Supervised Learning (using Python)

## Project: Finding Donors for CharityML

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

Install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project.

### Code

Note that the code included in `visuals.py` is to be invoked as an external module in the main image_donors.py file to generate quick visualizations 

### Run

In a terminal or command window, navigate to the top-level project directory `finding_donors/` (that contains this README) and run one of the following commands:

```bash
ipython notebook finding_donors.ipynb
```  
or
```bash
jupyter notebook finding_donors.ipynb
```

This will open the iPython Notebook software and project file in your browser.

### Data

The modified census dataset consists of approximately 32,000 data points, with each datapoint having 13 features. This dataset is a modified version of the dataset published in the paper *"Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",* by Ron Kohavi. You may find this paper [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf), with the original dataset hosted on [UCI](https://archive.ics.uci.edu/ml/datasets/Census+Income).

**Features**
- `age`: Age
- `workclass`: Working Class (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked)
- `education_level`: Level of Education (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool)
- `education-num`: Number of educational years completed
- `marital-status`: Marital status (Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse)
- `occupation`: Work Occupation (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces)
- `relationship`: Relationship Status (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried)
- `race`: Race (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black)
- `sex`: Sex (Female, Male)
- `capital-gain`: Monetary Capital Gains
- `capital-loss`: Monetary Capital Losses
- `hours-per-week`: Average Hours Per Week Worked
- `native-country`: Native Country (United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands)

**Target Variable**
- `income`: Income Class (<=50K, >50K)

### Machine learning model application and evaluation - Sequence of steps
- Create dataframe by importing the dataset
- Normalizing Categorical and Numerical Features
- Data processing to create dataframe with records w/ income >=50K
- Split data w/ Test and train dataset
- Naive Bayes prediction to set a standard
- Evaluating model performance based on prediction, accuracy, recall, fscore
- Apply 3 different supervised models - Decision tree, Random forest classification, ADA Boost and apply to the fitted trainined dataset and then predicting the test data 
- Determining what model to pick up for further optimization based on the model performance scores
- Turning the chosen model using Grid Search technique
- Final model evaluation by comparing the scores before grid searchCV technique to compare the difference between unoptimized and optimized models
- Extracting features of importance (Reduced data set features) and fitting using the chosen model - in this case , Random Forest classification to eventually perform prediction on test data for analysis purpose

### References

https://stackoverflow.com/questions/2595176/which-machine-learning-classifier-to-choose-in-general

https://www.quora.com/How-do-you-choose-a-machine-learning-algorithm

https://www.sciencedirect.com/science/article/pii/S1110866517302141

https://hackernoon.com/choosing-the-right-machine-learning-algorithm-68126944ce1f

https://www.datasciencecentral.com/profiles/blogs/how-to-choose-a-machine-learning-model-some-guidelines

https://machinelearningmastery.com/boosting-and-adaboost-for-machine-learning/##targetText=AdaBoost%20can%20be%20used%20to,decision%20trees%20with%20one%20level.

https://hackernoon.com/under-the-hood-of-adaboost-8eb499d78eab##targetText=AdaBoost%20can%20be%20used%20to,such%20as%20R%20and%20Python.

https://www.geeksforgeeks.org/decision-tree/##targetText=Strengths%20and%20Weakness%20of%20Decision,both%20continuous%20and%20categorical%20variables.
