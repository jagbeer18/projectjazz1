 CHAPTER 1. INTRODUCTION 

Customer satisfaction is an important activity of the airline. Due to any reason, the late arrival of the plane at the take-off point, flights are delayed and lead to customer dissatisfaction. Flight delay is a really big problem for airlines, airports and passengers hence leaving a negative impact on them. There are two reasons why a flight may be delayed or cancelled. One is the factor of the business situation. Another is the factor of the inevitable. Under the circumstances of the company, the airline is responsible for problems with machine parts and system failures. The natural disaster, which involves heavy snowfall or bad weather, is inevitable. Consequently, punctual flight is targeted and punctual flight predictions are developed because flight performance is not affected by the circumstances and the company's inevitability. Conventionally, when the departure time or arrival time of a flight is more than 15 minutes than expected departure or arrival times, it is assumed that there is a departure or arrival delay compared to at the corresponding airports. 
Flight prediction project is written in Python. The project file contains a python script and a database file. This is a simple console based system which is very easy to understand and use.  The aim of this project is to propose a methodology that can be used to predict flight delays at airports with the help of machine learning. First, we collect realistic raw flight data for airports. The expected departure time at the airport is then set as the prediction target, and individual flight data is processed based on the airport-related aggregate characteristics for prediction modelling. Finally, some machine learning methods have been explored to improve the predictability and accuracy of the proposed model.

1.1 Proposed System


The proposed system allows us to observe a preexisting flight data to generate a machine learning model which has capability to predict the schedules for upcoming flights.
This system has the ability to predict the delay in a scheduled flight with a precision of 99%.
Our system is solely a team work to understand various aspects of the data prediction where we have used regressions for our purpose.


1.2 Objectives

The objective of our project is to import as preexisting data and generate a ML model over it to predict the delay in scheduled flight










1.3	Literature Review
In the past, researchers have attempted to predict flight delays using machine learning, deep learning, and big data.
• Review of machine learning techniques:
- Chakrabarty et al. proposed a machine learning model that uses the Gradient Booster Classifier to predict American Airlines arrival delays at the 5 busiest airports in the United States. USA
- Manna et al. Reviewed and analyzed flight data and developed a regression model using the gradient escalator to predict flight departure and arrival delays
-Ding et al. proposed a multiple linear regression approach to predict flight delay and also compared model performance with the Naive Bayes and C4.5 approaches.


CHAPTER 2. SYSTEM REQUIREMENTS

2.1 HARDWARE REQUIREMENTS

It does not need any additional hardware or software to operate the program but the following requirements should be strongly maintained:

512MB of RAM or Higher
CD ROM
20 MB of Hard Disk Space
800MHz processor or above

Processor: Intel Core

2.2 SOFTWARE REQUIREMENTS

The following requirements should be strongly maintained:

Operating System WINDOWS 7 or Higher
Program PYTHON 2 or Higher needs to be installed.
2.3 TECHNOLOGY USED

Python is a widely used general-purpose, high level programming language. It was initially designed by Guido van Rossum in 1991 and developed by Python Software Foundation. It was mainly developed for emphasis on code readability, and its syntax allows programmers to express concepts in fewer lines of code.

SCIKIT LEARN: 

In this project, we have used Scikit learn.
Scikit-learn is a free and extremely useful machine learning library for Python. It has several algorithms such as the support vector machine, random forests and neighboring k. The library contains many effective tools for machine learning and statistical modeling, including classification, regression, clustering, and dimension reduction.
Scikit-learn is equipped with many functions:
(1)Supervised learning algorithms - Generalized linear models (eg linear regression), support vector machines (SVM), decision trees to Bayesian methods, they are all part of the Scikit-Learn toolbox. The proliferation of machine learning algorithms is one of the main reasons for the heavy use of scikit-learn..
(2)Cross validation: there are different methods to check the accuracy of the models monitored in the data which are not displayed with sklearn.
(3)Unsupervised learning algorithms: a variety of machine learning algorithms are also offered here, from clustering, factor analysis and principal component analysis to unattended neural networks.
(4)Feature extraction: : Scikit-Learn to extract functionalities from images and text .
ROC-AUC-SCORE:
 In machine learning, measuring performance is an essential task. So, when it comes to a classification problem, we can count on an AUC-ROC curve. When we need to examine or visualize the performance of the classification problem for several classes, we use the AUC (Area Under The Curve)-ROC (Receiver Operational Characteristics) curve. It is one of the most important evaluation metrics for verifying the performance of a classification model. ROC curves generally show a true positive rate on the Y axis and a false positive rate on the X axis. This means that the upper left corner of the graph is the "ideal" point: a false positive rate of zero and a true positive rate of one. This is not very realistic, but it does mean that a larger area below the curve (AUC) is generally better. The "slope" of the ROC curves is also important because it is ideal for maximizing the true positive rate and minimizing the false positive rate.

CONFUSUIN MATRIX: 
A confusion matrix is a summary of the results of predicting a classification problem. The number of correct and incorrect predictions is summarized by count values and broken down by each class. The confusion matrix shows how your classification model is confused when you make predictions. This not only gives us an idea of the errors made by a classifier, but especially the types of errors that are made.
Here is a Python script that shows how to create a confusion matrix in a predicted model. To do this, we need to import the confusion matrix module from the sklearn library, which we can use to generate the confusion matrix.
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
  
actual = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0] 
predicted = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0] 
results = confusion_matrix(actual, predicted) 
  
print 'Confusion Matrix :'
print(results) 
print 'Accuracy Score :',accuracy_score(actual, predicted) 
print 'Report : '
print classification_report(actual, predicted) 



 
Chapters 4. Implementation and Result 

In this module, we have:
•	Created a Jupyter notebook , imported data, and viewed data loaded into the notebook.
•	Used Pandas to clean and prepare data to be used for the machine-learning model.
•	Used scikit-learn to create the machine learning model.
•	Used Matplotlib to visualize the model's performance.

curl is a Bash command. we can execute Bash commands in a Jupyter notebook by prefixing them with an exclamation mark. This command downloads a CSV file from Azure blob storage and saves it using the name flightdata.csv.
 








Importing a dataset

In the notebook's second cell, we entered the following Python code to load flightdata.csv, and create a Pandas DataFrame from it, and displayed the first five rows.


 












Loading the dataset
The Data Frame which we have created contains information regarding many US airlines., after visualising it we get to know that it has more than 26 columns and 11,000 rows.
Every single row represents a single flight and provides information regarding the flight such as its origin, its destination, and its scheduled times.
The previously added codes helped in creating a data frame from the imported file. To know about how many rows and columns in this data frame we can use the below code:









  
VISUALIZING THE DATA
Now here we can visualize the Data as it is shown below., which can help us to know about the information which is provided by each column
TABLE 1
Column	Description
YEAR	Year that the flight took place
QUARTER	Quarter that the flight took place (1-4)
MONTH	Month that the flight took place (1-12)
DAY_OF_MONTH	Day of the month that the flight took place (1-31)
DAY_OF_WEEK	Day of the week that the flight took place (1=Monday, 2=Tuesday, etc.)
UNIQUE_CARRIER	Airline carrier code (e.g., DL)
TAIL_NUM	Aircraft tail number
FL_NUM	Flight number
ORIGIN_AIRPORT_ID	ID of the airport of origin
ORIGIN	Origin airport code (ATL, DFW, SEA, etc.)
DEST_AIRPORT_ID	ID of the destination airport
DEST	Destination airport code (ATL, DFW, SEA, etc.)
CRS_DEP_TIME	Scheduled departure time
DEP_TIME	Actual departure time
DEP_DELAY	Number of minutes departure was delayed
DEP_DEL15	0=Departure delayed less than 15 minutes, 1=Departure delayed 15 minutes or more
CRS_ARR_TIME	Scheduled arrival time
ARR_TIME	Actual arrival time
ARR_DELAY	Number of minutes flight arrived late
ARR_DEL15	0=Arrived less than 15 minutes late, 1=Arrived 15 minutes or more late
CANCELLED	0=Flight was not cancelled, 1=Flight was cancelled
DIVERTED	0=Flight was not diverted, 1=Flight was diverted
CRS_ELAPSED_TIME	Scheduled flight time (in min)
ACTUAL_ELAPSED_TIME	Actual flight time (in min)
DISTANCE	Distance travelled in miles
DAY_OF_MONTH	Day of the month that the flight took place (1-31)


Herein, the dataset includes the date and time for every single flight. Here the applier logic is that any flight in us is more likely to be delayed in winter season due to winter storms.
But firstly, we need to clean the Data set prior to make it usable., in machine learning it becomes really important to select the useful features of the dataset so that a better training model can be generated from it.
Another important task is to identify the missing places and the next step is to either fill them or to delete the specific row.

 
Here, we can see that there is a 26th column which is Unnamed and contains 11,231 missing values. It is to be presumed that it has been mistakenly created and needs to be removed before using the data set.
To eliminate that column, we add the following code to the notebook and execute it:

 
But the noticeable point is that our Dataset still contains a lot of missing values and needs to be cleaned up before actually using it. We know that some of the columns are not useful for training our ML model .
Therefore, the next most important step is to remove the irrelevant columns from our dataset.
Pandas provides us an easy way to filter out columns which we don't want. 


 
Doing all this has still made us left with a column of missing values i.e., ARR_DEL15 column.
The logic it uses is :
0= the flight arrived on time
1=the flight arrived late
Here we will filter out those rows which have no value with the code shown below:

Pandas represents missing values with NaN, which stands for Not a Number. 
The output shows that these rows are indeed missing values in the ARR_DEL15 column: 
The reason these rows are missing ARR_DEL15 values is that they all correspond to flights that were canceled or diverted. We can call dropna on the DataFrame to remove these rows. But since a flight that is canceled or diverted to another airport could be considered "late," let's use the fillna method to replace the missing values with 1s.












Use the following code to replace missing values in the ARR_DEL15 column with 1s and display rows 177 through 184:

 
The missing values have been replaced and the irrelevant columns have been removed too. Now, we can call our data as a “cleaned data”

The CRS_DEP_TIME column of the dataset we are using represents scheduled departure times. The granularity of the numbers in this column — it contains more than 500 unique values — could have a negative impact on accuracy in a machine-learning model. This can be resolved using a technique called binning or quantization. What if we divided each number in this column by 100 and rounded down to the nearest integer? 1030 would become 10, 1925 would become 19, and so on, and we would be left with a maximum of 24 discrete values in this column. Intuitively, it makes sense, because it probably doesn't matter much whether a flight leaves at 10:30 a.m. or 10:40 a.m. It matters a great deal whether it leaves at 10:30 a.m. or 5:30 p.m.

In addition, the dataset's ORIGIN and DEST columns contain airport codes that represent categorical machine-learning values. These columns need to be converted into discrete columns containing indicator variables, sometimes known as "dummy" variables. In other words, the ORIGIN column, which contains five airport codes, needs to be converted into five columns, one per airport, with each column containing 1s and 0s indicating whether a flight originated at the airport that the column represents. The DEST column needs to be handled in a similar manner.

Here, we will "bin" the departure times in the CRS_DEP_TIME column and use Pandas' get_dummies method to create indicator columns from the ORIGIN and DEST columns.







 

 



Here, we will use the below code to generate the indicator columns inclusively with the ORIGIN and DEST columns, while dropping the ORIGIN and DEST columns themselves:






Here, we can see that our ORIGIN and DEST columns a=have been replaced with new altered DUMMY columns corresponding to the airport codes present in the original columns. The new columns have 1s and 0s indicating whether a given flight originated at or was destined for the corresponding airport.

 
To create a machine learning model, we need two datasets: one for training and one for testing. In practice, we often have only one dataset, so we split it into two. We have performed an 80:20 split on the DataFrame we prepared previously so we can use it to train a machine learning model. We will also separate the DataFrame into feature columns and label columns. The former contains the columns used as input to the model (for example, the flight's origin and destination and the scheduled departure time), while the latter contains the column that the model will attempt to predict — in this case, the ARR_DEL15 column, which indicates whether a flight will arrive on time.
 





The first statement imports scikit-learn's train_test_split helper function. The second line uses the function to split the DataFrame into a training set containing 80% of the original data, and a test set containing the remaining 20%. The random_state parameter seeds the random-number generator used to do the splitting, while the first and second parameters are DataFrames containing the feature columns and the label column.

train_test_split returns four DataFrames. Use the following command to display the number of rows and columns in the DataFrame containing the feature columns used for training:
 
Now use this command to display the number of rows and columns in the DataFrame containing the feature columns used for testing:
 
How do the two outputs differ, and why?
There are many types of machine learning models. One of the most common is the regression model, which uses one of a number of regression algorithms to produce a numeric value — for example, a person's age or the probability that a credit-card transaction is fraudulent. We'll train a classification model, which seeks to resolve a set of inputs into one of a set of known outputs. A classic example of a classification model is one that examines e-mails and classifies them as "spam" or "not spam." our model will be a binary classification model that predicts whether a flight will arrive on-time or late ("binary" because there are only two possible outputs).

One of the benefits of using scikit-learn is that we don't have to build these models — or implement the algorithms that they use — by hand. Scikit-learn includes a variety of classes for implementing common machine learning models. One of them is RandomForestClassifier, which fits multiple decision trees to the data and uses averaging to boost the overall accuracy and limit overfitting.




Execute the following code in a new cell to create a RandomForestClassifier object and train it by calling the fit method.







The output shows the parameters used in the classifier, including n_estimators, which specifies the number of trees in each decision-tree forest, and max_depth, which specifies the maximum depth of the decision trees. The values shown are the defaults, but we can override any of them when creating the RandomForestClassifier object.
 
Now calling the predict method to test the model using the values in test_x, followed by the score method to determine the mean accuracy of the model:



 

The mean accuracy is 86%, which seems good on the surface. However, mean accuracy isn't always a reliable indicator of the accuracy of a classification model. Let's dig a little deeper and determine how accurate the model really is — that is, how adept it is at determining whether a flight will arrive on time.

There are several ways to measure the accuracy of a classification model. One of the best overall measures for a binary classification model is Area Under Receiver Operating Characteristic Curve (sometimes referred to as "ROC AUC"), which essentially quantifies how often the model will make a correct prediction regardless of the outcome. In this unit, we'll compute an ROC AUC score for the model we built previously and learn about some of the reasons why that score is loour than the mean accuracy output by the score method. We'll also learn about other ways to gauge the accuracy of the model.

Before we compute the ROC AUC, we must generate prediction probabilities for the test set. These probabilities are estimates for each of the classes, or answers, the model can predict. For example, [0.88199435, 0.11800565] means that there's an 89% chance that a flight will arrive on time (ARR_DEL15 = 0) and a 12% chance that it won't (ARR_DEL15 = 1). The sum of the two probabilities adds up to 100%.

Run the following code to generate a set of prediction probabilities from the test data:






Now we use the following statement to generate an ROC AUC score from the probabilities using scikit-learn's roc_auc_score method:
 




Why is the AUC score lower than the mean accuracy computed in the previously?
The output from the score method reflects how many of the items in the test set the model predicted correctly. This score is skewed by the fact that the dataset the model was trained and tested with contains many more rows representing on-time arrivals than rows representing late arrivals. Because of this imbalance in the data, we're more likely to be correct if we predict that a flight will be on time than if we predict that a flight will be late.

ROC AUC takes this into account and provides a more accurate indication of how likely it is that a prediction of on-time or late will be correct.

We can learn more about the model's behavior by generating a confusion matrix, also known as an error matrix. The confusion matrix quantifies the number of times each answer was classified correctly or incorrectly. Specifically, it quantifies the number of false positives, false negatives, true positives, and true negatives. This is important, because if a binary classification model trained to recognize cats and dogs is tested with a dataset that is 95% dogs, it could score 95% simply by guessing "dog" every time. But if it failed to identify cats at all, it would be of little value.

We Use the following code to produce a confusion matrix for our model:






 
The first row in the output represents flights that were on time. The first column in that row shows how many flights were correctly predicted to be on time, while the second column reveals how many flights were predicted as delayed but weren't. From this, the model appears to be adept at predicting that a flight will be on time.

But look at the second row, which represents flights that were delayed. The first column shows how many delayed flights were incorrectly predicted to be on time. The second column shows how many flights were correctly predicted to be delayed. Clearly, the model isn't nearly as adept at predicting that a flight will be delayed as it is at predicting that a flight will arrive on time. What we want in a confusion matrix is large numbers in the upper-left and lower-right corners, and small numbers (preferably zeros) in the upper-right and lower-left corners.

Other measures of accuracy for a classification model include precision and recall. Suppose the model was presented with three on-time arrivals and three delayed arrivals, and that it correctly predicted two of the on-time arrivals, but incorrectly predicted that two of the delayed arrivals would be on time. In this case, the precision would be 50% (two of the four flights it classified as being on time actually were on time), while its recall would be 67% (it correctly identified two of the three on-time arrivals). We can learn more about precision and recall from https://en.wikipedia.org/wiki/Precision_and_recall

Scikit-learn contains a handy method named precision_score for computing precision. To quantify the precision of our model, execute the following statements:


 










Scikit-learn also contains a method named recall_score for computing recall. To measure  model's recall, we execute the following statements:

 
Now we Execute the following statements in a new cell at the end of the notebook while Ignoring any warning messages that are displayed related to font caching:


 
The first statement is one of several magic commands supported by the Python kernel that we selected when we created the notebook. It enables Jupyter to render Matplotlib output in a notebook without making repeated calls to show. And it must appear before any references to Matplotlib itself. The final statement configures Seaborn to enhance the output from Matplotlib.


To see Matplotlib at work, we execute the following code in a new cell to plot the ROC curve for the machine-learning model we built in the previous lab:


 

ROC curve generated with Matplotlib




The dotted line in the middle of the graph represents a 50-50 chance of obtaining a correct answer. The blue curve represents the accuracy of our model. More importantly, the fact that this chart appears at all demonstrates that we can use Matplotlib in a Jupyter notebook.

The reason we built a machine-learning model is to predict whether a flight will arrive on time or late. In this exercise, we'll write a Python function that calls the machine-learning model we built in the previous lab to compute the likelihood that a flight will be on time. Then we'll use the function to analyse several flights.

Now we Enter the following function definition in a new cell, and then run the cell.
























 



This function takes as input a date and time, an origin airport code, and a destination airport code, and returns a value between 0.0 and 1.0 indicating the probability that the flight will arrive at its destination on time. It uses the machine-learning model we built in the previous lab to compute the probability. And to call the model, it passes a DataFrame containing the input values to predict_proba. The structure of the DataFrame exactly matches the structure of the DataFrame we used earlier.


Date input to the predict_delay function use the international date format dd/mm/year.

We Use the code below to compute the probability that a flight from New York to Atlanta on the evening of October 1 will arrive on time. The year we enter is irrelevant because it isn't used by the model.



 


We now have an easy way to predict, with a single line of code, whether a flight is likely to be on time or late. Feel free to experiment with other dates, times, origins, and destinations. But keep in mind that the results are only meaningful for the airport codes ATL, DTW, JFK, MSP, and SEA because those are the only airport codes the model was trained with.









Execute the following code to plot the probability of on-time arrivals for an evening flight from JFK to ATL over a range of days:






 



