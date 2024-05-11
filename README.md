# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:

To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm

Step1. Start the Program.

Step 2. Import the necessary packages.

Step 3. Read the given csv file and display the few contents of the data.

Step 4. Assign the features for x and y respectively.

Step 5. Split the x and y sets into train and test sets.

Step 6. Convert the Alphabetical data to numeric using CountVectorizer.

Step 7. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

Step 8. Find the accuracy of the model.

Step 9. Close the Program.

## Program:

Program to implement the SVM For Spam Mail Detection.

#### Developed by: YASHWANTH RAJA DURAI

#### RegisterNumber: 212222040184

```python
#import packages
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("spam.csv",encoding="latin-1")
df.head()

#checking the data information and null presence
df.info()
df.isnull().sum()

#assigning x and y array
x=df["v1"].values
y=df["v2"].values

#splitting the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#converting to numerical count in train and test set
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

#predicting y- i.e detecting spam
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

#checking the accuracy of the model
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

### Dataset:

![o1](https://github.com/ATHMAJ03/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118753139/1c52f7b0-9edf-4d3e-af8c-ea891fa84c75)

### Dataset information:

![o2](https://github.com/ATHMAJ03/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118753139/e17e41a6-cef1-4568-9e24-6924d7121771)

![o3](https://github.com/ATHMAJ03/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118753139/3b1347c8-2168-4e05-ba72-68b833be85bf)

### Detected spam:

![o4](https://github.com/ATHMAJ03/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118753139/b1b57d3e-fd05-43e1-a270-42d31e6066e8)


### Accuracy score of the model:

![o5](https://github.com/ATHMAJ03/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118753139/aedc79a5-6359-48ef-a7a9-989c30c4438d)


## Result:

Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
