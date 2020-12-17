Python version == 3.7.8

1. % pip install -r requirements.txt
2. % python classifier.py 

pasting 2 above will reproduce test-out.txt

2 will run the method predict_test_in_test() 

This method returns a prediction (using weights.h5) of the outputs of test-in.txt.

To train the classifier uncomment train() in main. (This will take some time and the threshold value may need to be changed by running predict_30_percent_testing_set() and hard coding in the new threshold value)

To get the ROC curve, confusion matrix and threshold value uncomment predict_30_percent_testing_set() in main.

This method will output the ROC curve and confusion matrix in the project directory as .png files. 

The file REPORT.txt contains the report for this assignment. 
