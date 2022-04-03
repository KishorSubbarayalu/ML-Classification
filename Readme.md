This Project contains the popular classification techniques in Machine Learning.

Data Set : Wine Quality
Data Source :  Kaggle https://www.kaggle.com/yasserh/wine-quality-dataset

Algorithms used:

	1. Decision Tree Classifier
		a. multi classification
			base model - 65% accuracy
		b. binary classification
			base model - 67.69% accuracy
			hyperparameter tuned model - 73.36% accuracy
		
	2. Random Forest Classifier
		a. multi classification
			base model - 69% accuracy
		b. binary classification
			base model - 79.91% accuracy
			hyperparameter tuned model - 80.35% accuracy
			
	3. Logistic Regression
		a. binary classification
			a1. Model trained on actual classifiers without any transformation or scaling - 78.60%
			a2. Model trained on classifiers with scaling - 78.17%
			a3. Model trained on transformed classifiers (square root) without scaling - 80.35%
			a4. Model trained on transformed classifiers (square root) with scaling - 79.91%
			a5. CV Model trained on transformed classifiers (square root) without scaling - 79.48%
