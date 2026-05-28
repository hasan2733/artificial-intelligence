# Accuracy Evaluation in Machine Learning
*A Comprehensive Guide to Evaluation Metrics, Error Functions & Techniques*
 
---
 
## 1. Introduction to Accuracy Evaluation
 
In Machine Learning and Artificial Intelligence, building a model is only half the work — the other half is knowing how well it performs. Accuracy evaluation is the process of measuring how closely a model's predictions match the actual (true) values.
 
Without proper evaluation, you might deploy a model that looks good on paper but fails in the real world. For example, a spam filter might correctly classify 95% of emails — but if it keeps missing actual spam, its recall is poor. Choosing the right metric is therefore just as important as building the model itself.
 
> **Key Insight:** No single metric tells the full story. Use multiple metrics together for a complete picture of model performance.
 
---
 
## 2. Classification Metrics
 
Classification metrics are used when the model predicts categories or labels — such as spam vs. not spam, disease vs. no disease, or fraud vs. legitimate transaction.
 
---
 
### 2.1 Accuracy Score
 
Accuracy is the most basic metric. It tells you the proportion of all predictions — both positive and negative — that were correct.
 
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)  ×  100
```
 
Where:
- **TP** = True Positives
- **TN** = True Negatives
- **FP** = False Positives
- **FN** = False Negatives
#### Real-World Example
 
You built a model to detect fraudulent bank transactions. You test it on 1,000 transactions:
 
- 800 legitimate transactions → model correctly identifies all 800 (TN = 800)
- 150 fraud transactions → model correctly flags 140 (TP = 140)
- 10 legitimate transactions are wrongly flagged as fraud (FP = 10)
- 10 fraud transactions are missed by the model (FN = 10)
```
Accuracy = (140 + 800) / 1000 × 100 = 94%
```
 
> **When to use:** When the dataset is balanced (roughly equal classes) and all types of errors carry equal cost.
 
> **Warning:** Accuracy is misleading on imbalanced datasets. If 990 out of 1,000 transactions are legitimate, a model that always predicts "legitimate" gets 99% accuracy — but catches zero fraud.
 
---
 
### 2.2 Confusion Matrix
 
A confusion matrix is a table that breaks down a model's predictions into four categories, giving a detailed view of where the model succeeds and where it fails.
 
|  | **Predicted: Positive** | **Predicted: Negative** |
|---|---|---|
| **Actual: Positive** | True Positive (TP) — Correctly predicted positive | False Negative (FN) — Missed positive — Type II Error |
| **Actual: Negative** | False Positive (FP) — Wrongly predicted positive — Type I Error | True Negative (TN) — Correctly predicted negative |
 
#### Medical Diagnosis Example
 
A model diagnoses patients for a disease. It tests 100 patients:
 
- 24 patients actually have the disease and are correctly diagnosed (TP = 24)
- 65 healthy patients are correctly told they are healthy (TN = 65)
- 8 healthy patients are wrongly told they have the disease (FP = 8)
- 3 sick patients are missed and told they are healthy (FN = 3)
```
Accuracy = (24 + 65) / 100 = 89%   — but 3 sick patients went undetected!
```
 
This shows why accuracy alone is not enough — the confusion matrix reveals the full picture.
 
---
 
### 2.3 Precision
 
Precision answers: *"Of all the cases the model predicted as positive, how many were actually positive?"* It measures how trustworthy a positive prediction is.
 
```
Precision = TP / (TP + FP)
```
 
#### Email Spam Filter Example
 
Your spam filter flags 50 emails as spam. Out of those 50:
 
- 45 are genuinely spam (TP = 45)
- 5 are legitimate emails that were wrongly flagged (FP = 5)
```
Precision = 45 / (45 + 5) = 0.90 = 90%
```
 
This means 90% of the time, when your filter says "spam", it is actually spam. The other 10% are false alarms that annoy the user.
 
> **When precision matters most:** When false positives are costly. Example: in a legal system, wrongly convicting an innocent person (FP) is very serious, so you want high precision.
 
---
 
### 2.4 Recall (Sensitivity / True Positive Rate)
 
Recall answers: *"Of all actual positive cases, how many did the model correctly identify?"* It measures the model's ability to find all relevant cases.
 
```
Recall = TP / (TP + FN)
```
 
#### Cancer Screening Example
 
A hospital tests 200 patients for cancer. 50 actually have cancer. The model identifies:
 
- 45 of the 50 cancer patients correctly (TP = 45)
- 5 cancer patients are missed (FN = 5)
```
Recall = 45 / (45 + 5) = 0.90 = 90%
```
 
The model catches 90% of cancer cases. However, 5 patients with cancer were missed — these are the most dangerous errors in healthcare.
 
> **When recall matters most:** When false negatives are costly. Example: missing a cancer diagnosis (FN) is far worse than a false alarm, so you want high recall.
 
#### The Precision–Recall Tradeoff
 
In most models, increasing precision tends to decrease recall, and vice versa. This is called the precision-recall tradeoff.
 
| Scenario | Prioritize | Reason |
|---|---|---|
| Cancer detection | Recall | Missing a sick patient is dangerous |
| Spam filtering | Precision | Blocking real emails frustrates users |
| Fraud detection | Recall | Missing fraud causes financial loss |
| Legal evidence | Precision | Wrongly accusing someone is unjust |
| Recommendation system | Precision | Irrelevant suggestions frustrate users |
 
---
 
### 2.5 F1 Score
 
The F1 Score is the harmonic mean of Precision and Recall. It gives a single balanced number that penalizes extreme imbalance between the two — unlike a simple average.
 
```
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
```
 
#### Why Harmonic Mean?
 
A regular average would reward high values on either side. For example, Precision = 1.0 and Recall = 0.0 would give an average of 0.5, which sounds decent — but the model is catching nothing! The harmonic mean is 0.0 in this case, correctly penalizing the imbalance.
 
#### Example — Document Retrieval System
 
A search engine retrieves 10 documents for a query. Out of those 10:
 
- 7 are relevant (TP = 7)
- 3 are irrelevant (FP = 3)
- There were 9 total relevant documents in the database, so 2 were missed (FN = 2)
```
Precision = 7 / (7+3) = 0.70
Recall    = 7 / (7+2) = 0.78
F1 Score  = 2 × (0.70 × 0.78) / (0.70 + 0.78) = 0.737 ≈ 74%
```
 
> **When to use F1:** When you want a single score that balances both precision and recall, especially on imbalanced datasets.
 
---
 
### 2.6 F2 Score
 
The F2 Score is a variation of the F1 Score that gives more weight to Recall than Precision. It is used when missing a positive case (false negative) is more harmful than a false alarm.
 
```
F2 Score = (1 + 2²) × (Precision × Recall) / (2² × Precision + Recall)
```
 
#### Medical Testing Example
 
In a test for a rare infectious disease, missing an infected person (FN) means they could spread the disease further. A false positive only means extra testing — manageable. Here, F2 Score is preferred.
 
```
If Precision = 0.60, Recall = 0.90:
F1 = 2×(0.60×0.90)/(0.60+0.90)       = 0.72
F2 = 5×(0.60×0.90)/(4×0.60+0.90)     = 0.833
```
 
The F2 score is higher, rewarding the strong recall.
 
---
 
### 2.7 Matthews Correlation Coefficient (MCC)
 
MCC is considered one of the best metrics for binary classification, especially on imbalanced datasets. It takes into account all four cells of the confusion matrix and returns a value between -1 and +1.
 
```
MCC = (TP×TN − FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]
```
 
| MCC Value | Interpretation |
|---|---|
| +1.0 | Perfect prediction |
| +0.5 to +1.0 | Strong prediction |
| 0 | No better than random guessing |
| -0.5 to 0 | Weak / poor model |
| -1.0 | Complete reversal of predictions |
 
#### Fraud Detection Example
 
In a dataset of 10,000 transactions, only 100 are fraud (1%). A naive model predicts "no fraud" for everything:
 
- Accuracy = 99% *(misleadingly high!)*
- MCC = 0 *(correctly reveals the model is useless — it catches no fraud)*
> **Best use case:** Imbalanced binary classification where both classes matter.
 
---
 
### 2.8 ROC Curve and AUC
 
The ROC (Receiver Operating Characteristic) Curve is a graph that shows the performance of a classification model at all possible classification thresholds. The AUC (Area Under the Curve) summarizes this graph into a single number.
 
| Term | Formula | Meaning |
|---|---|---|
| True Positive Rate (TPR) | TP / (TP + FN) | Also called Recall/Sensitivity |
| False Positive Rate (FPR) | FP / (FP + TN) | Rate of false alarms |
| AUC | Area under ROC curve | Overall model discrimination ability |
 
| AUC Value | Interpretation |
|---|---|
| 1.0 | Perfect model |
| 0.9 – 1.0 | Excellent |
| 0.8 – 0.9 | Good |
| 0.7 – 0.8 | Fair |
| 0.5 – 0.7 | Poor |
| 0.5 | Random guessing (no skill) |
| < 0.5 | Worse than random |
 
#### Disease Screening Example
 
Two models predict heart disease at threshold 0.5:
 
- **Model A:** TPR = 0.85, FPR = 0.15 → AUC = 0.92
- **Model B:** TPR = 0.70, FPR = 0.30 → AUC = 0.76
Model A is better at distinguishing sick from healthy patients across all thresholds.
 
> **Key advantage:** AUC-ROC is threshold-independent — it evaluates performance across all possible decision boundaries, making it ideal for comparing models.
 
---
 
### 2.9 Cohen's Kappa
 
Cohen's Kappa measures agreement between two classifiers (or a model and a human annotator), correcting for the agreement that could occur purely by chance.
 
```
κ = (Po − Pe) / (1 − Pe)
```
 
Where **Po** = observed agreement and **Pe** = expected agreement by chance.
 
| Kappa Value | Level of Agreement |
|---|---|
| < 0 | Poor (worse than chance) |
| 0.01 – 0.20 | Slight |
| 0.21 – 0.40 | Fair |
| 0.41 – 0.60 | Moderate |
| 0.61 – 0.80 | Substantial |
| 0.81 – 1.00 | Almost perfect |
 
#### Medical Annotation Example
 
Two radiologists examine 100 X-rays. They agree on 80 cases. By chance alone, they would agree on 50.
 
```
κ = (0.80 − 0.50) / (1 − 0.50) = 0.60   →  Substantial agreement
```
 
---
 
### 2.10 Null Error Rate
 
The Null Error Rate is the error rate you would get if you always predicted the most common class. It serves as a minimum performance baseline — your model must beat it to be considered useful.
 
```
Null Error Rate = 1 − (Count of majority class / Total samples)
```
 
#### Student Pass/Fail Example
 
In a class of 100 students, 75 passed (Class A) and 25 failed (Class B). If a model always predicts "Passed":
 
```
Null Error Rate = 1 − (75/100) = 0.25 = 25%
A useful model must have an error rate LOWER than 25%.
```
 
> **Use case:** Always calculate this before evaluating your model. If your model's accuracy is only slightly above the null error rate, it may just be guessing the majority class.
 
---
 
## 3. Regression Metrics
 
Regression metrics are used when the model predicts continuous numerical values — such as house prices, temperatures, stock values, or exam scores.
 
---
 
### 3.1 Mean Absolute Error (MAE)
 
MAE measures the average magnitude of errors without caring about direction (over or under). It treats all errors equally.
 
```
MAE = (1/n) × Σ |Actual_i − Predicted_i|
```
 
#### Weather Forecast Example
 
| Day | Actual (°C) | Predicted (°C) | Absolute Error |
|---|---|---|---|
| Monday | 22 | 20 | 2 |
| Tuesday | 25 | 27 | 2 |
| Wednesday | 18 | 15 | 3 |
| Thursday | 30 | 29 | 1 |
| Friday | 20 | 23 | 3 |
 
```
MAE = (2 + 2 + 3 + 1 + 3) / 5 = 2.2°C
```
 
On average, the model's temperature predictions are off by 2.2°C — easy to interpret.
 
> **When to use MAE:** When all errors should be treated equally and outliers should not dominate the metric.
 
---
 
### 3.2 Mean Squared Error (MSE)
 
MSE squares each error before averaging. Larger errors are penalized disproportionately more, making the model sensitive to outliers.
 
```
MSE = (1/n) × Σ (Actual_i − Predicted_i)²
```
 
#### Student Score Prediction Example
 
| Student | Actual Score | Predicted Score | Error | Error² |
|---|---|---|---|---|
| Alice | 85 | 80 | 5 | 25 |
| Bob | 90 | 85 | 5 | 25 |
| Charlie | 95 | 100 | -5 | 25 |
 
```
MSE = (25 + 25 + 25) / 3 = 25
```
 
#### Outlier Sensitivity Demo
 
If Charlie's prediction was 150 instead of 100:
 
```
Error = 95 − 150 = −55   →   Error² = 3025
New MSE = (25 + 25 + 3025) / 3 = 1025   (vs. 25 before!)
```
 
One outlier exploded the MSE from 25 to 1025.
 
> **Low MSE** = predictions are close to actual values. **High MSE** = large errors.
 
---
 
### 3.3 Root Mean Squared Error (RMSE)
 
RMSE is the square root of MSE. It brings the error back to the same units as the original data, making it far more interpretable than MSE while retaining outlier sensitivity.
 
```
RMSE = √MSE  =  √[ (1/n) × Σ (Actual_i − Predicted_i)² ]
```
 
#### House Price Prediction Example
 
A model predicts house prices. The MSE across the test set is 100,000,000:
 
```
RMSE = √100,000,000 = $10,000
```
 
| Metric | Value | Interpretation |
|---|---|---|
| MSE | 100,000,000 | Hard to interpret (units are USD²) |
| RMSE | $10,000 | Easy to interpret (same unit as price) |
 
#### Limitations of RMSE
 
- Sensitive to outliers — one large error dominates the result
- Cannot compare across datasets with different scales
- Does not show direction — cannot tell if model is over- or under-predicting
---
 
### 3.4 Mean Absolute Percentage Error (MAPE)
 
MAPE expresses error as a percentage of the actual value, making it easy to understand and compare across different scales and datasets.
 
```
MAPE = (1/n) × Σ | (Actual_i − Predicted_i) / Actual_i | × 100
```
 
#### Retail Sales Forecasting Example
 
| Product | Actual Sales | Predicted Sales | Absolute % Error |
|---|---|---|---|
| Shirts | 100 | 90 | \|100−90\|/100 × 100 = 10% |
| Pants | 120 | 110 | \|120−110\|/120 × 100 = 8.33% |
| Shoes | 80 | 70 | \|80−70\|/80 × 100 = 12.5% |
| Bags | 150 | 160 | \|150−160\|/150 × 100 = 6.67% |
 
```
MAPE = (10 + 8.33 + 12.5 + 6.67) / 4 = 9.38%
On average, the predictions are 9.38% off from actual sales.
```
 
#### Limitations of MAPE
 
- Cannot handle zero actual values — division by zero breaks the formula
- Biased toward smaller values: predicting 2 instead of 1 = 100% error; predicting 102 instead of 101 = 0.99% error
- Overestimation and underestimation of the same magnitude give different MAPE values (asymmetric)
---
 
### 3.5 Adjusted R-Squared
 
R-Squared (R²) measures how much of the variance in the target variable is explained by the model. Adjusted R² improves on R² by penalizing models that use unnecessary predictors, helping to prevent overfitting.
 
```
Adjusted R² = 1 − (1−R²) × [(n−1) / (n−k−1)]
```
 
Where **n** = number of data points, **k** = number of predictors (features).
 
#### Housing Price Regression Example
 
- **Model A** uses 3 relevant features (size, location, age): R² = 0.85, Adjusted R² = 0.84
- **Model B** adds 10 random irrelevant features: R² = 0.87, Adjusted R² = 0.79
Model B's R² is higher — but Adjusted R² reveals it is actually worse because the extra features add noise, not value.
 
> **Rule:** If adding a new feature truly improves the model, Adjusted R² increases. If the feature is irrelevant, Adjusted R² decreases even if R² goes up.
 
---
 
### 3.6 Log-Cosh Loss
 
Log-Cosh Loss is a regression loss function that behaves like MSE for small errors (smooth and sensitive) and like MAE for large errors (robust to outliers). It sits between the two extremes.
 
```
Log-Cosh Loss = Σ log(cosh(y_true − y_pred))
```
 
#### Real Estate Pricing Example
 
Most house price predictions are within $5,000 of actual. But a few luxury homes are off by $200,000.
 
- **Using MSE:** Those 3 luxury home errors dominate the entire loss, distorting the model
- **Using Log-Cosh:** Small errors penalized like MSE; large errors penalized like MAE — model stays balanced
> **Use case:** Neural network regression tasks where you want smooth gradients AND robustness to occasional large errors.
 
---
 
### 3.7 Huber Loss
 
Huber Loss is another hybrid between MAE and MSE, controlled by a threshold parameter delta (δ). For errors smaller than δ it behaves like MSE; for errors larger than δ it behaves like MAE.
 
```
Huber(y, ŷ) = { 0.5(y−ŷ)²          if |y−ŷ| ≤ δ
              { δ|y−ŷ| − 0.5δ²    if |y−ŷ| > δ
```
 
#### Sensor Data Regression Example
 
An IoT sensor occasionally malfunctions and produces extreme readings. With δ = 1.0:
 
- Normal errors (< 1.0): treated like MSE — model learns precisely
- Outlier errors (> 1.0): treated like MAE — outlier influence is capped
> **Advantage over Log-Cosh:** δ is tunable — you can control exactly where "outlier" behavior kicks in.
 
---
 
### 3.8 Cross-Entropy Loss
 
Cross-Entropy is a loss function for classification models that output probabilities. It measures how different the predicted probability distribution is from the actual distribution.
 
```
Cross-Entropy = −Σ [ actual_i × log(predicted_i) ]
```
 
#### Image Classification Example
 
A neural network classifies images into Cat, Dog, or Bird. For one image of a cat:
 
| Class | Actual (One-hot) | Predicted Probability | Contribution |
|---|---|---|---|
| Cat | 1 | 0.85 | −1 × log(0.85) = 0.163 |
| Dog | 0 | 0.10 | −0 × log(0.10) = 0 |
| Bird | 0 | 0.05 | −0 × log(0.05) = 0 |
 
```
Cross-Entropy = 0.163 + 0 + 0 = 0.163   (lower = better)
```
 
#### Why Not Just Use Accuracy for Training?
 
Accuracy is either 0 or 1 — it has no gradient for a model to learn from. Cross-Entropy produces a smooth curve: the more confident and correct the prediction, the lower the loss. This gradient is what allows backpropagation to work.
 
#### Limitations
 
- Requires one-hot encoded targets — increases dimensionality
- Sensitive to class imbalance — rare classes barely contribute to the loss
- Sensitive to overconfident wrong predictions — predicting 0.001 probability for the true class gives a very high loss
---
 
## 4. Types of Accuracy
 
---
 
### 4.1 Point Accuracy
 
Point Accuracy measures the percentage of predictions that are exactly correct — with zero deviation.
 
```
Point Accuracy = (Exactly Correct Predictions / Total Predictions) × 100
```
 
#### Temperature Forecast Example
 
A weather model predicts temperature for 5 days. The model was exactly right on Days 1, 3, and 5.
 
```
Point Accuracy = 3/5 × 100 = 60%
```
 
---
 
### 4.2 Percentage of Scale Range
 
This metric compares the predicted value to the full range of possible values. Useful for comparing predictions across datasets with different value ranges.
 
```
% of Scale Range = (Predicted − Minimum) / (Maximum − Minimum) × 100
```
 
#### Electric Vehicle Battery Example
 
A battery charge prediction model predicts 80% charge for a battery ranging from 0% to 100%:
 
```
% of Scale Range = (80 − 0) / (100 − 0) × 100 = 80%
```
 
The prediction falls at the 80th percentile of the possible range — useful for normalizing across different sensor scales.
 
---
 
### 4.3 Percentage of True Value
 
This measures the error as a percentage of the actual value, giving a relative sense of how large the error is.
 
```
% of True Value = (|Actual − Predicted| / Actual) × 100
```
 
#### Product Price Prediction Example
 
A pricing model predicts $90 when the actual price is $100:
 
```
Absolute Error = |100 − 90| = 10
% of True Value = 10 / 100 × 100 = 10%
```
 
The prediction is off by 10% of the actual value — immediately meaningful regardless of price scale.
 
---
 
## 5. Additional Important Concepts
 
---
 
### 5.1 Cross-Validation
 
Cross-validation tests how well a model generalizes to new data. You split the data into k equal "folds" and repeat training/testing k times, using each fold as the test set once.
 
#### k-Fold Cross-Validation Process (k = 5)
 
| Round | Training Folds | Test Fold | What You Get |
|---|---|---|---|
| 1 | Folds 2,3,4,5 | Fold 1 | Accuracy on Fold 1 |
| 2 | Folds 1,3,4,5 | Fold 2 | Accuracy on Fold 2 |
| 3 | Folds 1,2,4,5 | Fold 3 | Accuracy on Fold 3 |
| 4 | Folds 1,2,3,5 | Fold 4 | Accuracy on Fold 4 |
| 5 | Folds 1,2,3,4 | Fold 5 | Accuracy on Fold 5 |
 
```
Final Score = Average of all 5 test accuracies
Example: (0.88 + 0.91 + 0.87 + 0.90 + 0.89) / 5 = 0.89 = 89%
```
 
> **Why use it:** A model that scores 95% on one test split might be lucky. Cross-validation gives a more reliable and unbiased estimate of real-world performance.
 
---
 
### 5.2 Bias–Variance Tradeoff
 
Every machine learning model has two sources of error: bias and variance. Understanding the tradeoff is fundamental to improving model performance.
 
| Concept | Definition | Symptom | Fix |
|---|---|---|---|
| High Bias | Model too simple, misses patterns | Underfitting — poor on both train and test | Add features, use more complex model |
| High Variance | Model too complex, memorizes noise | Overfitting — great on train, poor on test | Regularization, more data, simpler model |
| Balanced | Captures true patterns, generalizes well | Good performance on both sets | Cross-validation, tuning |
 
#### Analogy
 
> A student who memorizes all past exam answers scores 100% on practice tests but fails the actual exam — **high variance (overfitting)**.
>
> A student who barely studies and guesses everything scores 50% everywhere — **high bias (underfitting)**.
 
---
 
### 5.3 Specificity and NPV (Often Missed)
 
Two important confusion matrix metrics that are frequently overlooked:
 
| Metric | Formula | Meaning | Example Use |
|---|---|---|---|
| Specificity (TNR) | TN / (TN + FP) | Of all true negatives, how many were correctly identified | Avoid over-treating healthy patients |
| Negative Predictive Value (NPV) | TN / (TN + FN) | When the model says negative, how often is it truly negative | Clearing patients as disease-free |
 
#### Complete Metrics Reference from Confusion Matrix
 
| Metric | Formula | Also Known As |
|---|---|---|
| Accuracy | (TP+TN) / Total | Overall correctness |
| Precision | TP / (TP+FP) | Positive Predictive Value (PPV) |
| Recall | TP / (TP+FN) | Sensitivity, True Positive Rate (TPR) |
| Specificity | TN / (TN+FP) | True Negative Rate (TNR) |
| F1 Score | 2×(P×R)/(P+R) | Harmonic mean of Precision and Recall |
| NPV | TN / (TN+FN) | Negative Predictive Value |
| FPR | FP / (FP+TN) | False Alarm Rate, 1 − Specificity |
| FNR | FN / (FN+TP) | Miss Rate, 1 − Recall |
 
---
 
### 5.4 Model Calibration
 
A well-calibrated model is one where the predicted probabilities actually match the true frequencies of outcomes.
 
#### Example
 
A model predicts "90% chance of rain" on 100 different days. If well-calibrated, it should actually rain on about 90 of those days. If it only rains 60 times, the model is overconfident and poorly calibrated.
 
| Predicted Probability | Well-Calibrated Model | Overconfident Model |
|---|---|---|
| 90% | Actual rate ≈ 90% | Actual rate ≈ 60% |
| 70% | Actual rate ≈ 70% | Actual rate ≈ 45% |
| 50% | Actual rate ≈ 50% | Actual rate ≈ 35% |
 
> **Fix:** Apply Platt Scaling or Isotonic Regression to recalibrate an overconfident model.
 
---
 
## 6. Choosing the Right Metric — Quick Reference
 
| Task Type | Dataset | Recommended Metric(s) | Reason |
|---|---|---|---|
| Binary Classification | Balanced | Accuracy, F1 | All errors equally important |
| Binary Classification | Imbalanced | F1, MCC, AUC-ROC | Accuracy misleads on rare classes |
| Multi-class Classification | Any | Macro/Micro F1, MCC | Average across all classes |
| Regression | No outliers | MAE, R² | Simple, interpretable |
| Regression | With outliers | Huber, Log-Cosh, RMSE | Outlier-robust |
| Regression | Cross-dataset | MAPE, % of True Value | Scale-independent |
| Probability output | Any | Cross-Entropy, AUC | Evaluates confidence |
| Annotation agreement | Any | Cohen's Kappa | Controls for chance agreement |
| Model comparison | Any | AUC-ROC, Adjusted R² | Threshold/feature-independent |
 
---
 
                                           *End of Document*
