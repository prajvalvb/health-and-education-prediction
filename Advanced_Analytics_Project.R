# ====================================================
# ADVANCED ANALYTICS PROJECT: CLASSIFICATION & REGRESSION
# Models: Decision Tree, KNN, and Multiple Linear Regression
# Dataset 1: Heart Disease (Classification)
# Dataset 2: Student Performance (Regression)
# Author: Prajwal Bharti
# ====================================================

# -----------------------------------------------------
# Decision tree apporoach
# Dataset: Heart Disease
# Author: Prajwal Bharti
# -----------------------------------------------------


# =========================================
# Data Loading
# =========================================
# Load packages
install.packages('tidyverse')
library(tidyverse)

# Read data 
heart <- read.csv("processed.cleveland.data", header = FALSE)

# Assign column names
colnames(heart) <- c("age", "sex", "cp", "trestbps", "chol", 
                     "fbs", "restecg", "thalach", "exang", 
                     "oldpeak", "slope", "ca", "thal", "target")

# Preview
head(heart)
str(heart)
view(heart)

#Replacing '?' with NA so it doesn't affect my model
heart[heart == "?"] <- NA

#Converting 'ca' and 'thal' columns from char to numeric
heart$ca <- as.numeric(heart$ca)
heart$thal <- as.numeric(heart$thal)

#Removing missing values if any
heart <- na.omit(heart)

#converting target variable to binary variable as 0 and 1
heart$target <- ifelse(heart$target == 0, 0, 1)

#Convert target to binary factor (0 = No disease, 1 = Disease)
heart$target <- factor(heart$target, labels = c("No", "Yes"))

#Preview
str(heart)
summary(heart)

# Check class balance
table(heart$target)

# =========================================
# Data Visualizing
# =========================================
install.packages('ggplot')
library(ggplot2)


#shows how many individuals have heart disease ("Yes") vs. those who do not ("No").
ggplot(heart, aes(x = target)) +
  geom_bar(fill = "skyblue") +
  labs(title = "Heart Disease Distribution", x = "Heart Disease", y = "Count")

#shows age distribution of heart disease patients 
ggplot(heart, aes(x = age, fill = target)) +
  geom_histogram(binwidth = 5, position = "dodge") +
  labs(title = "Age Distribution by Heart Disease", x = "Age", y = "Count")

#shows How age has affect on cholestrol
ggplot(heart, aes(x = age, y = chol, color = target)) +
  geom_point(alpha = 0.7) +
  labs(title = "Cholestrol Vs Age", x = "Age",y = "Cholestrol")

#shows How Max heart rate has affect on heart disease
ggplot(heart, aes(x = target, y = thalach, fill = target)) +
  geom_boxplot() +
  labs(title = "Max Heart Rate vs Heart Disease", x = "Heart Disease", y = "Max Heart Rate")


# =================================================
# Splitting the data into training and testing data
# =================================================
#Load caret package
install.packages('caret')
library(caret)    

#Set seed for reproducibility
set.seed(123)

#Create split index (80% training data)
split <- createDataPartition(heart$target, p = 0.8, list = FALSE)

#Create training and testing datasets
heart_train <- heart[split, ]
heart_test  <- heart[-split, ]


# =================================================
# Building the Decision Tree model
# =================================================
#installing required packages 
install.packages("rpart")       # For decision trees
install.packages("rpart.plot")  # For plotting trees
library(rpart)
library(rpart.plot)

# Train the decision tree model
tree_model <- rpart(target ~ ., data = heart_train, method = "class")

#Visualize the decision tree model
rpart.plot(tree_model, type = 3, extra = 102, fallen.leaves = TRUE)

#Testing the data
tree_predictions <- predict(tree_model, newdata = heart_test, type = "class")
confusionMatrix(tree_predictions, heart_test$target)

# Summary of Decision Tree model findings:
# The model achieved ~81.4% accuracy on the test set.
# Chest pain type (cp) and number of vessels colored (ca) were key predictors.
# The tree showed strong separation for high cp and high ca values indicating heart disease.





#-----------------------------------------------------
#KNN (K Nearest Neighbor) Approach
#Dataset: Heart Disease
#Author: Prajwal Bharti
#-----------------------------------------------------

# =================================================
# Loading the required packages
# =================================================
install.packages("class")
install.packages("e1071")
library(class)    # For knn()
library(e1071)    # Helps confusionMatrix work with kNN

# Min-Max normalization function
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Drop target column
heart_features <- heart[, -which(names(heart) == "target")]

# Normalize features
heart_norm <- as.data.frame(lapply(heart_features, normalize))

# Add target column back
heart_norm$target <- heart$target

# =================================================
# Splitting the data into training and testing data
# =================================================
# Create train/test split
set.seed(123)  # For reproducibility
split <- createDataPartition(heart_norm$target, p = 0.8, list = FALSE)

# Create train and test sets
knn_train <- heart_norm[split, ]
knn_test  <- heart_norm[-split, ]


knn_pred3 <- knn(train = knn_train[, -ncol(knn_train)],
                 test = knn_test[, -ncol(knn_test)],
                 cl = knn_train$target,
                 k = 3)

knn_pred5 <- knn(train = knn_train[, -ncol(knn_train)],
                 test = knn_test[, -ncol(knn_test)],
                 cl = knn_train$target,
                 k = 5)

knn_pred7 <- knn(train = knn_train[, -ncol(knn_train)],
                 test = knn_test[, -ncol(knn_test)],
                 cl = knn_train$target,
                 k = 7)

# Comparing the knn prediction with actual target column
confusionMatrix(knn_pred3, knn_test$target)
confusionMatrix(knn_pred5, knn_test$target)
confusionMatrix(knn_pred7, knn_test$target)

# Summary of kNN model findings:
# Best accuracy was achieved at k = 5 (~81.4%), similar to the Decision Tree.
# The model performed well in predicting "No" (healthy) cases with high sensitivity.
# Slightly lower specificity suggests room for improvement in detecting heart disease cases.





#-----------------------------------------------------
# REGRESSION MODEL: Student Performance
#Author: Prajwal Bharti
#-----------------------------------------------------

# =================================================
# Loading required packages
# =================================================
library(tidyverse)

# Load the student performance data
student <- read.csv("student-mat.csv", sep = ";")

# Preview the data
head(student)
str(student)

# Convert character columns to factors
student <- student %>%
  mutate_if(is.character, as.factor)

# Optional: check again after conversion
str(student)

# =================================================
# Building Regression model
# =================================================
student.lm <- lm(G3 ~ G1 + G2 + studytime, data = student)
summary(student.lm)

# =========================================
# Make a prediction
# For a new student with:
# G1(Grade1) = 13, G2(Grade2) = 16, studytime = 3
# =========================================
new_student <- data.frame(G1 = 13, G2 = 16, studytime = 3)
predict(student.lm, new_student)

# Predict final grade with 95% confidence interval (for mean prediction)
predict(student.lm, new_student, interval = "confidence")

# Predict final grade with 95% prediction interval (for individual prediction)
predict(student.lm, new_student, interval = "predict")


# =========================================
# Data Visualizing
# =========================================

# G1 vs G3
plot(G3 ~ G1, data = student, main = "G1 vs Final Grade (G3)",
     xlab = "First Sem Grade (G1)", ylab = "Final Grade")
abline(lm(G3 ~ G1, data = student), col = "blue")

# G2 vs G3
plot(G3 ~ G2, data = student, main = "G2 vs Final Grade (G3)",
     xlab = "Second Sem Grade (G2)", ylab = "Final Grade")
abline(lm(G3 ~ G2, data = student), col = "green")

# Studytime vs G3
plot(G3 ~ studytime, data = student, main = "Study Time vs Final Grade (G3)",
     xlab = "Study Time (1 = <2hrs, 4 = 10+ hrs)", ylab = "Final Grade")
abline(lm(G3 ~ studytime, data = student), col = "red")


# =========================================
# Build Reduced Model (without G2)
# =========================================
student.lm2 <- lm(G3 ~ G1 + studytime, data = student)
summary(student.lm2)


# =========================================
# Compare R-squared Values
# =========================================
# Full model R-squared
summary(student.lm)$r.squared

# Reduced model R-squared
summary(student.lm2)$r.squared

# =========================================
# Compare Predictions for Both Models
# =========================================
predict(student.lm, new_student, interval = "confidence")
predict(student.lm2, new_student, interval = "confidence")

# Summary of Regression Model findings:
# G1 and G2 were highly significant predictors of the final grade (G3).
# Adjusted R-squared was ~0.82, indicating strong model fit.
# RMSE was relatively low, showing good predictive accuracy.
# Study time had a smaller but positive impact on grades.

