#An application that estimates customer churn rate using ABC Multi State Bank's dataset.


#CustomerId → Müşteri kimliği (Gereksiz, çıkarılabilir)
#Surname → Soyadı (Analiz için gereksiz, çıkarılabilir)
#CreditScore → Kredi puanı
#Geography → Müşterinin bulunduğu ülke
#Gender → Cinsiyet
#Age → Yaş
#Tenure → Bankada kaç yıl kaldığı
#Balance → Hesap bakiyesi
#NumOfProducts → Sahip olunan banka ürünleri
#HasCrCard → Kredi kartı olup olmadığı (0 veya 1)
#IsActiveMember → Aktif müşteri olup olmadığı (0 veya 1)
#EstimatedSalary → Tahmini maaş
#Exited → Hedef değişken (1 = terk eden, 0 = etmeyen)


##################################################
# 1. DATA PREPERATION
# VERININ HAZIRLANMASI
##################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
pd.set_option("display.float_format",lambda x: "%.4f" %x)


##################################################
# 2. DATA READING
# VERININ HAZIRLANMASI
##################################################
df_=pd.read_csv("Bank Customer Churn Prediction.csv")
print(df_)
df=df_.copy()

print(df.head())
print(df.info())
print(df.describe().T)
print(df.nunique())
print(df.isnull().sum())
# The head(), info(), describe(), nunique() and isnull().sum()
# functions provide basic information about the data.


# Visualize the distribution of customer churn, churn rates by gender and country.
sns.countplot(x=df["churn"], palette="coolwarm")
plt.title("Customer Churn Distribution")
plt.show()

sns.countplot(x="gender", hue="churn", data=df, palette="viridis")
plt.title("Customer Churn by Gender")
plt.show()

sns.countplot(x="country", hue="churn", data=df, palette="magma")
plt.title("Customer Churn by Country")
plt.show()




# The age distribution of customers who left and did not leave is compared.
plt.figure(figsize=(8, 5))
sns.kdeplot(df[df["churn"] == 0]["age"], label="Terk Etmeyen", shade=True)
sns.kdeplot(df[df["churn"] == 1]["age"], label="Terk Eden", shade=True)
plt.title("Yaş Dağılımı - Müşteri Kaybı Karşılaştırması")
plt.legend()
plt.show()


# The gender column is numericized with LabelEncoder.
# (0: Female, 1: Male).
le = LabelEncoder()
df["gender"] = le.fit_transform(df["gender"])  # Erkek: 1, Kadın: 0

# # The country column is converted to categorical data using One-Hot Encoding.
df = pd.get_dummies(df, columns=["country"], drop_first=True)
#Germany → 1 ise Almanya’daki müşteri
#Spain → 1 ise İspanya’daki müşteri
#İkisi de 0 ise Fransa’daki müşteri
print(df.head())

# Correlations between data are visualized.
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Değişkenler Arası Korelasyon Matrisi")
plt.show()

# Numeric variables (credit score, age, tenure, etc.) are scaled with StandardScaler.
scaler = StandardScaler()
numeric_features = ["credit_score", "age", "tenure", "balance", "products_number", "estimated_salary"]

df[numeric_features] = scaler.fit_transform(df[numeric_features])



# Independent (X) and dependent (y) variables are determined.
# Data is divided into training and test sets.
X = df.drop(columns=["churn"])  # Bağımsız değişkenler
y = df["churn"]  # Hedef değişken

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)




# The model is trained using RandomForestClassifier.
# The accuracy of the model is evaluated with the classification report and confusion matrix.
X = df.drop('churn', axis=1)  # Bağımsız değişkenler
y = df['churn']  # Bağımlı değişken (Churn - Müşteri terk durumu)

# Splitting into training and test set (test set rate: %20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")





# Creating the model
model = RandomForestClassifier(random_state=42)

# Training the model
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Modelin doğruluğunu hesaplama
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculating model accuracy
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))




# Using RandomizedSearchCV, the model's hyperparameters are optimized.
# The best parameters are found.
param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Pattern search with RandomizedSearchCV
random_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_dist, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Model search with training data
random_search.fit(X_train, y_train)

# Printing the best parameters
print("Best parameters found: ", random_search.best_params_)

# Choosing the best model
best_model = random_search.best_estimator_

# Making predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluating model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with best parameters: {accuracy}")

# Performance report
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))




# Using GridSearchCV, the hyperparameters of the model are further optimized.
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Searching for patterns with GridSearchCV
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, verbose=2, n_jobs=-1)

# Model search with training data
grid_search.fit(X_train, y_train)

# Printing the best parameters
print("Best parameters found: ", grid_search.best_params_)

# Choosing the best model
best_model = grid_search.best_estimator_

# Making predictions on the test set
y_pred = best_model.predict(X_test)

# Model doğruluğunu değerlendirme
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with best parameters: {accuracy}")

# Performance report
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))



import joblib

# Modeli kaydetme
joblib.dump(best_model, 'bank_churn_model.pkl')

# Kaydedilen modeli yükleme
loaded_model = joblib.load('bank_churn_model.pkl')

# Test verisi ile tahmin yapma
y_pred = loaded_model.predict(X_test)
