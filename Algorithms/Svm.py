import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from constants import DATA_DIRECTORY
import seaborn as sns

df = pd.read_csv(f"{DATA_DIRECTORY}/1000630_with_leakages.csv", parse_dates=['Timestamp'])
df.dropna(inplace=True) # Drop empty entries

# Sort chronologically
df.sort_values("Timestamp", inplace=True)

# Calculate time-based split index for 30% leak zone
split_index = int(len(df) * 0.7)

# Add target column: 0 for no leak, 1 for leak -- Add leakage to the last 30% of data
df['Leak'] = 0
df.loc[df.index[split_index:], 'Leak'] = 1

# Time-based features
df['Hour'] = df['Timestamp'].dt.hour
df['DayOfWeek'] = df['Timestamp'].dt.dayofweek  # Monday=0, Sunday=6

# Statistical features
df['Leakage_Diff_Constant'] = df['Volume_with_Constant_Leakage'] - df['Volume']
df['Leakage_Diff_Variant'] = df['Volume_with_Variable_Leakage'] - df['Volume']
df['RollingMean'] = df['Leakage_Diff_Variant'].rolling(window=3).mean().bfill()
df['Volume_Diff'] = df['Volume'].diff().fillna(0)

X = df[['Leakage_Diff_Variant', 'Leakage_Diff_Constant', 'Hour', 'DayOfWeek', 'RollingMean', 'Volume_Diff']]
y = df['Leak']

# Train/test split (shuffled for generalization)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM
params = {
    'C': 10,
    'gamma': 0.01,
    'kernel': 'rbf'
}
svm = SVC( **params, class_weight='balanced')
svm.fit(X_train_scaled, y_train)

# Predict
y_pred = svm.predict(X_test_scaled)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Visualize feature vs label
plt.figure(figsize=(10, 4))
sns.histplot(data=df, x='Leakage_Diff_Variant', hue='Leak', bins=50, kde=True, palette='Set2')
plt.title("Leakage Difference Distribution")
plt.xlabel("Leakage_Diff_Variant (Volume_with_Variable_Leakage - Volume)")
plt.ylabel("Count")
plt.show()
