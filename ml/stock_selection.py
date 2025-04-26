# ml/stock_selection.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 假设我们有历史股票特征与标签数据
data = pd.DataFrame({
    'momentum': [0.1, 0.3, -0.2, 0.05, 0.4, -0.1, 0.2, 0.5],
    'volatility': [0.2, 0.15, 0.3, 0.25, 0.1, 0.35, 0.2, 0.12],
    'label': [1, 1, 0, 1, 1, 0, 1, 1]  # 1=好股, 0=坏股
})
X = data[['momentum', 'volatility']]
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred))
