import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

sales = pd.read_csv('./sales.csv')
print(sales)
print(sales.shape)

# Train model
X=sales['timeperiod'].values.reshape(-1, 1).ravel()
y=sales['sales'].values.reshape(-1, 1).ravel()

lm=LinearRegression()
model = lm.fit(X, y)

filehandler = open('./production/l2emodel.pkl', 'wb') 
pickle.dump(model, filehandler)