from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def train_model(model_name, X_train, y_train):
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree":
        model = DecisionTree(max_depth=7, random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators = 100, max_depth=7, random_state=42)
    model.fit(X_train, y_train)
    return model
