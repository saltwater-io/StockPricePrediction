import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


def run():
    quandl.ApiConfig.api_key = "[REDACTED]"
    data = quandl.get("WIKI/AMZN")

    # Adjusted close price
    data = data[['Adj. Close']]
    # print(data.head())

    # n days out into future
    forecast_days = 30

    # Target value column
    data['Predicted Value'] = data[['Adj. Close']].shift(-forecast_days)

    # convert data to numpy array
    x = np.array(data.drop(['Predicted Value'], 1))

    # remove last n rows
    x = x[:-forecast_days]

    # Convert data to numpy array (All of the values)
    prediction = np.array(data['Predicted Value'])
    prediction = prediction[:-forecast_days]

    # 80% training, 20% testing
    x_train, x_test, y_train, y_test = train_test_split(x, prediction, test_size=0.2)

    # Create and train Support Vector Machine (Regression)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(x_train, y_train)

    # Coefficient score R^2
    # Best possible score = 1.0
    svm_confidence = svr_rbf.score(x_test, y_test)
    print("svm confidence: ", svm_confidence)

    # Linear regression model
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    # Testing linear regression model
    lr_confidence = lr.score(x_test, y_test)
    print("lr_confidence: ", lr_confidence)

    # sets x forecast equal to last 30 rows of original dataset -> Adj. Close column
    x_forecast = np.array(data.drop(['Predicted Value'], 1))[-forecast_days:]
    print(x_forecast)

    # predictions for the next n days - support vector regressor
    svm_prediction = svr_rbf.predict(x_forecast)
    print(svm_prediction)

    # predictions for the next n days - linear regression
    lr_prediction = lr.predict(x_forecast)
    print(lr_prediction)


if __name__ == "__main__":
    run()
