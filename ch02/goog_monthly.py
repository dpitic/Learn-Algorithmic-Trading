"""Seasonality
This module studies the price variations based on months of GOOG data from 
2001 to 2018. The code regroups the data by months, calculates and returns the
monthly returns, and compares these returns in a histogram.
"""
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

from algolib.data import get_google_data, plot_rolling_statistics_ts, \
    plot_stationary_ts, augmented_dickey_fuller


def main():
    # Get GOOG data from Yahoo Finance from 2001-01-01 to 2018-01-01
    start_date = '2001-01-01'
    goog_data_file = 'data/goog_data_large.pkl'
    goog_data = get_google_data(goog_data_file, start_date=start_date)
    print(goog_data)

    # Calculate the average monthly percentage change of the daily adjusted
    # close price.
    goog_monthly_return = goog_data['Adj Close'].pct_change().groupby(
        [goog_data['Adj Close'].index.year,
         goog_data['Adj Close'].index.month]).mean()
    goog_monthly_return_list = []

    for i in range(len(goog_monthly_return)):
        goog_monthly_return_list.append(
            {'month': goog_monthly_return.index[i][1],
             'monthly_return': goog_monthly_return[i]})

    goog_monthly_return_df = pd.DataFrame(goog_monthly_return_list,
                                          columns=['month', 'monthly_return'])
    print(goog_monthly_return_df)
    goog_monthly_return_df.boxplot(column='monthly_return', by='month')

    ax = plt.gca()
    labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticklabels(labels)
    ax.set_ylabel('GOOG return')
    plt.title('GOOG Monthly Return 2001-2018')
    plt.suptitle('')

    # With trend
    plot_rolling_statistics_ts(
        goog_monthly_return,
        'GOOG prices rolling mean and standard deviation',
        'Monthly return')
    plot_rolling_statistics_ts(
        goog_data['Adj Close'],
        'GOOG prices rolling mean and standard deviation',
        'Daily prices', 365)

    # Without trend (stationary)
    plot_stationary_ts(goog_data['Adj Close'], 'GOOG prices without trend',
                       'Daily prices', window_size=365)

    # Augmented Dickey-Fuller test
    df = augmented_dickey_fuller(goog_data['Adj Close'])
    print('Augmented Dickey-Fuller test for adjusted close price:')
    print(df)
    df = augmented_dickey_fuller(goog_monthly_return)
    print('Augmented Dickey-Fuller test for monthly return:')
    print(df)

    # Forecast time series using Auto-Regression Integrated Moving Averages
    # (ARIMA) model
    plt.figure()
    plt.subplot(211)
    # Plot autocorrelation function (ACF) for monthly returns
    plot_acf(goog_monthly_return, ax=plt.gca(), lags=10)
    # Plot partial autocorrelation function (PACF) for monthly returns
    plt.subplot(212)
    plot_pacf(goog_monthly_return, ax=plt.gca(), lags=10)
    # The Autoregressive term AR(p) is the number of lags of dependent variable.
    # The lag value is p=1 when the PACF crosses the upper confidence interval
    # for the first time.
    # The Moving Average term MA(q) is the number of lags for errors in
    # prediction; error = (moving average) - (actual value)
    # The lag value is q=1 when the ACF plot crosses the upper confidence
    # interval for the first time.
    model = ARIMA(goog_monthly_return, order=(2, 0, 2))
    fitted_results = model.fit()
    plt.figure()
    goog_monthly_return.plot(label='GOOG Monthly Return $', lw=0.5)
    fitted_results.fittedvalues.plot(label='Forecast', color='red')
    plt.grid()
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
