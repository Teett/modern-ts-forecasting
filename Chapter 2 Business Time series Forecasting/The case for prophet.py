#%% Dependencies
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot

#%% Read the data
data = pd.read_csv('../data/daily-website-visitors.csv', thousands=',')
# %% Check the data
data.head()
data.shape
data.dtypes
# %% Process the data
data['Date'] = pd.to_datetime(data.Date)
ts_data = data[['First.Time.Visits', 'Date']]
ts_data = ts_data.query('Date >= "2017-01-01"')
ts_data = ts_data.sort_values(by='Date')
ts_data = ts_data.rename(columns={'First.Time.Visits':'y', 'Date':'ds'})
ts_data
# %% Plot the data
plt.plot('ds', 'y', data=ts_data)
plt.xlabel('Date')
plt.ylabel('Count')
plt.show()
# %% Train and test split
train_len = int(ts_data.shape[0] * 0.85)
train = ts_data.iloc[:train_len:]
test = ts_data.iloc[train_len:,:]
[train_len, len(test)]
# %% Build the model
# fit with default parameters
m = Prophet()
m.add_country_holidays(country_name='US') # add US holidays
m.fit(train)

# %% Generate data points for the future period
future = m.make_future_dataframe(periods=len(test), freq='d')
future.tail()

# %%
forecast = m.predict(future)
forecast.tail()
# %%
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)
# %%
