import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math as math
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pypfopt import EfficientFrontier, risk_models, expected_returns
#START FEATURE ENGINEERING#################################################################################################################

#10-2 DATA
bond_data = pd.read_excel(r'C:\Users\wgran\OneDrive\Desktop\BACKTESTS\T10Y2Y.xls', index_col=0)
#Make NaN values in the spread the average of the two values nearest
bond_data['Spread'] = bond_data['Spread'].interpolate(method='linear')

#XLRE and XLC don't have enough data, SO 9 SECTORS TOTAL
#SECTOR DATA

# 2009-07-04 is esp good start
start = "2001-01-01"
end = "2024-01-01"

train_start_date = '2010-01-01'
train_end_date = '2014-12-31'
test_start_date = '2015-01-01'
test_end_date = '2024-01-01'


sector_tickers = ['XLK', 'XLV', 'XLF', 'XLY','XLI', 'XLP', 'XLE', 'XLB', 'XLU']
data = yf.download(sector_tickers, start=start, end=end)['Adj Close']

#VIX DATA
vix_data = yf.download('^VIX', start='1999-01-01', end='2024-01-01')['Adj Close']
data['VIX_15d_Change'] = vix_data / vix_data.shift(15) - 1

column_names= ['Data', 'Data']
#UNEMPLOYMENT DATA
unemployment_claims_data = pd.read_excel(r'C:\Users\wgran\OneDrive\Desktop\BACKTESTS\unemployment_claims_data.xlsx', index_col=0, skiprows=6, names=column_names)[::-1].iloc[:,0].interpolate(method='linear')
monthly_unemployment_claims = unemployment_claims_data.resample('M').last()
monthly_percent_change_unemployment_claims = (monthly_unemployment_claims / monthly_unemployment_claims.shift(1) - 1) * 100
monthly_percent_change_unemployment_claims = monthly_percent_change_unemployment_claims.asfreq('D', method = 'ffill')
daily_unemployment_claims_data = unemployment_claims_data.asfreq('D', method='ffill')

#FED FUNDS DATA
fed_funds_data = pd.read_excel(r'C:\Users\wgran\OneDrive\Desktop\BACKTESTS\fed_funds_data.xlsx', index_col=0, skiprows=6, names=column_names)[::-1].iloc[:,0].interpolate(method='linear')
fed_funds_data = 100 - fed_funds_data
fed_funds_monthly = fed_funds_data.resample('M').last()
fed_funds_monthly = fed_funds_monthly.pct_change()
fed_funds_monthly = fed_funds_monthly.asfreq('D', method='ffill')

#Purchasing Manager data
PM_data = pd.read_excel(r'C:\Users\wgran\OneDrive\Desktop\BACKTESTS\purchasing_manager_data.xlsx', index_col=0, skiprows=6, names=column_names)[::-1].iloc[:,0].interpolate(method='linear')
#Percent change in monthly PM data
PM_Monthly_percent_change = (PM_data / PM_data.shift(1) - 1) * 100
PM_Monthly_percent_change = PM_Monthly_percent_change.asfreq('D', method = 'ffill')
PM_data = PM_data.asfreq('D', method='ffill')

#Consumer Sentiment data
consumer_sentiment_data = pd.read_excel(r'C:\Users\wgran\OneDrive\Desktop\BACKTESTS\consumer_sentiment_data.xlsx', index_col=0, skiprows=7, names=column_names)[::-1].iloc[:,0].interpolate(method='linear')
consumer_sentiment_monthly_percent_change = (consumer_sentiment_data / consumer_sentiment_data.shift(1) - 1) * 100
consumer_sentiment_monthly_percent_change = consumer_sentiment_monthly_percent_change.asfreq('D', method = 'ffill')
consumer_sentiment_data = consumer_sentiment_data.asfreq('D', method='ffill')

#GDP data
GDP_data = pd.read_excel(r'C:\Users\wgran\OneDrive\Desktop\BACKTESTS\GDP_data.xls', index_col=0, skiprows=10, names=column_names)[::-1].iloc[:,0].interpolate(method='linear')
GDP_quarterly = GDP_data.resample('Q').last()
#change in percent
GDP_quarterly_percent_change = GDP_quarterly.pct_change() * 100
GDP_quarterly_percent_change = GDP_quarterly_percent_change.asfreq('D', method = 'ffill')
GDP_data = GDP_data.asfreq('D', method='ffill')


#INFLATION DATA
inflation_data = pd.read_csv(r'C:\Users\wgran\OneDrive\Desktop\BACKTESTS\inflation_data.csv', index_col=0)
inflation_data.rename(columns={'CPILFESL_PC1': 'data'}, inplace=True)
inflation_data.index = pd.to_datetime(inflation_data.index)
inflation_data = inflation_data.asfreq('D', method = 'ffill')


#WTI DATA
WTI_data = (pd.read_csv(r'C:\Users\wgran\OneDrive\Desktop\BACKTESTS\WTI_data.csv', index_col='DATE')
              .replace('.', np.nan)
              .apply(pd.to_numeric, errors='coerce')
              .interpolate(method='linear')
              .astype(float))
WTI_data.index = pd.to_datetime(WTI_data.index)
WTI_M_Change = WTI_data.pct_change(periods=21) * 100


#Put all of the data together in data df
data["FF"] = fed_funds_data 
data['FF_M_Change'] = fed_funds_monthly
data['Unemployment_IC'] = daily_unemployment_claims_data 
data['Unemployment_IC_M_Change_%'] = monthly_percent_change_unemployment_claims
data['PM_Daily'] = PM_data 
data['PM_M_Change_%'] = PM_Monthly_percent_change
data['CS_Daily'] = consumer_sentiment_data 
data['CS_M_Change_%'] = consumer_sentiment_monthly_percent_change
data['GDP'] = GDP_data 
data['GDP_Q_Change_%'] = GDP_quarterly_percent_change
data['VIX'] = vix_data
data['VIX_15d_Points_Change'] = vix_data / vix_data.shift(15) - 1
data['10-2_spread'] = bond_data['Spread']
data['CPI_Daily'] = inflation_data['data']
data['WTI_Daily'] = WTI_data
data['WTI_M_Change_%'] = WTI_M_Change



#END FEATURE ENGINEERING#################################################################################################################



holdtime_days = 20


#baseline model has all possible features
generic_features = ['FF', 'FF_M_Change', 'Unemployment_IC', 'Unemployment_IC_M_Change_%', 'PM_Daily', 'GDP','PM_M_Change_%', 'CS_Daily', 'CS_M_Change_%', 'GDP_Q_Change_%', 'VIX', 'VIX_15d_Points_Change', '10-2_spread', 'CPI_Daily', 'WTI_Daily', 'WTI_M_Change_%']
momo_features = ['1M_MOM', '3M_MOM']
all_features = ['FF', 'FF_M_Change', 'Unemployment_IC', 'Unemployment_IC_M_Change_%', 'PM_Daily', 'PM_M_Change_%', 'CS_Daily', 'CS_M_Change_%', 'GDP_Q_Change_%', 'VIX', 'VIX_15d_Points_Change', '10-2_spread', 'CPI_Daily', 'WTI_Daily', 'WTI_M_Change_%', '1M_MOM', '3M_MOM']
#make a dict that has each sector as keys and the corresponding features as the values 
sector_specific_features = {}
for etf in sector_tickers:
    sector_specific_features[etf] = all_features


post_lasso_features = sector_specific_features
all_possible_features = ['const'] + generic_features + ['1M_MOM', '3M_MOM']
coefficient_matrix_ESTIMATE = pd.DataFrame(columns=sector_tickers, index=all_possible_features).astype(float)

for etf in sector_tickers:
    data[f'{etf}_1M_MOM'] = data[etf].pct_change(periods=21) * 100
    data[f'{etf}_3M_MOM'] = data[etf].pct_change(periods=63) * 100
for etf in sector_tickers:
    data[f'{etf}_future_return'] = (data[etf].shift(-holdtime_days) / data[etf] - 1) * 100
#update train data here!
#THIS IS TO AVOID A VERY SLY FORM OF LOOKAHEAD BIAS!
end_date_minus_holdtime_index =  data.index.get_loc(train_end_date)-holdtime_days
end_date_minus_holdtime_date = data.index[end_date_minus_holdtime_index]
train_data = data.loc[train_start_date:end_date_minus_holdtime_date].copy()
test_data = data.loc[test_start_date:test_end_date].copy()




r_squared_values = {}
#multi sector version with loop:
for sector in sector_tickers:
    X = train_data[generic_features].copy()
    if any('1M' in feature for feature in sector_specific_features[sector]):
        X['1M_MOM'] = train_data[f'{sector}_1M_MOM'].copy()
    if any('3M' in feature for feature in sector_specific_features[sector]):
        X['3M_MOM'] = train_data[f'{sector}_3M_MOM'].copy()
    Y = train_data[f'{sector}_future_return']
    model = sm.OLS(Y, sm.add_constant(X)).fit()
    print(model.summary())
    r_squared_values[sector] = model.rsquared_adj
    for feature in ['const'] + list(X.columns):
        coefficient_matrix_ESTIMATE.at[feature, sector] = model.params.get(feature, 0)
average_r_squared = np.mean(list(r_squared_values.values()))
print("Coefficient Matrix:")
print(coefficient_matrix_ESTIMATE)
print(f'\nAverage adj R-squared value: {average_r_squared:.4f}')





def sector_sorter(coefficient_matrix_ESTIMATE, data, current_date):
    sorted_sectors = {}
    current_data = data.loc[current_date]
    for etf in sector_tickers:
        coef_col = coefficient_matrix_ESTIMATE[etf]
        intercept = coef_col.get('const', 0)
        feature_values = current_data.reindex(coef_col.index.drop('const', errors='ignore'), fill_value=0).astype(float)
        prediction = (coef_col.drop('const', errors='ignore') * feature_values).sum() + intercept
        sorted_sectors[etf] = prediction
    sorted_sectors = pd.Series(sorted_sectors, name='Prediction').sort_values(ascending=False)
    return sorted_sectors




def equal_weight(aum):# improve this later
    stock_num = 8
    return aum / stock_num


positions = pd.DataFrame(0, index=data.index, columns=sector_tickers)
def position_updater(aum, positions, current_date, data, sorted_sectors):
    #Calculate the new weight for each position in the portfolio
    new_weight = equal_weight(aum)  
    #Get index of today's date
    current_date_index = data.index.get_loc(current_date)
    #Get today's row of price data
    current_prices = data.loc[current_date]
    #The new position has 0 gains here, so it begins realizing pnl the day after initiated
    position_realized_gains_index = current_date_index + 1
    position_realized_gains_date = data.index[position_realized_gains_index]
    #LONGS:
    for sector in sorted_sectors.index[0:4]:
        positions.at[position_realized_gains_date, sector] = math.floor(new_weight / current_prices[sector])
    #SHORTS:
    for sector in sorted_sectors.index[5:9]:  
        positions.at[position_realized_gains_date, sector] = -(math.floor(new_weight / current_prices[sector]))
    return positions
    
    



aum_over_time = pd.Series(index=data.index, dtype='float64')
def daily_AUM_update(current_date, data, lastAUM, positions):
    #each new day, when this is called, we care about the prev day's prices to back out daily change in pnl
    position_index = data.index.get_loc(current_date)
    
    #Assuming 'data' contains price data, and 'positions' contains the number of shares/units for each sector
    todays_prices = data.loc[current_date]  # Use loc with current_date directly
    prev_day_date = data.index[position_index - 1]
    prev_day_prices = data.loc[prev_day_date]
    
    #Initialize pnl variables
    long_pnl = 0
    short_pnl = 0
    
    #Loop through each sector in positions DataFrame
    for sector in positions.columns:
        #Check if the position was short:
        if positions.at[prev_day_date, sector] < 0:
            price_change_in_favor = prev_day_prices[sector] - todays_prices[sector]
            short_pnl += price_change_in_favor * abs(positions.at[prev_day_date, sector]) *3
        #Check if the position was long:
        elif positions.at[prev_day_date, sector] > 0:  # Corrected the condition here
            price_change_in_favor = todays_prices[sector] - prev_day_prices[sector]
            long_pnl += price_change_in_favor * positions.at[prev_day_date, sector] *3
    
    daily_pnl = long_pnl + short_pnl
    newAUM = lastAUM + daily_pnl
    
    return newAUM


import pandas as pd
import statsmodels.api as sm
import numpy as np

#want the data from data df up to the current date, back to train_start_date
def reEstimateCoefficients(data, train_start_date, current_date, sector_specific_features):
    #THIS IS TO AVOID A VERY SLY FORM OF LOOKAHEAD BIAS!
    current_date_minus_holdtime_index = data.index.get_loc(current_date) - holdtime_days
    current_date_minus_holdtime_date = data.index[current_date_minus_holdtime_index]
    realized_data = data.loc[train_start_date:current_date_minus_holdtime_date].copy()
    coefficient_matrix_ESTIMATE = pd.DataFrame(columns=sector_tickers, index=all_possible_features).astype(float)
    for etf in sector_tickers:
        data[f'{etf}_1M_MOM'] = data[etf].pct_change(periods=21) * 100
        data[f'{etf}_3M_MOM'] = data[etf].pct_change(periods=63) * 100
    for etf in sector_tickers:
        data[f'{etf}_future_return'] = (data[etf].shift(-holdtime_days) / data[etf] - 1) * 100
    for sector in sector_tickers:
        X = realized_data[generic_features].copy()
        if any('1M' in feature for feature in sector_specific_features[sector]):
            X['1M_MOM'] = realized_data[f'{sector}_1M_MOM'].copy()
        if any('3M' in feature for feature in sector_specific_features[sector]):
            X['3M_MOM'] = realized_data[f'{sector}_3M_MOM'].copy()
        Y = realized_data[f'{sector}_future_return']
        model = sm.OLS(Y, sm.add_constant(X)).fit()
        #print(model.summary())
        r_squared_values[sector] = model.rsquared_adj
        for feature in ['const'] + list(X.columns):
           coefficient_matrix_ESTIMATE.at[feature, sector] = model.params.get(feature, 0)
        new_train_start_date = pd.to_datetime(train_start_date) + pd.Timedelta(days=20)
        if new_train_start_date > pd.to_datetime(current_date):
           raise ValueError("New training start date exceeds the current date.")



    return coefficient_matrix_ESTIMATE, new_train_start_date





#MAIN BACKTEST

from pandas import to_datetime

positions = pd.DataFrame(0, index=data.index, columns=sector_tickers)
aum_over_time = pd.Series(index=data.index, dtype='float64')
aum_OOS = pd.Series(index=test_data.index, dtype='float64')
starting_AUM = 100000
current_AUM = starting_AUM
day_counter = 0
oos_starting_aum = 0

#holdtime_days = 20
test_start_date_dt = to_datetime(test_start_date)
train_start_date_dt = pd.to_datetime(train_start_date)
filtered_data = data.loc[test_start_date_dt:]
for current_date in filtered_data.index:
    #if it's day 0 or if it's time to rebalance:
    if day_counter == 0 or day_counter % holdtime_days == 0:
        #first do the change in aum
        current_AUM = daily_AUM_update(current_date, data, current_AUM, positions)
        #reestimate the params(no lookahead bias)
        test_start_date_dt = to_datetime(test_start_date)
        test_end_date_dt = to_datetime('2023-10-01')
        if current_date >= test_start_date_dt and current_date < test_end_date_dt:
            train_start_date = reEstimateCoefficients(data, train_start_date, current_date, sector_specific_features)[1]
            coefficient_matrix_ESTIMATE = reEstimateCoefficients(data, train_start_date, current_date, sector_specific_features)[0]
            
        #rmbr to fill in daily aum tracker
        aum_over_time[current_date] = current_AUM
        if current_date >= test_start_date_dt:
            aum_OOS[current_date] = current_AUM
        #decide how to position at the close of the current day after closing old positions
        sorted_sectors = sector_sorter(coefficient_matrix_ESTIMATE, data, current_date)
        #update the positions: rmbr this sets positions for the NEXT DAY CLOSE
        positions = position_updater(current_AUM, positions, current_date, data, sorted_sectors)
    
    else: 
        #if we didnt take new positions at rebalance period, still update aum
        current_AUM = daily_AUM_update(current_date, data, current_AUM, positions)
        #rmbr to fill in daily aum tracker
        aum_over_time[current_date] = current_AUM
        #Have to rmbr to update positions here
        position_index = data.index.get_loc(current_date)
        #find the index # of the row where position was initiated
        future_pos_index = position_index  + 1 #tmrws index
        if future_pos_index < len(data.index):
            next_date = data.index[data.index.get_loc(current_date) + 1]#tmrws date
            positions.loc[next_date] = positions.loc[current_date]#fill in the next days positions
        if current_date >= test_start_date_dt:
            aum_OOS[current_date] = current_AUM
    if current_date == test_start_date_dt:
        oos_starting_aum = current_AUM
    day_counter += 1
    
print(current_AUM)

aum_over_time.plot(title='AUM Over Time')
aum_OOS.plot(title='OOS AUM Over Time')



def Simple_sharpe_calc(aum_series, rf_rate=0.0):
    #Convert the risk-free rate from annual to daily
    rf_daily = (1 + rf_rate)**(1/252) - 1
    #Calculate daily returns from AUM
    daily_returns = aum_series.pct_change().dropna()  # Removing the first NaN value
    #Annualize the mean and std of daily returns
    ann_return = np.mean(daily_returns) * 252
    ann_vol = np.std(daily_returns) * np.sqrt(252)
    #Calculate the Sharpe Ratio
    sharpe = (ann_return - rf_rate) / ann_vol
    return sharpe, ann_vol, ann_return


sharpe_ratio = Simple_sharpe_calc(aum_over_time)[0]
ann_vol = Simple_sharpe_calc(aum_over_time)[1] * 100
annual_ret = Simple_sharpe_calc(aum_over_time)[2] * 100
oos_sharpe = Simple_sharpe_calc(aum_OOS)[0]
oos_ann_vol = Simple_sharpe_calc(aum_OOS)[1] * 100
oos_annual_ret = Simple_sharpe_calc(aum_OOS)[2] * 100
print("annual return(simple annual avg): " + str(annual_ret))
print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Ann Vol: {ann_vol}")
print("OOS annual return: " + str(oos_annual_ret))
print(f"OOS Sharpe Ratio: {oos_sharpe}")
print(f"OOS Ann Vol: {oos_ann_vol}")



starting_AUM = 100000  # Starting value for both strategy AUM and SPY


spy_data = yf.download('SPY', start=test_start_date, end=test_end_date)['Adj Close']
spy_performance = (spy_data / spy_data.iloc[0]) * starting_AUM

plt.figure(figsize=(14, 7))
plt.plot(aum_over_time, label='Strategy AUM')
plt.plot(spy_performance, label='SPY Benchmark')
plt.title('Strategy AUM vs. SPY Benchmark')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)


strategy_end_value = aum_over_time.iloc[-1]
spy_end_value = spy_performance.iloc[-1]
plt.annotate(f"${strategy_end_value:.2f}", 
             (aum_over_time.index[-1], strategy_end_value),
             textcoords="offset points",
             xytext=(0,10),
             ha='center')
plt.annotate(f"${spy_end_value:.2f}", 
             (spy_performance.index[-1], spy_end_value),
             textcoords="offset points",
             xytext=(0,10),
             ha='center')

plt.show()

#Log Scale Plot
plt.figure(figsize=(14, 7))
plt.plot(aum_over_time, label='Strategy AUM')
plt.plot(spy_performance, label='SPY Benchmark')
plt.title('Strategy AUM vs. SPY Benchmark (Log Scale)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.yscale('log')  
plt.legend()
plt.grid(True)


plt.annotate(f"${strategy_end_value:.2f}", 
             (aum_over_time.index[-1], strategy_end_value),
             textcoords="offset points",
             xytext=(0,10),
             ha='center')
plt.annotate(f"${spy_end_value:.2f}", 
             (spy_performance.index[-1], spy_end_value),
             textcoords="offset points",
             xytext=(0,10),
             ha='center')

plt.show()
