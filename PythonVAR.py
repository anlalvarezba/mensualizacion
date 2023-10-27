import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf

from tabulate import tabulate
import sys

#######Import the datasets
filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/Raotbl6.csv'
df = pd.read_csv(filepath, parse_dates=['date'], index_col='date')
print(df.shape)  # (123, 8)
df.tail()


#######Visualize the Time Series
# Plot
fig, axes = plt.subplots(nrows=4, ncols=2, dpi=120, figsize=(10,6))
for i, ax in enumerate(axes.flatten()):
    data = df[df.columns[i]]
    ax.plot(data, color='red', linewidth=1)
    # Decorations
    ax.set_title(df.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();
plt.savefig("originalTimeSeries.png")

#######Testing Causation using Granger’s Causality Test
from statsmodels.tsa.stattools import grangercausalitytests
maxlag = 12
test = 'ssr_chi2test'

def grangers_causation_matrix(data, variables, test='ssr_chi2test'):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables)), dtype=float), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag)
            p_values = [round(test_result[i+1][0][test][1], 4) for i in range(maxlag)]
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
            # Manually print the results
            # The next step prints each iteration of the code
            # print(f'Y = {r}, X = {c}, P Values = {p_values}')
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

causation_matrix = grangers_causation_matrix(df, variables=df.columns)

# Print the causation matrix on the Terminal
# print("Granger Causality Matrix:")
# print(causation_matrix)

# Define the file path where you want to save the output
output_file_path1 = "granger_causality_matrix.txt"

# Open the file for writing
with open(output_file_path1, 'w') as file:
    # Write the causation matrix to the file
    file.write("Granger Causality Matrix:\n")
    file.write(causation_matrix.to_string())

# Print a message indicating the file has been saved
print(f"Output saved to {output_file_path1}")






#######Testing Cointegration using Soren Johanssen test
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

    # Define the output file path
    output_file_path2 = "cointegration_test_output.txt"

    # Open the file for writing
    with open(output_file_path2, 'w') as file:
        # Write the summary to the file
        file.write('Name   ::  Test Stat > C(95%)    =>   Signif  \n')
        file.write('--'*20 + '\n')
        for col, trace, cvt in zip(df.columns, traces, cvts):
            file.write(adjust(col) + ':: ' + adjust(round(trace, 2), 9) + " > " + adjust(cvt, 8) + ' => ' + str(trace > cvt) + '\n')

    # Print a message indicating the file has been saved
    print(f"Output saved to {output_file_path2}")

# Call the cointegration_test function
cointegration_test(df)

########### 8. Split the Series into Training and Testing Data
nobs = 4
df_train, df_test = df[0:-nobs], df[-nobs:]

# Check size
print(df_train.shape)  # (119, 8)
print(df_test.shape)  # (4, 8)


########### 9. Check for Stationarity and Make the Time Series Stationary
### Usin the Augmented Dickey-Fuller Test (ADF Test) to check stationarity
def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Create a summary string
    summary = f"===================================================\n"
    summary += f'Augmented Dickey-Fuller Test on "{name}"\n'
    summary += ' ' + '-'*47 + '\n'
    summary += f'Null Hypothesis: Data has unit root. Non-Stationary.\n'
    summary += f'Significance Level    = {signif}\n'
    summary += f'Test Statistic        = {output["test_statistic"]}\n'
    summary += f'No. Lags Chosen       = {output["n_lags"]}\n'

    for key,val in r[4].items():
        summary += f'Critical value {adjust(key)} = {round(val, 3)}\n'

    if p_value <= signif:
        summary += f"=> P-Value = {p_value}. Rejecting Null Hypothesis.\n"
        summary += f"=> Series is Stationary.\n"
    else:
        summary += f"=> P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.\n"
        summary += f"=> Series is Non-Stationary.\n"

    return summary


# ADF Test on each column
print('ADF Test on original series:\n')
output_strings = []
for name, column in df_train.items():
    output_string = adfuller_test(column, name=column.name)
    output_strings.append(output_string)
    print(output_string)
    print('\n')

# Define the output file path
output_file_path3 = "adfuller_test_output.txt"

# Save the output strings to a text file
with open(output_file_path3, 'w') as file:
    for output_string in output_strings:
        file.write(output_string)

# Print a message indicating the file has been saved
print(f"Output saved to {output_file_path3}")


####---------The ADF test confirms none of the time series is stationary. 
####---------Let’s difference all of them once and check again:

# 1st difference
df_differenced = df_train.diff().dropna()

# ADF Test on each column of 1st Differences Dataframe
print('ADF Test on 1rst differenced series:\n')
output_strings_1diff = []
for name, column in df_differenced.items():
    output_string_1diff = adfuller_test(column, name=column.name)
    output_strings_1diff.append(output_string_1diff)
    print(output_string_1diff)
    print('\n')

# Define the output file path
output_file_path3_1diff = "adfuller_test_output_1diff.txt"

# Save the output strings to a text file
with open(output_file_path3_1diff, 'w') as file:
    for output_string_1diff in output_strings_1diff:
        file.write(output_string_1diff)

# Print a message indicating the file has been saved
print(f"Output saved to {output_file_path3_1diff}")



####---------The ADF test confirms some of the time series are stationary but not everyone. 
####---------Let’s difference all of them once and check again:

# 2st difference
df_differenced2 = df_differenced.diff().dropna()

# ADF Test on each column of 2st Differences Dataframe
print('ADF Test on 2nd differenced series:\n')
output_strings_2diff = []
for name, column in df_differenced2.items():
    output_string_2diff = adfuller_test(column, name=column.name)
    output_strings_2diff.append(output_string_2diff)
    print(output_string_2diff)
    print('\n')

# Define the output file path
output_file_path3_2diff = "adfuller_test_output_2diff.txt"

# Save the output strings to a text file
with open(output_file_path3_2diff, 'w') as file:
    for output_string_2diff in output_strings_2diff:
        file.write(output_string_2diff)

# Print a message indicating the file has been saved
print(f"Output saved to {output_file_path3_2diff}")



############ 10. Select the Order (P) of VAR model
############ We do this with the stationary series
model = VAR(df_differenced2)
fitComparison = f'Fit comparison estimates to Select the Order (P) of the VAR model\n'
for i in [1,2,3,4,5,6,7,8,9]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')
    fitComparison += f'-----------------------------\n'
    fitComparison += f'Lag Order ="{i}"\n'
    fitComparison += f'AIC : "{result.aic}"\n'
    fitComparison += f'BIC : "{result.bic}"\n'
    fitComparison += f'FPE : "{result.fpe}"\n'
    fitComparison += f'HQIC : "{result.hqic}"\n'

# Define the output file path
output_file_path4 = "fitComparison_tests.txt"

# Save the output strings to a text file
with open(output_file_path4, 'w') as file:
    file.write(fitComparison)

##### Alternate method to choose the order(p):
x = model.select_order(maxlags=12)
alternateEstimates = x.summary()
print(alternateEstimates)

# Define the output file path
output_file_path5 = "fitComparison_alternative.txt"


# Save the output table to a text file
with open(output_file_path5, 'w') as file:
    file.write(tabulate(alternateEstimates))


############ 11. Train the VAR Model of Selected Order(p)
model_fitted = model.fit(4)
regressionsummary = model_fitted.summary()

# Define the output file path to save it
output_file_path6 = "VarRegression_results.txt"
sys.stdout = open(output_file_path6, 'w')
print(regressionsummary)
print(type(regressionsummary))
# Reset sys.stdout to its original value
sys.stdout = sys.__stdout__
# Close the output file
# sys.stdout.close()
# Print a message indicating the file has been saved
print(f"Output saved to {output_file_path6}")


########### 12. Check for Serial Correlation of Residuals (Errors) using Durbin Watson Statistic
out = durbin_watson(model_fitted.resid)
residualsCorrelation = f'Serial Correlation of Residuals (Errors):\n'
print(f'Serial Correlation of Residuals (Errors):')
for col, val in zip(df.columns, out):
    print(col.rjust(10), ':', round(val, 2))
    residualsCorrelation += f'{col.rjust(10)} : {round(val, 2)}\n'

# Define the output file path
output_file_path7 = "residualsCorrelation.txt"

# Save the Residuals Correlation to a text file
with open(output_file_path7, 'w') as file:
    file.write(residualsCorrelation)

# Print a message indicating the file has been saved
print(f"Output saved to {output_file_path7}")


########### 13. Forecast VAR model using statsmodels
# Get the lag order
lag_order = model_fitted.k_ar
print(lag_order)  #> 4

# Input data for forecasting
forecast_input = df_differenced2.values[-lag_order:]
forecast_input
print(forecast_input)

# Forecast
fc = model_fitted.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=df.index[-nobs:], columns=df.columns + '_2d')
df_forecast
print(df_forecast)



######## 14. Invert the transformation to get the real forecast
def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

df_results = invert_transformation(df_train, df_forecast, second_diff=True)        
df_results.loc[:, ['rgnp_forecast', 'pgnp_forecast', 'ulc_forecast', 'gdfco_forecast',
                   'gdf_forecast', 'gdfim_forecast', 'gdfcf_forecast', 'gdfce_forecast']]

# Print forecasts on their original scale on console
print(df_results)

# Define the output file path
output_file_path8 = "forecastsOriginalScale.csv"

# Save the forecasts in their original scale as .csv file
df_results.to_csv(output_file_path8)

# Print a message indicating the file has been saved
print(f"Output saved to {output_file_path8}")



############# 15. Plot of Forecast vs Actuals
fig, axes = plt.subplots(nrows=int(len(df.columns)/2), ncols=2, dpi=150, figsize=(10,10))
for i, (col,ax) in enumerate(zip(df.columns, axes.flatten())):
    df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    df_test[col][-nobs:].plot(legend=True, ax=ax);
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();
plt.savefig("forecastsVsActuals.png")



############### 16. Evaluate the Forecasts
#### Compute a set of metrics: MAPE, ME, MAE, MPE, RMSE, corr and minmax.
def forecast_accuracy(forecast, actual):
    forecast = np.array(forecast)  # Convert to NumPy array
    actual = np.array(actual)      # Convert to NumPy array

    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.column_stack([forecast, actual]), axis=1)
    maxs = np.amax(np.column_stack([forecast, actual]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})

forecastAccuracy = f'Metrics to evaluate the forecast accuracy:\n'
forecastAccuracy += f'______________________________________________\n'

print('Forecast Accuracy of: rgnp')
forecastAccuracy += f'Forecast Accuracy of: rgnp\n'
accuracy_prod = forecast_accuracy(df_results['rgnp_forecast'].values, df_test['rgnp'])
for k, v in accuracy_prod.items():
    print(k.rjust(10), ': ', round(v,4))
    forecastAccuracy += f'{k.rjust(10)} : {round(v,4)}\n'

print('\nForecast Accuracy of: pgnp')
forecastAccuracy += f'Forecast Accuracy of: pgnp\n'
accuracy_prod = forecast_accuracy(df_results['pgnp_forecast'].values, df_test['pgnp'])
for k, v in accuracy_prod.items():
    print(k.rjust(10), ': ', round(v,4))
    forecastAccuracy += f'{k.rjust(10)} : {round(v,4)}\n'

print('\nForecast Accuracy of: ulc')
forecastAccuracy += f'Forecast Accuracy of: ulc\n'
accuracy_prod = forecast_accuracy(df_results['ulc_forecast'].values, df_test['ulc'])
for k, v in accuracy_prod.items():
    print(k.rjust(10), ': ', round(v,4))
    forecastAccuracy += f'{k.rjust(10)} : {round(v,4)}\n'

print('\nForecast Accuracy of: gdfco')
forecastAccuracy += f'Forecast Accuracy of: gdfco\n'
accuracy_prod = forecast_accuracy(df_results['gdfco_forecast'].values, df_test['gdfco'])
for k, v in accuracy_prod.items():
    print(k.rjust(10), ': ', round(v,4))
    forecastAccuracy += f'{k.rjust(10)} : {round(v,4)}\n'

print('\nForecast Accuracy of: gdf')
forecastAccuracy += f'Forecast Accuracy of: gdf\n'
accuracy_prod = forecast_accuracy(df_results['gdf_forecast'].values, df_test['gdf'])
for k, v in accuracy_prod.items():
    print(k.rjust(10), ': ', round(v,4))
    forecastAccuracy += f'{k.rjust(10)} : {round(v,4)}\n'

print('\nForecast Accuracy of: gdfim')
forecastAccuracy += f'Forecast Accuracy of: gdfim\n'
accuracy_prod = forecast_accuracy(df_results['gdfim_forecast'].values, df_test['gdfim'])
for k, v in accuracy_prod.items():
    print(k.rjust(10), ': ', round(v,4))
    forecastAccuracy += f'{k.rjust(10)} : {round(v,4)}\n'

print('\nForecast Accuracy of: gdfcf')
forecastAccuracy += f'Forecast Accuracy of: gdfcf\n'
accuracy_prod = forecast_accuracy(df_results['gdfcf_forecast'].values, df_test['gdfcf'])
for k, v in accuracy_prod.items():
    print(k.rjust(10), ': ', round(v,4))
    forecastAccuracy += f'{k.rjust(10)} : {round(v,4)}\n'

print('\nForecast Accuracy of: gdfce')
forecastAccuracy += f'Forecast Accuracy of: gdfce\n'
accuracy_prod = forecast_accuracy(df_results['gdfce_forecast'].values, df_test['gdfce'])
for k, v in accuracy_prod.items():
    print(k.rjust(10), ': ', round(v,4))
    forecastAccuracy += f'{k.rjust(10)} : {round(v,4)}\n'

# Define the output file path
output_file_path9 = "forecastsAccuracy.txt"

# Save the Forecasts accuracy metrics to a text file
with open(output_file_path9, 'w') as file:
    file.write(forecastAccuracy)

# Print a message indicating the file has been saved
print(f"Output saved to {output_file_path9}")

