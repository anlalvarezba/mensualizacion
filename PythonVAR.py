import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

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
plt.savefig("filename.png")

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

    # Print Summary
    # print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    # print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    # print(f' Significance Level    = {signif}')
    # print(f' Test Statistic        = {output["test_statistic"]}')
    # print(f' No. Lags Chosen       = {output["n_lags"]}')

    # Create a summary string
    summary = f'Augmented Dickey-Fuller Test on "{name}"\n'
    summary += ' ' + '-'*47 + '\n'
    summary += f'Null Hypothesis: Data has unit root. Non-Stationary.\n'
    summary += f'Significance Level    = {signif}\n'
    summary += f'Test Statistic        = {output["test_statistic"]}\n'
    summary += f'No. Lags Chosen       = {output["n_lags"]}\n'


    for key,val in r[4].items():
        # print(f' Critical value {adjust(key)} = {round(val, 3)}')
        summary += f'Critical value {adjust(key)} = {round(val, 3)}\n'

    if p_value <= signif:
        # print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        # print(f" => Series is Stationary.")
        summary += f"=> P-Value = {p_value}. Rejecting Null Hypothesis.\n"
        summary += f"=> Series is Stationary.\n"
    else:
        # print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        # print(f" => Series is Non-Stationary.") 
        summary += f"=> P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.\n"
        summary += f"=> Series is Non-Stationary.\n"
        summary += f"\n"

    return summary


# ADF Test on each column
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
