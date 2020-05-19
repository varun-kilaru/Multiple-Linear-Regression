# Multiple-Linear-Regression

<h2>A multiple linear regression model build to predict house prices based on various features such as area, bedrooms, bathrooms, airconditioning etc.</h2>
<h3>The significant features are selected using Recursive Feature Selection and also using Variance Inflation Factor & p-values</h3>
<h3>Data :</h3>
<pre>
   price  area  bedrooms  bathrooms  stories  ... hotwaterheating airconditioning parking prefarea furnishingstatus
0  13300000  7420         4          2        3  ...              no             yes       2      yes        furnished
1  12250000  8960         4          4        4  ...              no             yes       3       no        furnished
2  12250000  9960         3          2        2  ...              no              no       2      yes   semi-furnished
3  12215000  7500         4          2        2  ...              no             yes       3      yes        furnished
4  11410000  7420         4          1        2  ...              no             yes       2       no        furnished
</pre>
<img src="SLR/Figure_1.png">
<img src="SLR/Figure_2.png">
<h3>Summary of the model :</h3> 
<pre>
 OLS Regression Results
==============================================================================
Dep. Variable:                  price   R-squared:                       0.666
Model:                            OLS   Adj. R-squared:                  0.658
Method:                 Least Squares   F-statistic:                     82.37
Date:                Tue, 19 May 2020   Prob (F-statistic):           6.67e-83
Time:                        16:32:45   Log-Likelihood:                 373.00
No. Observations:                 381   AIC:                            -726.0
Df Residuals:                     371   BIC:                            -686.6
Df Model:                           9
Covariance Type:            nonrobust
===================================================================================
                      coef    std err          t      P>|t|      [0.025      0.975]
-----------------------------------------------------------------------------------
const               0.0242      0.013      1.794      0.074      -0.002       0.051
area                0.2367      0.030      7.779      0.000       0.177       0.297
bathrooms           0.2070      0.022      9.537      0.000       0.164       0.250
stories             0.1096      0.017      6.280      0.000       0.075       0.144
mainroad            0.0536      0.014      3.710      0.000       0.025       0.082
guestroom           0.0390      0.013      2.991      0.003       0.013       0.065
hotwaterheating     0.0921      0.022      4.213      0.000       0.049       0.135
airconditioning     0.0710      0.011      6.212      0.000       0.049       0.094
parking             0.0669      0.018      3.665      0.000       0.031       0.103
prefarea            0.0653      0.012      5.513      0.000       0.042       0.089
==============================================================================
Omnibus:                       91.542   Durbin-Watson:                   2.107
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              315.402
Skew:                           1.044   Prob(JB):                     3.25e-69
Kurtosis:                       6.938   Cond. No.                         10.0
==============================================================================
</pre>
<h3>R^2 Value:</h3>
<pre>
r2 score on train data :  0.666457060116814

r2 score on test data :  0.6481740917926486
</pre>

