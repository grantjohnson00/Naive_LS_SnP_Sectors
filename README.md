# Naive_LS_SnP_Sectors
A relatively naive version of using a multi-factor model to construct a L/S portfolio out of S&amp;P500 sectors


This was done in the hopes of having a good "HF strategy pitch" at a nearby college. The rough thesis was that different forms of fundamental/price data could be engineered as features to predict cross-sectional returns between S&P500 sectors. I/we opted to use simple OLS regression to estimate coefficients; this was done mostly to make the findings more economically interpretable since the competition had an emphasis on that. Once I/we realized that the OOS results had no chance of winning the competition, we dropped out. This text and posting of this project is mostly for myself: I want to reflect on the project, do similar bigger and better projects, and look back on this one day. 


Findings: generic economic/fundamental and price features for medium term(~20 day) cross-sectional return prediction using linear regression doesn't cut it. Both the features and the model used in this project are too simple. I think the results are unsurprising: model assumptions were violated, coefs going into OOS were overfit(though I perform rolling data truncation to refit the model on more recent data before each portfolio rebalance), and, overall, there was just little predictive power. 


I plan to pursue the portfolio optimization component of future related projects once I can more thoroughly predict returns.


Attached picture shows OOS equity curve; the OOS sharpe is ~0.5. 



