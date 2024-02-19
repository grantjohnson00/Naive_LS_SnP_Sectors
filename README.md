# Naive_LS_SnP_Sectors
A relatively naive version of using a multi-factor model to construct a L/S portfolio out of S&amp;P500 sectors


This was done in the hopes of having a good "HF strategy pitch" at a nearby college. The rough thesis was that different forms of fundamental/price data could be engineered as features to predict cross-sectional returns between S&P500 sectors. I/we opted to use simple OLS regression to estimate coefficients; this was done mostly to make the findings more economically interpretable since the competition had focus on that. Once I/we realized that the OOS results had no chance of winning the competition, we dropped out. This text and posting of this project is mostly for myself: I want to reflect on the project, do similar bigger and better projects, and look back on this one day. 


Findings: generic economic/fundamental and price features for medium term(~20 day) cross-sectioanl return prediction using linear regression doesn't cut it. The features in this project and the model are too simple. I think the results are unsurprising: model assumptions were violated, OOS results were overfit, and there was just little predictive power. 

Attached picture shows OOS equity curve; the OOS sharpe is ~0.5. 



