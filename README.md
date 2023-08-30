# Analyzing-Retail-Trade-Dynamics
The Retail Liquidity Program (RLP) is an initiative introduced by the U.S. Securities and Exchange Commission (SEC) to enhance market liquidity and improve trading conditions for retail investors. The program aims to provide retail investors with access to better execution prices by encouraging market makers to provide price improvement on retail orders. By fostering competition and incentivizing improved execution quality, the RLP seeks to empower retail investors and promote a fair and efficient marketplace. Through this program, retail investors can potentially benefit from increased liquidity and the opportunity to obtain more favorable prices for their trades.


Explore the dynamics of retail trading through observed RPI and traded RIP across 5 exchanges. Discover temporal trading patterns, unveil concealed cyclical trends, and gather insights that influence the retail sector.
Concentrated on each exchange’s observations and trading of each form of RPI with regard to price, time, and volatility. I used sub-penny increments or mid-quotes as an indicator and filtered data accordingly to determine that the trade truly accrued at RPI. Trades on the exchanges for the Bats(Y), NASDAQ(B), NYSE Arca(P), and NYSE(N) occur in sub-penny increments. Additionally, trade only takes place at mid-quotes on the IEX. 
Investigated two categories of assets: ETSs and individual stock. I selected SPY as a ETF and American Airlines (AAL) as a stock. 
ETFs are the most RPI traded stocks, it is difficult to discern RPI kinds only by looking at these graphs. But if we look closely, the majority of observed RPI are two-sided. It is difficult to determine whether volatility or time, or jointly have any impact on the RPIs, unless there are some exceptions (unusual trade which we were lucky to observe on further analysis). Comparatively, the two-sided RPI is evident in a variety of ways. 
Use of Binary regression model like logestic regression to discover relation between volatility, with several RPI's
Mathematically, 
(a/1-a) = 𝛃0 + 𝛃(volatility) , where a is the probability of A observed and 1-a otherwise. 
(b/1-b) = 𝛃0 + 𝛃(volatility) , where a is the probability of B observed and 1-b otherwise.
(c/1-c) = 𝛃0 + 𝛃(volatility) , where a is the probability of C observed and 1-c otherwise
