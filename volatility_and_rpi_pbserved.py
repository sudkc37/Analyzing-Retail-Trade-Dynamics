#!/usr/bin/env python
# coding: utf-8

# In[2]:
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import time
from arch import arch_model
import gc
# In[5]:
trade_rpi = pd.read_csv('FINAL_RPI_508.csv.gz')
# In[6]:
final_trade = pd.read_csv('FINAL_TRADE_508.csv.gz')
# In[7]:
def volatility(df):
    df = df.groupby('SYM_ROOT')
    dic = {}
    for i, group in df:
        dic[i] = group.copy()
    for i, df in dic.items():
        model = arch_model(df['PRICE'].pct_change().dropna(), p=1,q=1, vol='GARCH', rescale=False).fit(disp='off')
        df['Volatility'] = model.conditional_volatility
        final_df = pd.concat(dic.values())
    return final_df
# In[8]:
def merger(df1, df2):
    try:
        [df.sort_values(by='time_PART', inplace=True) for df in [df1, df2]]
        new_df = pd.merge_asof(df1, df2, on='time_PART', by=['SYM_ROOT','EX'])
        return new_df
    except Exception as e:
        print("An error occurred:", e)
        return None
# In[9]:
file_vol = volatility(final_trade)
file_vol
# In[ ]:
# In[10]:
merged_file = merger(trade_rpi, file_vol)
# In[11]:
merged_file = merged_file.dropna(thresh=merged_file.shape[1]-1)
# In[12]:
merged_file
# In[13]:
gc.collect()
# In[14]:
#Trade happened at Subpenny or midquotes at 5 exchanges
def Subpenny_midquotes(df):
    exch = ['B', 'P', 'Y', 'V', 'N']
    potential_rlp_trade = df[df['EX'].isin(exch)]
    sub_midquotes = potential_rlp_trade[(potential_rlp_trade['Subpenny'] == 1) | (potential_rlp_trade['Sub_Mid'] == 1)] 
    return sub_midquotes
# In[15]:
#trade happend at 4 exchanges at subpenny
def subpenny(df):
    exch = ['B', 'P', 'Y', 'N']
    potential_rlp_trade = df[df['EX'].isin(exch)]
    subpenny = potential_rlp_trade[potential_rlp_trade['Subpenny'] == 1]  
    return subpenny
# In[16]:
#trade happened at IEX (RLP trade only happens as midquotes at IEX)
def iex_midquotes(df):
    potential_rlp_trade = df[df['EX']=='V']
    midquotes = potential_rlp_trade[potential_rlp_trade['Sub_Mid'] == 1]  
    return midquotes
# In[17]:
potential_trade_happned_rlp = Subpenny_midquotes(merged_file)
potential_trade_happned_rlp
# In[18]:
trade_happened_rlp_4 = subpenny(merged_file)
trade_happened_rlp_4 
# In[19]:
trade_happened_iex_mid= iex_midquotes(merged_file)
trade_happened_iex_mid
# In[20]:
aal = trade_happened_rlp_4[trade_happened_rlp_4['SYM_ROOT']=='AAL']
# In[21]:
aal
# In[22]:
aal[(aal['EX']=='B') & (aal['RPI']=='C')]
# In[23]:
c = trade_happened_rlp_4[trade_happened_rlp_4['EX']=='N']
c
# In[24]:
c[c['SYM_ROOT']=='SLB']

# Observed RPI LOgit
# In[25]:



spy_data = merged_file[merged_file['SYM_ROOT']=='SPY']
idx = spy_data.loc[spy_data['RPI']=='A'].index
spy_data.loc[idx, 'A'] = 1
spy_data['A'].fillna(0, inplace=True)
idx = spy_data.loc[spy_data['RPI']=='B'].index
spy_data.loc[idx, 'B'] = 1
spy_data['B'].fillna(0, inplace=True)
idx = spy_data.loc[spy_data['RPI']=='C'].index
spy_data.loc[idx, 'C'] = 1
spy_data['C'].fillna(0, inplace=True)

# In[26]:
from statsmodels.formula.api import logit
# In[27]:
form = 'A  ~ Volatility'
olsresults = logit(form, spy_data).fit()
print(olsresults.summary())

# In[28]:
form = 'B  ~ Volatility'
olsresults = logit(form, spy_data).fit()
print(olsresults.summary())
# In[29]:
form = 'C  ~ Volatility'
olsresults = logit(form, spy_data).fit()
print(olsresults.summary())


# In[30]:
aal_data = merged_file[merged_file['SYM_ROOT']=='AAL']
idx = aal_data.loc[aal_data['RPI']=='A'].index
aal_data.loc[idx, 'A'] = 1
aal_data['A'].fillna(0, inplace=True)
idx = aal_data.loc[aal_data['RPI']=='B'].index
aal_data.loc[idx, 'B'] = 1
aal_data['B'].fillna(0, inplace=True)
idx = aal_data.loc[aal_data['RPI']=='C'].index
aal_data.loc[idx, 'C'] = 1
aal_data['C'].fillna(0, inplace=True)

# In[31]:
form = 'A  ~ Volatility'
olsresults = logit(form, aal_data).fit()
print(olsresults.summary())

# In[32]:
form = 'B  ~ Volatility'
olsresults = logit(form, aal_data).fit()
print(olsresults.summary())

# In[33]:
form = 'C  ~ Volatility'
olsresults = logit(form, aal_data).fit()
print(olsresults.summary())

# In[34]:
c[c['SYM_ROOT']=='AAL']
# In[35]:

'''
types == exchange name

'''
def plots(df, types):
    fig,ax1=plt.subplots(figsize=(14,8))
    ax1.plot(df.index, df['PRICE'].values, color='olive')
    ax1.set_ylabel('Price', color='olive')
    ax1.tick_params(axis='y', labelcolor='olive')
    ax2=ax1.twinx()
    ax2.plot(df.index, df['Volatility'].values, color='red')
    ax2.set_ylabel('Volatility', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax1.scatter(df.loc[(df['EX']==types) & (df['RPI']=='C')].index, df.loc[(df['EX']==types) & (df['RPI']=='C')]['PRICE'].values, color='red', marker='o', alpha=0.5,s=90)
    ax1.scatter(df.loc[(df['EX']==types) & (df['RPI']=='A')].index, df.loc[(df['EX']==types) & (df['RPI']=='A')]['PRICE'].values, color='darkblue', marker='*', alpha=0.5,s=90)
    ax1.scatter(df.loc[(df['EX']==types) & (df['RPI']=='B')].index, df.loc[(df['EX']==types) & (df['RPI']=='B')]['PRICE'].values, color='cyan', marker='+', alpha=0.5,s=50)
    ax1.legend(['price','B = Quotes-Offer','A= Quotes-Bid','C= Two-sided'])
    ax1.set_xlabel('time')
    ax1.set_title('RLP Traded at exachange {}'.format(types))
    plt.show()
    
# In[93]:
aal_4 = trade_happened_rlp_4[(trade_happened_rlp_4['SYM_ROOT']=='AAL')]
spy_4 = trade_happened_rlp_4[trade_happened_rlp_4['SYM_ROOT']=='SPY']

# In[99]:
'''
only need to change trading ticker and exchange type ['B', 'P', 'Y', 'N']

Trade happened at suppenny increment! at day 1
'''

plots(trade_happened_rlp_4[(trade_happened_rlp_4['SYM_ROOT']=='AAL') & (trade_happened_rlp_4['Volatility'] <= 0.0009)],'B')

# In[98]:
plots(trade_happened_rlp_4[(trade_happened_rlp_4['SYM_ROOT']=='AAL') & (trade_happened_rlp_4['Volatility'] <= 0.0009)],'P')
# In[97]:
plots(trade_happened_rlp_4[(trade_happened_rlp_4['SYM_ROOT']=='AAL') & (trade_happened_rlp_4['Volatility'] <= 0.0009)],'Y')
# In[96]:
plots(trade_happened_rlp_4[(trade_happened_rlp_4['SYM_ROOT']=='AAL') & (trade_happened_rlp_4['Volatility'] <= 0.0009)],'N')
# In[41]:
plots(trade_happened_iex_mid[trade_happened_iex_mid['SYM_ROOT']=='AAL'],'V')
# In[ ]:
# In[42]:
plots(trade_happened_rlp_4[(trade_happened_rlp_4['SYM_ROOT']=='SPY') & (trade_happened_rlp_4['Volatility'] <= 0.00009)],'B')
# In[43]:
plots(trade_happened_rlp_4[(trade_happened_rlp_4['SYM_ROOT']=='SPY') & (trade_happened_rlp_4['Volatility'] <= 0.00009)],'P')
# In[44]:
plots(trade_happened_rlp_4[(trade_happened_rlp_4['SYM_ROOT']=='SPY') & (trade_happened_rlp_4['Volatility'] <= 0.00009)],'Y')
# In[45]:
plots(trade_happened_rlp_4[(trade_happened_rlp_4['SYM_ROOT']=='SPY') & (trade_happened_rlp_4['Volatility'] <= 0.00009)],'N')
# In[46]:


'''
only need to change trading ticker and exchange remains constant

Trade happened at mid-quotes ! at day 1
'''

plots(trade_happened_iex_mid[(trade_happened_iex_mid['SYM_ROOT']=='SPY') & (trade_happened_iex_mid['Volatility']<=0.00009)],'V')
# In[47]:
q = trade_happened_rlp_4[(trade_happened_rlp_4['SYM_ROOT']=='SPY') & (trade_happened_rlp_4['EX']=='B')]
q
# In[48]:
q = q[q['Volatility'] < 0.00094]
# In[49]:
q['Volatility'].describe()

# Correlations:

# # RPI Traded

# In[50]:



trade_happened_rlp_4 


# In[51]:
data = trade_happened_rlp_4[trade_happened_rlp_4['SYM_ROOT']=='SPY']
# In[52]:
data 


# In[53]:
idx = data.loc[data['RPI']=='A'].index
data.loc[idx, 'A'] = 1
data['A'].fillna(0, inplace=True)

# In[54]:
idx = data.loc[data['RPI']=='B'].index
data.loc[idx, 'B'] = 1
data['B'].fillna(0, inplace=True)

# In[55]:
idx = data.loc[data['RPI']=='C'].index
data.loc[idx, 'C'] = 1
data['C'].fillna(0, inplace=True)

# In[ ]:
# In[ ]:
# 

# In[ ]:


# In[ ]:
# In[56]:


import statsmodels.discrete.discrete_model as discrete
# In[57]:
spy = trade_happened_rlp_4[trade_happened_rlp_4['SYM_ROOT']=='SPY']
# In[58]:
spy['RPI'].value_counts()
# In[59]:
results = discrete.MNLogit(spy['RPI'], data['Volatility']).fit()
print(results.summary())
# In[ ]:
# In[ ]:
# In[60]:
form = 'A  ~ Volatility'
olsresults = logit(form, data).fit()
print(olsresults.summary())

# In[61]:
form = 'B  ~ Volatility'
olsresults = logit(form, data).fit()
print(olsresults.summary())

# In[62]:
form = 'C  ~ Volatility'
olsresults = logit(form, data).fit()
print(olsresults.summary())


# In[63]:
aal = trade_happened_rlp_4[trade_happened_rlp_4['SYM_ROOT']=='AAL']
idx = aal.loc[aal['RPI']=='A'].index
aal.loc[idx, 'A'] = 1
aal['A'].fillna(0, inplace=True)
idx = aal.loc[aal['RPI']=='B'].index
aal.loc[idx, 'B'] = 1
aal['B'].fillna(0, inplace=True)
idx = aal.loc[aal['RPI']=='C'].index
aal.loc[idx, 'C'] = 1
aal['C'].fillna(0, inplace=True)

# In[64]:
form = 'A  ~ Volatility'
olsresults = logit(form, aal).fit()
print(olsresults.summary())


# In[65]:
form = 'B  ~ Volatility'
olsresults = logit(form, aal).fit()
print(olsresults.summary())


# In[66]:
form = 'C  ~ Volatility'
olsresults = logit(form, aal).fit()
print(olsresults.summary())


# In[67]:
merged_file
rpi_observed_B = merged_file[merged_file['EX']=='B']
rpi_observed_P = merged_file[merged_file['EX']=='P']
rpi_observed_Y = merged_file[merged_file['EX']=='Y']
rpi_observed_N = merged_file[merged_file['EX']=='N']
rpi_observed_V = merged_file[merged_file['EX']=='V']
# In[68]:
rpi_observed_N
# In[69]:
def plots_observed(df, types):
    fig,ax1=plt.subplots(figsize=(14,8))
    ax1.plot(df.index, df['PRICE'].values, color='olive')
    ax1.set_ylabel('Price', color='olive')
    ax1.tick_params(axis='y', labelcolor='olive')
    ax1.scatter(df.loc[(df['EX']==types) & (df['RPI']=='C')].index, df.loc[(df['EX']==types) & (df['RPI']=='C')]['PRICE'].values, color='red', marker='o', alpha=0.5,s=90)
    ax1.scatter(df.loc[(df['EX']==types) & (df['RPI']=='A')].index, df.loc[(df['EX']==types) & (df['RPI']=='A')]['PRICE'].values, color='darkblue', marker='*', alpha=0.5,s=90)
    ax1.scatter(df.loc[(df['EX']==types) & (df['RPI']=='B')].index, df.loc[(df['EX']==types) & (df['RPI']=='B')]['PRICE'].values, color='cyan', marker='+', alpha=0.5,s=50)
    ax1.legend(['price','B = Quotes-Offer','A= Quotes-Bid','C= Two-sided'])
    ax1.set_xlabel('time')  
    ax1.set_title('RLP observed at exachange {}'.format(types))
    plt.show()


# In[ ]:
# In[70]:
plots_observed(merged_file[(merged_file['SYM_ROOT']=='SPY') & (merged_file['Volatility'] <= 0.00009)],'B')
# In[71]:
plots_observed(merged_file[(merged_file['SYM_ROOT']=='SPY') & (merged_file['Volatility'] <= 0.00009)],'P')
# In[72]:
plots_observed(merged_file[(merged_file['SYM_ROOT']=='SPY') & (merged_file['Volatility'] <= 0.00009)],'Y')

# In[73]:
plots_observed(merged_file[(merged_file['SYM_ROOT']=='SPY') & (merged_file['Volatility'] <= 0.00009)],'N')
# In[74]:
rip_observed_spy_B = len(merged_file[(merged_file['SYM_ROOT']=='SPY') & (merged_file['EX']=='B')]['RPI'])
rip_observed_spy_P = len(merged_file[(merged_file['SYM_ROOT']=='SPY') & (merged_file['EX']=='P')]['RPI'])
rip_observed_spy_Y = len(merged_file[(merged_file['SYM_ROOT']=='SPY') & (merged_file['EX']=='Y')]['RPI'])
rip_observed_spy_N = len(merged_file[(merged_file['SYM_ROOT']=='SPY') & (merged_file['EX']=='N')]['RPI'])
rip_observed_spy_V = len(merged_file[(merged_file['SYM_ROOT']=='SPY') & (merged_file['EX']=='V')]['RPI'])
# In[75]:
observed_rpi_spy_df = pd.DataFrame([rip_observed_spy_B, rip_observed_spy_P,rip_observed_spy_Y,
                              rip_observed_spy_N, rip_observed_spy_V], index=['B','P','Y','N','V'], columns=['RPI Observed "SPY"'])
# In[76]:
rip_traded_spy_B = len(trade_happened_rlp_4[(trade_happened_rlp_4['SYM_ROOT']=='SPY') & (trade_happened_rlp_4['EX']=='B')]['RPI'])
rip_traded_spy_P = len(trade_happened_rlp_4[(trade_happened_rlp_4['SYM_ROOT']=='SPY') & (trade_happened_rlp_4['EX']=='P')]['RPI'])
rip_traded_spy_Y = len(trade_happened_rlp_4[(trade_happened_rlp_4['SYM_ROOT']=='SPY') & (trade_happened_rlp_4['EX']=='Y')]['RPI'])
rip_traded_spy_N = len(trade_happened_rlp_4[(trade_happened_rlp_4['SYM_ROOT']=='SPY') & (trade_happened_rlp_4['EX']=='N')]['RPI'])
rip_traded_spy_V = len(trade_happened_iex_mid[(trade_happened_iex_mid['SYM_ROOT']=='SPY') & (trade_happened_iex_mid['EX']=='V')]['RPI'])


# In[77]:
traded_rpi_spy_df = pd.DataFrame([rip_traded_spy_B, rip_traded_spy_P,rip_traded_spy_Y,
                              rip_traded_spy_N, rip_traded_spy_V], index=['B','P','Y','N','V'], columns=['RPI Traded "SPY"'])
# In[78]:
s = pd.concat([observed_rpi_spy_df,traded_rpi_spy_df], axis=1)
s

# In[79]:
plt.figure(figsize=(10,6))
plt.bar(s.index, s['RPI Observed "SPY"'])
plt.bar(s.index, s['RPI Traded "SPY"'])
plt.xlabel('Exchanges')
plt.ylabel('RPIs')
plt.title('SPY RIP observed Vs Traded')
plt.legend(['observed','Traded'])
plt.show()


# In[80]:
plots_observed(merged_file[merged_file['SYM_ROOT']=='AAL'],'B')
# In[81]:
plots_observed(merged_file[merged_file['SYM_ROOT']=='AAL'],'P')
# In[82]:
plots_observed(merged_file[merged_file['SYM_ROOT']=='AAL'],'Y')
# In[83]:
plots_observed(merged_file[merged_file['SYM_ROOT']=='AAL'],'N')
# In[84]:
plots_observed(merged_file[merged_file['SYM_ROOT']=='AAL'],'V')


# In[85]:
rip_observed_aal_B = len(merged_file[(merged_file['SYM_ROOT']=='AAL') & (merged_file['EX']=='B')]['RPI'])
rip_observed_aal_P = len(merged_file[(merged_file['SYM_ROOT']=='AAL') & (merged_file['EX']=='P')]['RPI'])
rip_observed_aal_Y = len(merged_file[(merged_file['SYM_ROOT']=='AAL') & (merged_file['EX']=='Y')]['RPI'])
rip_observed_aal_N = len(merged_file[(merged_file['SYM_ROOT']=='AAL') & (merged_file['EX']=='N')]['RPI'])
rip_observed_aal_V = len(merged_file[(merged_file['SYM_ROOT']=='AAL') & (merged_file['EX']=='V')]['RPI'])
# In[86]:
observed_rpi_aal_df = pd.DataFrame([rip_observed_aal_B, rip_observed_aal_P,rip_observed_aal_Y,
                              rip_observed_aal_N, rip_observed_aal_V], index=['B','P','Y','N','V'], columns=['RPI Observed "AAL"'])

# In[87]:
rip_traded_aal_B = len(trade_happened_rlp_4[(trade_happened_rlp_4['SYM_ROOT']=='AAL') & (trade_happened_rlp_4['EX']=='B')]['RPI'])
rip_traded_aal_P = len(trade_happened_rlp_4[(trade_happened_rlp_4['SYM_ROOT']=='AAL') & (trade_happened_rlp_4['EX']=='P')]['RPI'])
rip_traded_aal_Y = len(trade_happened_rlp_4[(trade_happened_rlp_4['SYM_ROOT']=='AAL') & (trade_happened_rlp_4['EX']=='Y')]['RPI'])
rip_traded_aal_N = len(trade_happened_rlp_4[(trade_happened_rlp_4['SYM_ROOT']=='AAL') & (trade_happened_rlp_4['EX']=='N')]['RPI'])
rip_traded_aal_V = len(trade_happened_iex_mid[(trade_happened_iex_mid['SYM_ROOT']=='AAL') & (trade_happened_iex_mid['EX']=='V')]['RPI'])


# In[88]:
traded_rpi_aal_df = pd.DataFrame([rip_traded_aal_B, rip_traded_aal_P,rip_traded_aal_Y,
                              rip_traded_aal_N, rip_traded_aal_V], index=['B','P','Y','N','V'], columns=['RPI Traded "AAL"'])
# In[89]:
d = pd.concat([observed_rpi_aal_df,traded_rpi_aal_df], axis=1)
d
# In[ ]:
# In[90]:
plt.figure(figsize=(10,6))
plt.bar(d.index, d['RPI Observed "AAL"'])
plt.bar(d.index, d['RPI Traded "AAL"'])
plt.xlabel('Exchanges')
plt.ylabel('RPIs')
plt.title('AAL RIP observed Vs Traded')
plt.legend(['observed','Traded'])
plt.show()
