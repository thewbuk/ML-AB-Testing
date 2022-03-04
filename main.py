#!/usr/bin/env python
# coding: utf-8

# ## Loading Libs

# In[1]:


from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats import power as pwr
from scipy.stats import norm
import statsmodels.api as sm
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
random.seed(55)


# ## Reading Data

# In[2]:


df = pd.read_csv('ab_data.csv')


# In[4]:


df.info()


# In[5]:


df.head()


# In[6]:


df.groupby(['group', 'converted']).agg('count')


# In[7]:


df.drop(df.query("group == 'control' and landing_page == 'new_page'").index, inplace=True)
df.drop(df.query(
    "group == 'treatment' and landing_page == 'old_page'").index, inplace=True)

df.groupby(['group', 'converted']).agg('count')


# In[8]:


df.info()


# ## Cleaning Duplicated Users

# In[9]:


df[df.duplicated(['user_id'], keep=False)]


# In[10]:


df.drop_duplicates(['user_id'], inplace=True)


# In[11]:


assert len(df['user_id'].unique()) == df['user_id'].size


# In[12]:


df.info()


# In[13]:


df['converted'].mean()


# In[14]:


df.groupby(['group']).describe()


# In[15]:


df.groupby(['group']).agg({'converted': ['sum', 'count', 'mean']})


# In[16]:


df[['group', 'converted']].groupby(['group']).agg('mean').T


# ## Calculate Prob. of New and Old Pages Respectively

# In[17]:


p_old_page = df[['group', 'converted']].query(
    "group == 'control'")['converted'].mean()
p_new_page = df[['group', 'converted']].query(
    "group == 'treatment'")['converted'].mean()
act_p_diff = p_new_page - p_old_page

print('p_old_page:\t{}\np_new_page:\t{}\np_diff:\t\t{}'.format(
    p_old_page, p_new_page, act_p_diff))


# ## Calculate Counts of New and Old Pages Respectively

# In[18]:


n_old = len(df[['group']].query("group == 'control'"))
n_new = len(df[['group']].query("group == 'treatment'"))

print('n_old:\t{}\nn_new:\t{}'.format(n_old, n_new))


# ## Simulating Randomly With Respect to Probs.
#
# In this step, to generalize and collect concrete information about the landing page with new feature, we simulate an experiment N (10000) times with respect to control and treatment's probs.
#
# According to this experiment, we will find that new feature does not have any impact to over user in case it repeats N times.
# Because roughly more than half of repating experiments has lower diff_probs than actual diff_prob (`act_p_diff = -0.0015782389853555567`) in terms of probs of two sampling groups. `(act_p_diff < p_diffs).mean() = 0.5107`
#
# Our expectation might be same with that result. So, We will conduct AB test to claim that new feature does not have any impact on user.

# In[19]:


p_diffs = []

for _ in range(10000):
    new_page_converted = np.random.choice(
        [1, 0], size=n_new, p=[p_new_page, (1 - p_new_page)]).mean()
    old_page_converted = np.random.choice(
        [1, 0], size=n_old, p=[p_old_page, (1 - p_old_page)]).mean()
    diff = new_page_converted - old_page_converted
    p_diffs.append(diff)


# In[20]:


plt.hist(p_diffs)
plt.xlabel('p_diffs')
plt.ylabel('Frequency')
plt.title('Plot of 10K simulated p_diffs')


# In[21]:


p_diffs = np.array(p_diffs)
(act_p_diff < p_diffs).mean()


# # A/B Testing

# ## A/B Testing: Calculate Z critical score and p_value (area) Over Dataset

# In[26]:


convert_old = sum(df.query("group == 'control'")['converted'])
convert_new = sum(df.query("group == 'treatment'")['converted'])

z_score, p_value = sm.stats.proportions_ztest(
    [convert_old, convert_new], [n_old, n_new], alternative='smaller')
print('z_critical_value: ', z_score)
print('p_critical_value: ', p_value)


# ## A/B Testing: Calculate Z score corresponding to Alpha (0.05)

# In[23]:


print('p-value: ', norm.cdf(z_score))
# Tells us how significant our z-score is

# for our single-sides test, assumed at 95% confidence level, we calculate:
print('z_alfa: ', norm.ppf(1 - (0.05)))


# ## A/B Testing: Calculate Beta and Power corresponding to Alpha (0.05), Effect Size

# In[25]:


# sm.stats.zt_ind_solve_power(effect_size=-0.0048, alpha=0.05, power=0.1, alternative='smaller')

es = proportion_effectsize(p_new_page, p_old_page)
ratio = (n_new / n_old)
power = pwr.NormalIndPower().power(es, n_old / ratio, alpha=0.05,
                                   ratio=ratio, alternative='smaller')
beta = 1 - power

print('power: ', power)
print('beta: ', beta)
