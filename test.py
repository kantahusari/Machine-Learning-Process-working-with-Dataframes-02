#01
myDF = pd.read_csv(r"00-Data/test1.csv")
myDF


# In[67]:


#02
myDF.info(memory_usage='deep')


# In[68]:


#03
myDF.memory_usage()


# In[69]:


#04
myDF['Continent'] = myDF['Continent'].astype('category')
myDF.dtypes


# In[70]:


#05
myDF.Continent.cat.codes


# In[71]:


#07
myDF = pd.read_csv(r"00-Data/test1.csv")
pd.get_dummies(myDF['Continent'])


# In[72]:


#08
mySchool = pd.read_csv(r"00-Data/test2.csv")


# In[73]:


#09
import numpy as np


# In[74]:


#09
percent = round(len(mySchool) * 0.3)


# In[75]:


#09
test, train = np.split(mySchool, [percent])


# In[76]:


#09
pd.DataFrame(train).set_index('ID').to_csv('training.csv')
pd.DataFrame(test).set_index('ID').to_csv('testing.csv')


# In[77]:


#10
feature_cols = ['Math', 'Physics']
X = train.loc[:, feature_cols]


# In[78]:


#11
Y = train.Result


# In[79]:


#12
from sklearn.linear_model import LogisticRegression


# In[80]:


#12
logReg = LogisticRegression(solver='lbfgs')
logReg.fit(X, Y)


# In[81]:


#13
Y_actual_test = test


# In[82]:


#14
testing = test.loc[:, feature_cols]
Y_pred_test = logReg.predict(testing)


# In[83]:


#15
Y_actual_test


# In[84]:


#15
Y_pred_test


# In[85]:


#16
from sklearn import tree


# In[86]:


#16
tree = tree.DecisionTreeClassifier()
tree.fit(X, Y)


# In[87]:


#16
testing = test.loc[:, feature_cols]
Y_pred_test = tree.predict(testing)


# In[88]:


#16
Y_pred_test


# In[89]:


#17
train.to_pickle('train.pkl')
test.to_pickle('test.pkl')


# In[90]:


#18
mySchool['Time'] = mySchool['Time'].astype('datetime64')
mySchool.dtypes


# In[91]:


#19
hour = mySchool.Time.dt.hour
year = mySchool.Time.dt.year


# In[92]:


#19
mySchool['Hour'] = hour
mySchool['Year'] = year


# In[93]:


#19
mySchool.head()


# In[94]:


#20
ts = '1/1/2015'
newMySchool = mySchool['Time'] >= ts
newMySchool = mySchool[newMySchool]


# In[95]:


#20
newMySchool.head()


# In[96]:


#21
newMySchool.Time.dt.year.value_counts()


# In[97]:


#22
mySchool.Math.duplicated(keep='last')


# In[98]:


#22
mySchool.Physics.duplicated(keep='last')


# In[99]:


#23
mySchool.Math.duplicated(keep=False)


# In[100]:


#23
mySchool.Physics.duplicated(keep=False)


# In[101]:


#27
rows = np.random.randint(0, 101, 12)
rows = np.split(rows, 3)
myDF = pd.DataFrame(rows, [10,20,30], ['Red', 'Green', 'Blue', 'Yellow'])


# In[102]:


#27
myDF


# In[103]:


#28
rows = np.random.randint(0, 101, 12).astype('float')
rows = np.split(rows, 3)
myDF = pd.DataFrame(rows, [10,20,30], ['Red', 'Green', 'Blue', 'Yellow'])


# In[104]:


#28
myDF


# In[105]:


#29
myDF.max(axis=1)


# In[106]:


#30
myDF.idxmax(axis='columns')




