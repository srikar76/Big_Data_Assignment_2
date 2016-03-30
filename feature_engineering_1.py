
# coding: utf-8

# In[8]:

#get_ipython().magic(u'matplotlib inline')
import cv2
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from collections import Counter
from scipy.stats.mstats import mode
import sklearn
from sklearn.naive_bayes import MultinomialNB
from math import sqrt
import scipy.stats as st


# In[9]:

#function to unpickle the dataset
def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


# In[10]:

#storing the unpickled data into a dictionary called my_data
my_data = {'data': unpickle("data_batch_1")['data'], 'labels': unpickle("data_batch_1")['labels']}
for each in range(2, 6):
    my_data['data'] = np.concatenate((my_data['data'], unpickle("data_batch_" + str(each))['data']), axis=0)
    my_data['labels'] = np.concatenate((my_data['labels'], unpickle("data_batch_" + str(each))['labels']), axis=0)
print my_data

# In[11]:

# assigning the data and labels to respective lists
data = my_data['data']
labels = my_data['labels']
label_names = unpickle("batches.meta")['label_names']
print data

# In[12]:

# assigning the data and labels to respective lists
X_train = my_data['data']
y_train = my_data['labels']
label_names = unpickle("batches.meta")['label_names']


# In[13]:

# unpickling the test data and storing in a dictionary called test_data
test_data = {}
test_data.update(unpickle("test_batch"))
X_test = test_data['data']
y_test = test_data['labels']


# In[14]:

#plotting the feature points
x1 = []
y1 = []
z1 = []
for item in X_train:
 x1.append(item[0])
 y1.append(item[1])
 z1.append(item[2])
fig1 = plt.figure() 
ax = Axes3D(fig1) 
pltData = [x1,y1,z1] 
ax.scatter(pltData[0], pltData[1], pltData[2], 'bo') 


xLine = ((min(pltData[0]), max(pltData[0])), (0, 0), (0,0)) 
ax.plot(xLine[0], xLine[1], xLine[2], 'r') 
yLine = ((0, 0), (min(pltData[1]), max(pltData[1])), (0,0)) 
ax.plot(yLine[0], yLine[1], yLine[2], 'r')
zLine = ((0, 0), (0,0), (min(pltData[2]), max(pltData[2])))
ax.plot(zLine[0], zLine[1], zLine[2], 'r') 
 
 
ax.set_xlabel("x-axis") 
ax.set_ylabel("y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("CIFAR-10 PCA")
plt.show()


# In[15]:

#performs canny edge detection
edges ={}
edges = cv2.Canny(data,100,200)
print edges


# In[16]:

# Initiate STAR detector
orb = cv2.ORB()

# find the keypoints with ORB
kp = orb.detect(data,None)

# compute the descriptors with ORB
kp, des = orb.compute(data, kp)
kp
print des


# In[17]:

#computing the principle components
X = np.array(X_train)
pca = PCA(n_components=9)
abc = pca.fit_transform(X)
pca1 = pca.score(X)
print abc,pca1


# In[18]:

#plotting the principle components
x1 = []
y1 = []
z1 = []
for item in abc:
 x1.append(item[0])
 y1.append(item[1])
 z1.append(item[2])
fig1 = plt.figure() 
ax = Axes3D(fig1) 
pltData = [x1,y1,z1] 
ax.scatter(pltData[0], pltData[1], pltData[2], 'bo') 
xLine = ((min(pltData[0]), max(pltData[0])), (0, 0), (0,0)) 
ax.plot(xLine[0], xLine[1], xLine[2], 'r') 
yLine = ((0, 0), (min(pltData[1]), max(pltData[1])), (0,0)) 
ax.plot(yLine[0], yLine[1], yLine[2], 'r')
zLine = ((0, 0), (0,0), (min(pltData[2]), max(pltData[2])))
ax.plot(zLine[0], zLine[1], zLine[2], 'r')  
ax.set_xlabel("x-axis") 
ax.set_ylabel("y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("CIFAR-10 PCA")
plt.show()


# In[19]:

#creating dictionaries of each class of images in training set
class_data = {'airplane': [], 'automobile': [],'bird':[],'cat':[], 'deer':[], 'dog':[],'frog':[],'horse':[], 'ship':[], 'truck':[]}


# In[20]:

#creating dictionaries of each class of images in test set
class_data_test = {'airplane': [], 'automobile': [],'bird':[],'cat':[], 'deer':[], 'dog':[],'frog':[],'horse':[], 'ship':[], 'truck':[]}


# In[21]:

# appending each class images in the training set based on the labels
for each in range(0, 50000):
    if y_train[each] == 0:
        class_data['airplane'].append(X_train[each])
    if y_train[each] == 1:
        class_data['automobile'].append(X_train[each])
    if y_train[each] == 2:
        class_data['bird'].append(X_train[each])
    if y_train[each] == 3:
        class_data['cat'].append(X_train[each])
    if y_train[each] == 4:    
        class_data['deer'].append(X_train[each])
    if y_train[each] == 5:     
        class_data['dog'].append(X_train[each])
    if y_train[each] == 6:    
        class_data['frog'].append(X_train[each])
    if y_train[each] == 7:    
        class_data['horse'].append(X_train[each])
    if y_train[each] == 8:    
        class_data['ship'].append(X_train[each])
    if y_train[each] == 9:    
        class_data['truck'].append(X_train[each])


# In[22]:

# appending each class images in the test set based on the labels
for each in range(0, 10000):
    if y_test[each] == 0:
        class_data_test['airplane'].append(X_test[each])
    if y_test[each] == 1:
        class_data_test['automobile'].append(X_test[each])
    if y_test[each] == 2:
        class_data_test['bird'].append(X_test[each])
    if y_test[each] == 3:
        class_data_test['cat'].append(X_test[each])
    if y_test[each] == 4:    
        class_data_test['deer'].append(X_test[each])
    if y_test[each] == 5:     
        class_data_test['dog'].append(X_test[each])
    if y_test[each] == 6:    
        class_data_test['frog'].append(X_test[each])
    if y_test[each] == 7:    
        class_data_test['horse'].append(X_test[each])
    if y_test[each] == 8:    
        class_data_test['ship'].append(X_test[each])
    if y_test[each] == 9:    
        class_data_test['truck'].append(X_test[each])


# In[23]:

# function to calculate mean ,variance, and standard deviation for each class of images
def get_stats(class_name):
    mean = 0
    variance = 0
    std_deviation = 0
    size = len(class_data[class_name])
    for each in class_data[class_name]:
        mean += each.mean()
        variance += each.var()
        std_deviation += each.std()
    mean = mean / size
    variance = variance / size
    std_deviation = std_deviation / size
    return mean, variance, std_deviation, size


# In[24]:

print get_stats('airplane')
print get_stats('automobile')
print get_stats('bird')
print get_stats('cat')
print get_stats('deer')
print get_stats('dog')
print get_stats('frog')
print get_stats('horse')
print get_stats('ship')
print get_stats('truck')


# In[25]:

# function to calculate the t-value for the classes of images that are being compared
def tstat(class_name1, class_name2):
    X1 = get_stats(class_name1)[0]
    X2 = get_stats(class_name2)[0]
    S1 = get_stats(class_name1)[1]
    S2 = get_stats(class_name2)[1]
    N1=5000
    N2=5000
    t1= X1-X2
    t2= sqrt((S1/N1+S2/N2))
    t11=t1/t2
    S1 = get_stats(class_name1)[1]
    S2 = get_stats(class_name2)[1]
    sum1 =((get_stats(class_name1)[1]+get_stats(class_name2)[1]))*((get_stats(class_name1)[1]+get_stats(class_name2)[1]))
    s14 =(get_stats(class_name1)[1]*get_stats(class_name1)[1])
    s24=(get_stats(class_name2)[1]*get_stats(class_name2)[1])
    n1 = 5000*5000
    v1= 4999
    n2 = 5000*5000
    v2=4999
    sum2=  (s14/n1*v1 + s24/n2*v2)
    df1 = (sum1/sum2)
    df= df1/10000
    p_value = 2*st.t.cdf(-np.abs(t11),df)
    if p_value<= 0.5:
        print(False)
    else:
        print (True)
    return t11, df, p_value


# In[26]:

print tstat('airplane','deer')


# In[27]:

#performing k-means clustering with 10 clusters each for each class of images
k_means = KMeans(n_clusters=10, n_jobs=8)
k_means.fit(X_train)


# In[28]:

k_means.predict(class_data_test['airplane'])


# In[29]:

# function to give the respective number of class of images in all the clusters
def get_count(class_name):
    return Counter(k_means.predict(class_data_test[class_name]))


# In[30]:

print get_count('truck')


# In[31]:

# function to get the probability of a particular class of images with the cluster number that it is present 
from collections import Counter
def get_prob(class_name):
    c = Counter(k_means.predict(class_data_test['truck']))
    clusternumber = c.most_common(1)[0][0]
    val = (c.most_common(1)[0][1])
    return clusternumber, val


# In[32]:

print get_prob('truck')


# In[33]:

# performing multinomial logistic regression to find out the accuracy of the classification of images in the training set with the testing set
model = LogisticRegression(solver = "lbfgs", multi_class = "multinomial")
model = model.fit(X_train, y_train)
print model.score(X_test, y_test) 

