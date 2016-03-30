

# coding: utf-8

# In[1]:

#get_ipython().magic(u'matplotlib inline')

import cPickle
import sklearn
import numpy as np
from math import sqrt
from scipy import stats
import scipy.stats as st
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats.mstats import mode
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression

# In[2]:

#function to unpickle the dataset
def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


# In[28]:

#storing the unpickled data into a dictionary called my_data
my_data = {'data': unpickle("data_batch_1")['data'], 'labels': unpickle("data_batch_1")['labels']}
for each in range(2, 6):
    my_data['data'] = np.concatenate((my_data['data'], unpickle("data_batch_" + str(each))['data']), axis=0)
    my_data['labels'] = np.concatenate((my_data['labels'], unpickle("data_batch_" + str(each))['labels']), axis=0)
print "Data,   Lables,   Lable Names"
print my_data


# In[4]:

# assigning the data and labels to respective lists
data = my_data['data']
labels = my_data['labels']
label_names = unpickle("batches.meta")['label_names']
print " Image  Data"
print data


# In[29]:

# unpickling the test data and storing in a dictionary called test_data
test_data_dict = {}
test_data_dict.update(unpickle("test_batch"))
test_data = test_data_dict['data']
X_test = []
y_test = test_data_dict['labels']
print "Test Data Dictionary"
print test_data_dict


# In[6]:

#creating two empty lists for storing feature vectors
X_train = []
y_train = my_data['labels']


# In[7]:

#plotting the feature points
x1 = []
y1 = []
z1 = []
for item in data:
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


# In[8]:

# calculating the feature vectors by extracting features such as average intensity of all pixels and max & min of pixels(training set)
# segmenting image into 4 parts
for each in range(0, 50000):
    p1dr = data[each][0:256]
    p1dg = data[each][1024:1280]
    p1db = data[each][2048:2304]
    p2dr = data[each][256:512]
    p2dg = data[each][1280:1536]
    p2db = data[each][2304:2560]
    p3dr = data[each][512:768]
    p3dg = data[each][1536:1792]
    p3db = data[each][2560:2816]
    p4dr = data[each][768:1024]
    p4dg = data[each][1972:2048]
    p4db = data[each][2816:3072]
    
    feat = []
    
    #count number of rgb pixels and find average
    redcount = 0;
    greencount = 0;
    bluecount = 0;
    for x in range(0, 256):
        redcount += data[each][x]
    for x in range(1024, 1280):
        greencount += data[each][x]
    for x in range(2048, 2304):
        bluecount += data[each][x]
    feat.append(redcount/768) 
    feat.append(greencount/768)
    feat.append(bluecount/768)
    feat.append(np.amax(p1dr))
    feat.append(np.amax(p1dg))
    feat.append(np.amax(p1db))
    feat.append(np.amin(p1dr))
    feat.append(np.amin(p1dg))
    feat.append(np.amin(p1db))

    #count number of rgb pixels and find average
    redcount = 0;
    greencount = 0;
    bluecount = 0;
    for x in range(256,512):
        redcount += data[each][x]
    for x in range(1280,1536):
        greencount += data[each][x]
    for x in range(2304,2560):
        bluecount += data[each][x]
    feat.append(redcount/768) 
    feat.append(greencount/768)
    feat.append(bluecount/768)
    feat.append(np.amax(p2dr))
    feat.append(np.amax(p2dg))
    feat.append(np.amax(p2db))
    feat.append(np.amin(p2dr))
    feat.append(np.amin(p2dg))
    feat.append(np.amin(p2db))

    #count number of rgb pixels and find average
    redcount = 0;
    greencount = 0;
    bluecount = 0;
    for x in range(512,768):
        redcount += data[each][x]
    for x in range(1536,1792):
        greencount += data[each][x]
    for x in range(2560,2816):
        bluecount += data[each][x]
    feat.append(redcount/768) 
    feat.append(greencount/768)
    feat.append(bluecount/768)
    feat.append(np.amax(p3dr))
    feat.append(np.amax(p3dg))
    feat.append(np.amax(p3db))
    feat.append(np.amin(p3dr))
    feat.append(np.amin(p3dg))
    feat.append(np.amin(p3db))

    #count number of rgb pixels and find average
    redcount = 0;
    greencount = 0;
    bluecount = 0;
    for x in range(768,1024):
        redcount += data[each][x]
    for x in range(1792,2048):
        greencount += data[each][x]
    for x in range(2816,3072):
        bluecount += data[each][x]
    feat.append(redcount/768) 
    feat.append(greencount/768)
    feat.append(bluecount/768)
    feat.append(np.amax(p4dr))
    feat.append(np.amax(p4dg))
    feat.append(np.amax(p4db))
    feat.append(np.amin(p4dr))
    feat.append(np.amin(p4dg))
    feat.append(np.amin(p4db))

    X_train.append(feat)

# In[9]:

# calculating the feature vectors by extracting features such as average intensity of all pixels and max & min of pixels(test set)
# segmenting image into 4 parts
for each in range(0, 10000):
    p1dr = test_data[each][0:256]
    p1dg = test_data[each][1024:1280]
    p1db = test_data[each][2048:2304]
    p2dr = test_data[each][256:512]
    p2dg = test_data[each][1280:1536]
    p2db = test_data[each][2304:2560]
    p3dr = test_data[each][512:768]
    p3dg = test_data[each][1536:1792]
    p3db = test_data[each][2560:2816]
    p4dr = test_data[each][768:1024]
    p4dg = test_data[each][1972:2048]
    p4db = test_data[each][2816:3072]
    
    feat1 = []
    
    #count number of rgb pixels and find average
    redcount = 0;
    greencount = 0;
    bluecount = 0;
    for x in range(0, 256):
        redcount += test_data[each][x]
    for x in range(1024, 1280):
        greencount += test_data[each][x]
    for x in range(2048, 2304):
        bluecount += test_data[each][x]
    feat1.append(redcount/768) 
    feat1.append(greencount/768)
    feat1.append(bluecount/768)
    feat1.append(np.amax(p1dr))
    feat1.append(np.amax(p1dg))
    feat1.append(np.amax(p1db))
    feat1.append(np.amin(p1dr))
    feat1.append(np.amin(p1dg))
    feat1.append(np.amin(p1db))

    #count number of rgb pixels and find average
    redcount = 0;
    greencount = 0;
    bluecount = 0;
    for x in range(256,512):
        redcount += test_data[each][x]
    for x in range(1280,1536):
        greencount += test_data[each][x]
    for x in range(2304,2560):
        bluecount += test_data[each][x]
    feat1.append(redcount/768) 
    feat1.append(greencount/768)
    feat1.append(bluecount/768)
    feat1.append(np.amax(p2dr))
    feat1.append(np.amax(p2dg))
    feat1.append(np.amax(p2db))
    feat1.append(np.amin(p2dr))
    feat1.append(np.amin(p2dg))
    feat1.append(np.amin(p2db))

    #count number of rgb pixels and find average
    redcount = 0;
    greencount = 0;
    bluecount = 0;
    for x in range(512,768):
        redcount += test_data[each][x]
    for x in range(1536,1792):
        greencount += test_data[each][x]
    for x in range(2560,2816):
        bluecount += test_data[each][x]
    feat1.append(redcount/768) 
    feat1.append(greencount/768)
    feat1.append(bluecount/768)
    feat1.append(np.amax(p3dr))
    feat1.append(np.amax(p3dg))
    feat1.append(np.amax(p3db))
    feat1.append(np.amin(p3dr))
    feat1.append(np.amin(p3dg))
    feat1.append(np.amin(p3db))

    #count number of rgb pixels and find average
    redcount = 0;
    greencount = 0;
    bluecount = 0;
    for x in range(768,1024):
        redcount += test_data[each][x]
    for x in range(1792,2048):
        greencount += test_data[each][x]
    for x in range(2816,3072):
        bluecount += test_data[each][x]
    feat1.append(redcount/768) 
    feat1.append(greencount/768)
    feat1.append(bluecount/768)
    feat1.append(np.amax(p4dr))
    feat1.append(np.amax(p4dg))
    feat1.append(np.amax(p4db))
    feat1.append(np.amin(p4dr))
    feat1.append(np.amin(p4dg))
    feat1.append(np.amin(p4db))

    X_test.append(feat1)

# In[10]:

#computing the principle components
X = np.array(X_train)
pca = PCA(n_components=36)
abc = pca.fit_transform(X)
pca1 = pca.score(X)
print "PCA of Feature Vectors"
print abc,pca1


# In[11]:

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


# In[12]:

#creating dictionaries of each class of images in training set
class_data = {'airplane': [], 'automobile': [],'bird':[],'cat':[], 'deer':[], 'dog':[],'frog':[],'horse':[], 'ship':[], 'truck':[]}


# In[13]:

#creating dictionaries of each class of images in test set
class_data_test = {'airplane': [], 'automobile': [],'bird':[],'cat':[], 'deer':[], 'dog':[],'frog':[],'horse':[], 'ship':[], 'truck':[]}


# In[14]:

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


# In[15]:

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


# In[16]:

# function to calculate mean ,variance, and standard deviation for each class of images
def get_stats(class_name):
    mean = 0
    variance = 0
    std_deviation = 0
    size = len(class_data[class_name])
    for each in class_data[class_name]:
        mean += np.mean(each)
        variance += np.var(each)
        std_deviation += np.std(each)
    mean = mean / size
    variance = variance / size
    std_deviation = std_deviation / size
    return mean, variance, std_deviation, size


# In[17]:
print "Mean,  Variance,   Standard Deviation,    Size"
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


# In[18]:

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


# In[19]:
print " T-value,   Degree of Freedom,    P-Value"
print tstat('airplane','deer')


# In[20]:

#performing k-means clustering with 10 clusters each for each class of images
print "K-means Clustering on CIFAR-10"
k_means = KMeans(n_clusters=10, n_jobs=8)
print k_means.fit(X_train)


# In[21]:
print "Predicting the respective class of images in cluster"
print k_means.predict(class_data_test['airplane'])


# In[22]:

# function to give the respective number of class of images in all the clusters
def get_count(class_name):
    return Counter(k_means.predict(class_data_test[class_name]))


# In[23]:
print " Cluster Number AND Number of Images of Particular Class"
print get_count('truck')


# In[24]:

# function to get the probability of a particular class of images with the cluster number that it is present 
from collections import Counter
def get_prob(class_name):
    c = Counter(k_means.predict(class_data_test[class_name]))
    clusternumber = c.most_common(1)[0][0]
    val = (c.most_common(1)[0][1])
    return clusternumber, val


# In[25]:
print "Cluster Number AND Probability"
print get_prob('truck')


# In[26]:

# performing multinomial logistic regression to find out the accuracy of the classification of images in the training set with the testing set
model = LogisticRegression(solver = "lbfgs", multi_class = "multinomial")
model = model.fit(X_train, y_train)
model = model.fit(X_train, y_train)
print "Accuracy Score"
print model.score(X_test, y_test)

