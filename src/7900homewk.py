#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries:
import numpy as np

# Define input features:
input_features = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])

print (input_features.shape)
print (input_features)

# Define target output:
target_output = np.array([[0,1,1,1,0,1,0,1]])

# Reshaping our target output into vector:
target_output = target_output.reshape(8,1)

print(target_output.shape)
print (target_output)

# Define weights:
weights = np.array([[0.1],[0.2],[0.2]])
print(weights.shape)
print (weights)


# Bias weight:
bias = 0.3

# Learning Rate:
lr = 0.03

# Sigmoid function:
def sigmoid(x):
    return 1/(1+np.exp(-x))


# Derivative of sigmoid function:
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

# Main logic for neural network:
# Running our code 10000 times:

for epoch in range(10000):
    inputs = input_features

    #Feedforward input:
    in_o = np.dot(inputs, weights) + bias 

    #Feedforward output:
    out_o = sigmoid(in_o) 

    #Backpropogation
    #Calculating error
    error = out_o - target_output

    #Going with the formula:
    x = error.sum()
    print("error")
    print(x)

    #Calculating derivative:
    derror_douto = error
    douto_dino = sigmoid_der(out_o)

    #Multiplying individual derivatives:
    deriv = derror_douto * douto_dino 

    #Multiplying with the 3rd individual derivative:
    #Finding the transpose of input_features:
    inputs = input_features.T
    deriv_final = np.dot(inputs,deriv)

    #Updating the weights values:
    weights -= lr * deriv_final 
    #Updating the bias weight value:
    for i in deriv:
        bias -= lr * i #Check the final values for weight and bias
print('\n')
print("final weight")
print(weights)
print('\n')
print("final bias")
print(bias) 
print('\n')

#Taking inputs:
single_point = np.array([0,1,1]) 

#1st step:
result1 = np.dot(single_point, weights) + bias 

#2nd step:
result2 = sigmoid(result1) 

#Print final result
print("The result should be close to one")
print(result2) 
print('\n')
#Taking inputs:
single_point = np.array([1,1,0]) 

#1st step:
result1 = np.dot(single_point, weights) + bias 

#2nd step:
result2 = sigmoid(result1) 

#Print final result
print("The result should be close to zero")
print(result2) 
print('\n')


#Taking inputs:
single_point = np.array([0,0,1]) 

#1st step:
result1 = np.dot(single_point, weights) + bias 
#2nd step:

result2 = sigmoid(result1) 

#Print final result
print("The result should be close to one")
print(result2)
print('\n')


# In[4]:


# Import required libraries:
import numpy as np

# Define input features:
input_features = np.array([[0,0],[0,1],[1,0],[1,1]])

print (input_features.shape)
print (input_features)

# Define target output:
target_output = np.array([[0,0,1,0]])

# Reshaping our target output into vector:
target_output = target_output.reshape(4,1)

print(target_output.shape)
print (target_output)

# Define weights:
weights = np.array([[0.1],[0.2]])
print(weights.shape)
print (weights)


# Bias weight:
bias = 0.3

# Learning Rate:
lr = 0.03

# Sigmoid function:
def sigmoid(x):
    return 1/(1+np.exp(-x))


# Derivative of sigmoid function:
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

# Main logic for neural network:
# Running our code 10000 times:

for epoch in range(10000):
    inputs = input_features

    #Feedforward input:
    in_o = np.dot(inputs, weights) + bias 

    #Feedforward output:
    out_o = sigmoid(in_o) 

    #Backpropogation
    #Calculating error
    error = out_o - target_output

    #Going with the formula:
    x = error.sum()
    print("error")
    print(x)

    #Calculating derivative:
    derror_douto = error
    douto_dino = sigmoid_der(out_o)

    #Multiplying individual derivatives:
    deriv = derror_douto * douto_dino 

    #Multiplying with the 3rd individual derivative:
    #Finding the transpose of input_features:
    inputs = input_features.T
    deriv_final = np.dot(inputs,deriv)

    #Updating the weights values:
    weights -= lr * deriv_final 
    #Updating the bias weight value:
    for i in deriv:
        bias -= lr * i #Check the final values for weight and bias
print('\n')
print("final weight")
print(weights)
print('\n')
print("final bias")
print(bias) 
print('\n')

#Taking inputs:
single_point = np.array([0,0]) 

#1st step:
result1 = np.dot(single_point, weights) + bias 

#2nd step:
result2 = sigmoid(result1) 

#Print final result
print("The result should be close to zero")
print(result2)
print("###################")
print('\n')



#Taking inputs:
single_point = np.array([0,1]) 

#1st step:
result1 = np.dot(single_point, weights) + bias 

#2nd step:
result2 = sigmoid(result1) 

#Print final result
print("The result should be close to zero")
print(result2) 
print("###################")
print('\n')


#Taking inputs:
single_point = np.array([1,0]) 

#1st step:
result1 = np.dot(single_point, weights) + bias 
#2nd step:

result2 = sigmoid(result1) 

#Print final result
print("The result should be close to one")
print(result2)
print("###################")
print('\n')




#Taking inputs:
single_point = np.array([1,1]) 

#1st step:
result1 = np.dot(single_point, weights) + bias 
#2nd step:

result2 = sigmoid(result1) 

#Print final result
print("The result should be close to zero")
print(result2)
print("###################")
print('\n')


# In[5]:


# normal_curve.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# if using a Jupyter notebook, inlcude:
get_ipython().run_line_magic('matplotlib', 'inline')
# define constants
mu = 998.8 
sigma = 73.10
x1 = 900
x2 = 1100
# calculate the z-transform
z1 = ( x1 - mu ) / sigma
z2 = ( x2 - mu ) / sigma
x = np.arange(z1, z2, 0.001) # range of x in spec
x_all = np.arange(-10, 10, 0.001) # entire range of x, both in and out of spec
# mean = 0, stddev = 1, since Z-transform was calculated
y = norm.pdf(x,0,1)
y2 = norm.pdf(x_all,0,1)
# build the plot
fig, ax = plt.subplots(figsize=(9,6))
plt.style.use('fivethirtyeight')
ax.plot(x_all,y2)

ax.fill_between(x,y,0, alpha=0.3, color='b')
ax.fill_between(x_all,y2,0, alpha=0.1)
ax.set_xlim([-4,4])
ax.set_xlabel('# of Standard Deviations Outside the Mean')
ax.set_yticklabels([])
ax.set_title('Normal Gaussian Curve')

plt.savefig('normal_curve.png', dpi=72, bbox_inches='tight')
plt.show()


# In[9]:


# normal_curve.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# if using a Jupyter notebook, inlcude:
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy
mean = 0
standard_deviation = 1

x_values = np.arange(-5, 5, 0.1)
y_values = scipy.stats.norm(mean, standard_deviation)

plt.plot(x_values, y_values.pdf(x_values))


# In[36]:


# normal_curve.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# if using a Jupyter notebook, inlcude:
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy
# the mean
mean = -10
# the standard of deviation: how far the possible values extend from the mean.
# in this case not very far b/c the standard of deviation is 1.
standard_deviation = 1
# the x range
x_values = np.arange(-15, -5, 0.1)
# the y range
y_values = scipy.stats.norm(mean, standard_deviation)


plt.plot(x_values, y_values.pdf(x_values))

plt.xlabel("X axis label")
plt.ylabel("Y axis label")


# In[ ]:





# In[34]:


# normal_curve.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# if using a Jupyter notebook, inlcude:
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy
# the mean
mean = 10
# the standard of deviation: how far the possible values extend from the mean.
# in this case not very far b/c the standard of deviation is 1.
standard_deviation = 1
# the x range
x_values = np.arange(5, 15, 0.1)

# the y range
y_values = scipy.stats.norm(mean, standard_deviation)

plt.plot(x_values, y_values.pdf(x_values))

plt.xlabel("X axis label")
plt.ylabel("Y axis label")


# In[39]:


# Import required libraries:
import numpy as np

# Define input features:
input_features = np.array([[-12,0.05],[-10,0.40],[-8,0.05],[-11,0.15],[-11.5,0.21],[-11.2,0.30],[-9.5,0.10],
                           [-9,0.20],[-9.7,0.10],[8,0.05],[10,0.40],[12,0.05],[9.2,0.15],[9.6,0.21],[9.8,0.8],
                          [11,0.10],[10.2,0.15],[11.7,0.09]])

print (input_features.shape)
print (input_features)

# Define target output:
target_output = np.array([[0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1]])

# Reshaping our target output into vector:
target_output = target_output.reshape(18,1)

print(target_output.shape)
print (target_output)

# Define weights:
weights = np.array([[0.1],[0.2]])
print(weights.shape)
print (weights)


# Bias weight:
bias = 0.3

# Learning Rate:
lr = 0.03

# Sigmoid function:
def sigmoid(x):
    return 1/(1+np.exp(-x))


# Derivative of sigmoid function:
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

# Main logic for neural network:
# Running our code 10000 times:

for epoch in range(10000):
    inputs = input_features

    #Feedforward input:
    in_o = np.dot(inputs, weights) + bias 

    #Feedforward output:
    out_o = sigmoid(in_o) 

    #Backpropogation
    #Calculating error
    error = out_o - target_output

    #Going with the formula:
    x = error.sum()
    print("error")
    print(x)

    #Calculating derivative:
    derror_douto = error
    douto_dino = sigmoid_der(out_o)

    #Multiplying individual derivatives:
    deriv = derror_douto * douto_dino 

    #Multiplying with the 3rd individual derivative:
    #Finding the transpose of input_features:
    inputs = input_features.T
    deriv_final = np.dot(inputs,deriv)

    #Updating the weights values:
    weights -= lr * deriv_final 
    #Updating the bias weight value:
    for i in deriv:
        bias -= lr * i #Check the final values for weight and bias
print('\n')
print("final weight")
print(weights)
print('\n')
print("final bias")
print(bias) 
print('\n')

print("Four group Zero")
#Taking inputs:
single_point = np.array([-11,0.05]) 

#1st step:
result1 = np.dot(single_point, weights) + bias 

#2nd step:
result2 = sigmoid(result1) 

#Print final result
print("The result should be close to zero")
print(result2)
print("###################")
print('\n')



#Taking inputs:
single_point = np.array([-11.4,0.21]) 

#1st step:
result1 = np.dot(single_point, weights) + bias 

#2nd step:
result2 = sigmoid(result1) 

#Print final result
print("The result should be close to zero")
print(result2) 
print("###################")
print('\n')


#Taking inputs:
single_point = np.array([-11.8,0.08]) 

#1st step:
result1 = np.dot(single_point, weights) + bias 
#2nd step:

result2 = sigmoid(result1) 

#Print final result
print("The result should be close to zero")
print(result2)
print("###################")
print('\n')




#Taking inputs:
single_point = np.array([-9,0.05]) 

#1st step:
result1 = np.dot(single_point, weights) + bias 
#2nd step:

result2 = sigmoid(result1) 

#Print final result
print("The result should be close to zero")
print(result2)
print("###################")
print('\n')






print('\n')
print('\n')
print('\n')


print("Four group One")
#Taking inputs:
single_point = np.array([10,0.05]) 

#1st step:
result1 = np.dot(single_point, weights) + bias 

#2nd step:
result2 = sigmoid(result1) 

#Print final result
print("The result should be close to one")
print(result2)
print("###################")
print('\n')



#Taking inputs:
single_point = np.array([11.2,0.10]) 

#1st step:
result1 = np.dot(single_point, weights) + bias 

#2nd step:
result2 = sigmoid(result1) 

#Print final result
print("The result should be close to one")
print(result2) 
print("###################")
print('\n')


#Taking inputs:
single_point = np.array([9,0.23]) 

#1st step:
result1 = np.dot(single_point, weights) + bias 
#2nd step:

result2 = sigmoid(result1) 

#Print final result
print("The result should be close to one")
print(result2)
print("###################")
print('\n')




#Taking inputs:
single_point = np.array([10.1,0.25]) 

#1st step:
result1 = np.dot(single_point, weights) + bias 
#2nd step:

result2 = sigmoid(result1) 

#Print final result
print("The result should be close to one")
print(result2)
print("###################")
print('\n')


# In[29]:


from PIL import Image
from numpy import asarray
import os
# load the image
image = Image.open('image1.jpeg')


# convert image to numpy array
data = asarray(image)
print(type(data))
# summarize shape
print(data.shape)

# create Pillow image
image2 = Image.fromarray(data)
print(type(image2))

# summarize image details
print(image2.mode)
print(image2.size)


# In[92]:


from PIL import Image
from numpy import asarray
import numpy as np


# load the image
# This will be the training images
im1 = Image.open('image1.jpeg')
im2 = Image.open('image2.jpeg')
im3 = Image.open('image3.jpeg')
im4 = Image.open('image4.jpeg')
im5 = Image.open('image5.jpeg')
im6 = Image.open('image6.jpeg')
im7 = Image.open('image7.jpeg')
im8 = Image.open('image8.jpeg')
im9 = Image.open('image9.jpeg')
im10 = Image.open('image10.jpeg')
im11 = Image.open('image11.jpeg')
im12 = Image.open('image12.jpeg')
im13 = Image.open('image13.jpeg')
im14 = Image.open('image14.jpeg')
im15 = Image.open('image15.jpeg')
im16 = Image.open('image16.jpeg')
im17 = Image.open('image17.jpeg')
im18 = Image.open('image18.jpeg')
im19 = Image.open('image19.jpeg')
im20 = Image.open('image20.jpeg')
im21 = Image.open('image21.jpeg')
im22 = Image.open('image22.jpeg')
im23 = Image.open('image23.jpeg')
im24 = Image.open('image24.jpeg')
im25 = Image.open('image25.jpeg')
im26 = Image.open('image26.jpeg')
im27 = Image.open('image27.jpeg')
im28 = Image.open('image28.jpeg')
im29 = Image.open('image29.jpeg')
im30 = Image.open('image30.jpeg')
im31 = Image.open('image31.jpeg')
im32 = Image.open('image32.jpeg')
im33 = Image.open('image33.jpeg')
im34 = Image.open('image34.jpeg')
im35 = Image.open('image35.jpeg')
im36 = Image.open('image36.jpeg')
im37 = Image.open('image37.jpeg')
im38 = Image.open('image38.jpeg')










# convert image to numpy array
data1 = asarray(im1)
data2 = asarray(im2)
data3 = asarray(im3)
data4 = asarray(im4)
data5 = asarray(im5)
data6 = asarray(im6)
data7 = asarray(im7)
data8 = asarray(im8)
data9 = asarray(im9)
data10 = asarray(im10)
data11 = asarray(im11)
data12 = asarray(im12)
data13 = asarray(im13)
data14 = asarray(im14)
data15 = asarray(im15)
data16 = asarray(im16)
data17 = asarray(im17)
data18 = asarray(im18)
data19 = asarray(im19)
data20 = asarray(im20)
data21 = asarray(im21)
data22 = asarray(im22)
data23 = asarray(im23)
data24 = asarray(im24)
data25 = asarray(im25)
data26 = asarray(im26)
data27 = asarray(im27)
data28 = asarray(im28)
data29 = asarray(im29)
data30 = asarray(im30)
data31 = asarray(im31)
data32 = asarray(im32)
data33 = asarray(im33)
data34 = asarray(im34)
data35 = asarray(im35)
data36 = asarray(im36)
data37 = asarray(im37)
data38 = asarray(im38)




# Check array size by printing test arrays
print("check the array size by printing two test arrays: It is a 7 by 6 matrix")
print("##########")
print(data1)
print("##########")
print(data2)
print("##########")



# print out the transpose of the test matrix
print('\n')
print("check the transpose array by printing two test arrays: It is a 6 by 7 matrix")
print("##########")
data1T = data1.transpose()
print(data1T)
print("##########")
data2T = data2.transpose()
print(data2T)
print("##########")

data3T = data3.transpose()
data4T = data4.transpose()
data5T = data5.transpose()
data6T = data6.transpose()
data7T = data7.transpose()
data8T = data8.transpose()
data9T = data9.transpose()
data10T = data10.transpose()
data11T = data11.transpose()
data12T = data12.transpose()
data13T = data13.transpose()
data14T = data14.transpose()
data15T = data15.transpose()
data16T = data16.transpose()
data17T = data17.transpose()
data18T = data18.transpose()
data19T = data19.transpose()
data20T = data20.transpose()
data21T = data21.transpose()
data22T = data22.transpose()
data23T = data23.transpose()
data24T = data24.transpose()
data25T = data25.transpose()
data26T = data26.transpose()
data27T = data27.transpose()
data28T = data28.transpose()
data29T = data29.transpose()
data30T = data30.transpose()
data31T = data31.transpose()
data32T = data32.transpose()
data33T = data33.transpose()
data34T = data34.transpose()
data35T = data35.transpose()
data36T = data36.transpose()
data37T = data37.transpose()
data38T = data38.transpose()


a1 = np.array([[1, 0], 
               [0, 0],
               [0, 0],
               [0, 0],
               [0, 0],
               [0, 0],
               [0, 0]])

a2 = np.array([[1, 1], 
               [0, 0],
               [0, 0],
               [0, 0],
               [0, 0],
               [0, 0],
               [0, 0]])

a3 = np.array([[1, 1], 
               [1, 0],
               [0, 0],
               [0, 0],
               [0, 0],
               [0, 0],
               [0, 0]])

a4 = np.array([[1, 1], 
               [1, 1],
               [0, 0],
               [0, 0],
               [0, 0],
               [0, 0],
               [0, 0]])

a5 = np.array([[1, 1], 
               [1, 1],
               [1, 0],
               [0, 0],
               [0, 0],
               [0, 0],
               [0, 0]])

a6 = np.array([[1, 1], 
               [1, 1],
               [1, 1],
               [0, 0],
               [0, 0],
               [0, 0],
               [0, 0]])

a7 = np.array([[1, 1], 
               [1, 1],
               [1, 1],
               [1, 0],
               [0, 0],
               [0, 0],
               [0, 0]])


a8 = np.array([[1, 1], 
               [1, 1],
               [1, 1],
               [1, 1],
               [0, 0],
               [0, 0],
               [0, 0]])



a9 = np.array([[1, 1], 
               [1, 1],
               [1, 1],
               [1, 1],
               [1, 0],
               [0, 0],
               [0, 0]])


a10 = np.array([[1, 1], 
               [1, 1],
               [1, 1],
               [1, 1],
               [1, 1],
               [0, 0],
               [0, 0]])


a11 = np.array([[1, 1], 
               [1, 1],
               [1, 1],
               [1, 1],
               [1, 1],
               [1, 0],
               [0, 0]])


a12 = np.array([[1, 1], 
               [1, 1],
               [1, 1],
               [1, 1],
               [1, 1],
               [1, 1],
               [0, 0]])





w1 = np.dot(data1T, a1)
print("##############")
print("print out w1")
print(w1)
print("##############")
w2 = np.dot(data2T, a1)
w3 = np.dot(data3T, a1)



w4 = np.dot(data4T, a2)
w5 = np.dot(data5T, a2)
w6 = np.dot(data6T, a2)



w7 = np.dot(data7T, a3)
w8 = np.dot(data8T, a3)
w9 = np.dot(data9T, a3)


w10 = np.dot(data10T, a4)
w12 = np.dot(data11T, a4)
w11 = np.dot(data12T, a4)

w13 = np.dot(data13T, a5)
w14 = np.dot(data14T, a5)
w15 = np.dot(data15T, a5)





w16 = np.dot(data16T, a6)
w17 = np.dot(data17T, a6)
w18 = np.dot(data18T, a6)




w19 = np.dot(data19T, a7)
w20 = np.dot(data20T, a7)
w21 = np.dot(data21T, a7)



w22 = np.dot(data22T, a8)
w23 = np.dot(data23T, a8)
w24 = np.dot(data24T, a8)



w25 = np.dot(data25T, a9)
w26 = np.dot(data26T, a9)
w27 = np.dot(data27T, a9)


w28 = np.dot(data28T, a10)
w29 = np.dot(data29T, a10)
w30 = np.dot(data30T, a10)



w31 = np.dot(data31T, a11)
w32 = np.dot(data32T, a11)
w33 = np.dot(data33T, a11)
w34 = np.dot(data34T, a11)
             
w35 = np.dot(data35T, a12)
w36 = np.dot(data36T, a12)
w37 = np.dot(data38T, a12)
w38 = np.dot(data38T, a12)



W = w1 + w2 
#+ w3 + w4 
#+ w5 + w6 + w7 + w8 + w9 + w10 + w11 + w12 
#+ w13 + w14 + w15 + w16 + w17 + w18 + w19 + w20 + w21 + w22 + w23 + w24 + w25 + w26 + w27 + w28 + w29 + w30 + w31 + w32 + w33+ w34 + w35 + w36 + w37 + w38 





add1 = w1 + w2

print("#######")
print("regular add")
print(add1)
print("#######")

print("########")
print("Add np")
add2 = np.add(w1, w2)
print(add2)
print("#########")
print('\n')

print("##########")
print("print out W")
print(W)
print("##########")



Result = np.dot(data2, W)

print(Result)
count = 0    
    

 #Activation function   
for row in range(0, 7):
    for column in range(0, 2):
        if Result[row][column] > 0:
            Result[row][column] = 1
        elif Result[row][column] == 0:
            Result[row][column] = 0
            

for row in range(0, 7):
    for column in range(0, 2):
        if Result[row][column] == 1:
            count += 1
                 
print("")   
print(Result)
print(count)       
        
        

    
    


# In[53]:


from PIL import Image
from numpy import asarray
import numpy as np


#input features
input1 = np.array([[1,0,0,0]]) 
input2 = np.array([[1,1,0,0]]) 
input3 = np.array([[0,0,0,1]]) 
input4 = np.array([[0,0,1,1]]) 

#output features
out1 = np.array([[1, 0]]) 
out2 = np.array([[1, 0]]) 
out3 = np.array([[0, 1]]) 
out4 = np.array([[0, 1]]) 

#transpose the input
input1T = input1.transpose()
input2T = input2.transpose()
input3T = input3.transpose()
input4T = input4.transpose()

#multiply the transposed inputs to their corresponding output
o1  = np.dot(input1T, out1)
o2 = np.dot(input2T, out2)
o3 = np.dot(input3T, out3)
o4 = np.dot(input4T, out4)

W = o1 + o2 + o3 + o4

Result = np.dot(input1, W)


 #Activation function   
for row in range(0, 1):
    for column in range(0, 2):
        if Result[row][column] > 0:
            Result[row][column] = 1
        elif Result[row][column] == 0:
            Result[row][column] = 0
            


print("######")
print("for the first input [1,0,0,0] ")
print(Result)
print("######")



Result = np.dot(input4, W)
 #Activation function   
for row in range(0, 1):
    for column in range(0, 2):
        if Result[row][column] > 0:
            Result[row][column] = 1
        elif Result[row][column] == 0:
            Result[row][column] = 0
            

print('\n')
print("for the fourth input [0,0,1,1]")
print("######")
print(Result)
print("######")

NoisyInput = np.array([[1,0,0,1]]) 



Result = np.dot(NoisyInput, W)


 #Activation function   
for row in range(0, 1):
    for column in range(0, 2):
        if Result[row][column] > 0:
            Result[row][column] = 1
        elif Result[row][column] == 0:
            Result[row][column] = 0
            


print("######")
print("for the noisy input [1,0,0,1] ")
print("In this case the noisy input represents nothing")
print(Result)
print("######")



# In[50]:


from PIL import Image
from numpy import asarray
import numpy as np


# load the image
# This will be the training images


im1 = Image.open('image1.jpeg')

# convert image to numpy array
data1 = asarray(im1)


#print out the array
print("print out the array of the image. In this case image1")
print(data1)

#flatten image to vector 
X = data1.flatten()
print('\n')
print("print flatten matrix")
print(X)

XT = X

#image had to be shape before, you can properly transpose
XT.shape = (42,1)
XT = np.transpose(XT)



#print out transpose matrix 
print('\n')
print("transpose of matrix")
print(XT)



# Memory matrix
M = np.dot(X, XT)
#print out memory matrix
print('\n')
print("print out the memory matrix of image1. A 42 * 42 matrix")
print(M)




# Testing the input to see the resulting matrix
W = np.dot(XT, M)

print('\n')
print("print out image1 transpose matrix after it has been multiply by the memory matrix")
print(W)

#Re Reshape the matrix back to it's prevous form
Result = np.reshape(W, (7, 6))



print('\n')
print("print out reshaped matrix")
print(Result)



count = 0    
    

 #Activation function   
for row in range(0, 7):
    for column in range(0, 6):
        if Result[row][column] > 0:
            Result[row][column] = 1
        elif Result[row][column] == 0:
            Result[row][column] = 0
            
#count the total number of firing. That is count the total number of ones.
for row in range(0, 7):
    for column in range(0, 6):
        if Result[row][column] == 1:
            count += 1
                 
print('\n')  
print("print out matrix after it has gone through the activation function")
print(Result)
print("The total number of firing is")
print(count)  




# noisy image1





Nmage1 = np.array([[  0, 252, 255, 252, 253,   0],
                   [255,  13,   0,   0,  0, 255],
                   [255,   0,   0,   9,   0, 255],
                   [248, 255, 236, 253, 255, 240],
                   [0,   0,  10,   8,   0, 255],
                   [255,   0,   7,   2,   0, 255],
                   [255,   0,  10,   0,  0, 248]])

print('\n')
print('\n')
print("noisy image1 from changing three numbers in the matrix to zero")
print("########################")
print('\n')
print("print noisy image1")
print(Nmage1)


#flatten image to vector 
X = Nmage1.flatten()
print('\n')
print("print flatten matrix")
print(X)

XT = X

#image had to be shape before, you can properly transpose
XT.shape = (42,1)
XT = np.transpose(XT)



# Testing the input to see the resulting matrix
W = np.dot(XT, M)



#Re Reshape the matrix back to it's prevous form
Result = np.reshape(W, (7, 6))



print('\n')
print("print out reshaped matrix")
print(Result)



count = 0    
    

 #Activation function   
for row in range(0, 7):
    for column in range(0, 6):
        if Result[row][column] > 0:
            Result[row][column] = 1
        elif Result[row][column] == 0:
            Result[row][column] = 0
            
#count the total number of firing. That is count the total number of ones.
for row in range(0, 7):
    for column in range(0, 6):
        if Result[row][column] == 1:
            count += 1
                 

                
                
print('\n')  
print("print out matrix after it has gone through the activation function")
print("The matrix was able to recall after three inputs were change")
print(Result)
print("The total number of firing is")
print(count)  

print("#####################")




Nmage1 = np.array([[  0, 252, 0, 252, 253,   0],
                   [0,  13,   0,   0,  0, 255],
                   [255,   0,   0,   9,   0, 255],
                   [248, 255, 236, 253, 0, 240],
                   [255,   0,  10,   8,   0, 255],
                   [0,   0,   7,   2,   0, 255],
                   [255,   0,  10,   0,  13, 0]])

print('\n')
print('\n')
print("noisy image1 from changing six numbers in the matrix to zero")
print("########################")
print('\n')
print("print noisy image1")
print(Nmage1)


#flatten image to vector 
X = Nmage1.flatten()
print('\n')
print("print flatten matrix")
print(X)

XT = X

#image had to be shape before, you can properly transpose
XT.shape = (42,1)
XT = np.transpose(XT)



# Testing the input to see the resulting matrix
W = np.dot(XT, M)



#Re Reshape the matrix back to it's prevous form
Result = np.reshape(W, (7, 6))



print('\n')
print("print out reshaped matrix")
print(Result)



count = 0    
    

 #Activation function   
for row in range(0, 7):
    for column in range(0, 6):
        if Result[row][column] > 0:
            Result[row][column] = 1
        elif Result[row][column] == 0:
            Result[row][column] = 0
            
#count the total number of firing. That is count the total number of ones.
for row in range(0, 7):
    for column in range(0, 6):
        if Result[row][column] == 1:
            count += 1
                 

                
                
print('\n')  
print("print out matrix after it has gone through the activation function")
print("The matrix was able to recall after six inputs were change")
print(Result)
print("The total number of firing is")
print(count)  

print("#####################")





Nmage1 = np.array([[  0, 0, 255, 0, 253,   0],
                   [255,  13,   0,   0,  11, 255],
                   [255,   0,   0,   9,   0, 255],
                   [0, 255, 0, 253, 0, 240],
                   [255,   0,  10,   8,   0, 0],
                   [0,   0,   0,   2,   0, 255],
                   [255,   0,  0,   0,  13, 248]])

print('\n')
print('\n')
print("noisy image1 from changing nine numbers in the matrix to zero")
print("########################")
print('\n')
print("print noisy image1")
print(Nmage1)


#flatten image to vector 
X = Nmage1.flatten()
print('\n')
print("print flatten matrix")
print(X)

XT = X

#image had to be shape before, you can properly transpose
XT.shape = (42,1)
XT = np.transpose(XT)



# Testing the input to see the resulting matrix
W = np.dot(XT, M)



#Re Reshape the matrix back to it's prevous form
Result = np.reshape(W, (7, 6))



print('\n')
print("print out reshaped matrix")
print(Result)



count = 0    
    

 #Activation function   
for row in range(0, 7):
    for column in range(0, 6):
        if Result[row][column] > 0:
            Result[row][column] = 1
        elif Result[row][column] == 0:
            Result[row][column] = 0
            
#count the total number of firing. That is count the total number of ones.
for row in range(0, 7):
    for column in range(0, 6):
        if Result[row][column] == 1:
            count += 1
                 
                
                
print('\n')  
print("print out matrix after it has gone through the activation function")
print("The matrix was able to recall after nine inputs were change")
print(Result)
print("The total number of firing is")
print(count)  

print("#####################")



Nmage1 = np.array([[  0, 252, 0, 0, 253,   0],
                   [255,  13,   0,   0,  0, 255],
                   [0,   0,   0,   9,   0, 0],
                   [0, 255, 0, 253, 255, 240],
                   [255,   0,  10,   8,   0, 0],
                   [0,   0,   0,   0,   0, 255],
                   [0,   0,  10,   0,  13, 248]])



print('\n')
print('\n')
print("noisy image1 from changing twelve numbers in the matrix to zero")
print("########################")
print('\n')
print("print noisy image1")
print(Nmage1)


#flatten image to vector 
X = Nmage1.flatten()
print('\n')
print("print flatten matrix")
print(X)

XT = X

#image had to be shape before, you can properly transpose
XT.shape = (42,1)
XT = np.transpose(XT)

# Testing the input to see the resulting matrix
W = np.dot(XT, M)

#Re Reshape the matrix back to it's prevous form
Result = np.reshape(W, (7, 6))


print('\n')
print("print out reshaped matrix")
print(Result)

count = 0    

 #Activation function   
for row in range(0, 7):
    for column in range(0, 6):
        if Result[row][column] > 0:
            Result[row][column] = 1
        elif Result[row][column] == 0:
            Result[row][column] = 0
            
#count the total number of firing. That is count the total number of ones.
for row in range(0, 7):
    for column in range(0, 6):
        if Result[row][column] == 1:
            count += 1
                                
print('\n')  
print("print out matrix after it has gone through the activation function")
print("The matrix was able to recall after twelve inputs were change")
print(Result)
print("The total number of firing is")
print(count)  

print("#####################")

Nmage1 = np.array([[  0, 252, 0, 0, 253,   0],
                   [0,  13,   0,   0,  0, 0],
                   [255,   0,   0,   0,   0, 0],
                   [0, 0, 236, 0, 0, 0],
                   [0,   0,  10,   8,   0, 0],
                   [0,   0,   0,   2,   0, 255],
                   [0,   0,  0,   0,  13, 0]])
print('\n')
print('\n')
print("noisy image1 from changing nineteen numbers in the matrix to zero")
print("########################")
print('\n')
print("print noisy image1")
print(Nmage1)


#flatten image to vector 
X = Nmage1.flatten()
print('\n')
print("print flatten matrix")
print(X)

XT = X

#image had to be shape before, you can properly transpose
XT.shape = (42,1)
XT = np.transpose(XT)

# Testing the input to see the resulting matrix
W = np.dot(XT, M)

#Re Reshape the matrix back to it's prevous form
Result = np.reshape(W, (7, 6))

print('\n')
print("print out reshaped matrix")
print(Result)

count = 0    
    
#Activation function   
for row in range(0, 7):
    for column in range(0, 6):
        if Result[row][column] > 0:
            Result[row][column] = 1
        elif Result[row][column] == 0:
            Result[row][column] = 0
            
#count the total number of firing. That is count the total number of ones.
for row in range(0, 7):
    for column in range(0, 6):
        if Result[row][column] == 1:
            count += 1              
print('\n')  
print("print out matrix after it has gone through the activation function")
print("The matrix was able to recall after nineteen inputs were change")
print(Result)
print("The total number of firing is")
print(count)  

print("#####################")


Nmage1 = np.array([[  0, 0, 0, 0, 0,   0],
                   [0,  133,   0,   0,  0, 0],
                   [0,   0,   0,   9,   0, 255],
                   [0, 0, 0, 0, 255, 0],
                   [0,   0,  0,   1,   0, 0],
                   [0,   0,   1,   1,   0, 0],
                   [0,   0,  0,   0,  13, 0]])

print('\n')
print('\n')
print("noisy image1 from changing 24 numbers in the matrix to zero to three other numbers change to 1")
print("########################")
print('\n')
print("print noisy image1")
print(Nmage1)


#flatten image to vector 
X = Nmage1.flatten()
print('\n')
print("print flatten matrix")
print(X)

XT = X

#image had to be shape before, you can properly transpose
XT.shape = (42,1)
XT = np.transpose(XT)

# Testing the input to see the resulting matrix
W = np.dot(XT, M)

#Re Reshape the matrix back to it's prevous form
Result = np.reshape(W, (7, 6))

print('\n')
print("print out reshaped matrix")
print(Result)

count = 0    

 #Activation function   
for row in range(0, 7):
    for column in range(0, 6):
        if Result[row][column] > 0:
            Result[row][column] = 1
        elif Result[row][column] == 0:
            Result[row][column] = 0
            
#count the total number of firing. That is count the total number of ones.
for row in range(0, 7):
    for column in range(0, 6):
        if Result[row][column] == 1:
            count += 1
                                 
print('\n')  
print("print out matrix after it has gone through the activation function")
print("The matrix was able to recall after 27 inputs were change")
print(Result)
print("The total number of firing is")
print(count)  

print("#####################")

Print("I will stop here")


# In[2]:


import pandas as pd
from PIL import Image
from numpy import asarray
import numpy as np
import matplotlib.pyplot as plt
#!pip install pyexcel


import math
import pyexcel as pe    
import pyexcel.ext.xls   



#your_matrix = pe.get_array('RBF_Data.xlsx')


df = pd.read_excel('RBF_Data.xlsx')
print(df)


#Convert the column one also known as X to an array
X_array = df[["Unnamed: 0"]].to_numpy()
print("##########")
print('\n')
print("print out X")
print(X_array)





#convert column two also known as Y to an array
Y_array = df[["Unnamed: 1"]].to_numpy()
print("##########")
print('\n')
print("print out Y")
print(Y_array)





#convert column three also known as Label to an array
L_array = df[["Label"]].to_numpy()
print("##########")
print('\n')
print("print out Label")
print(L_array)
print("########")





##### print out the group to figure out the best K to use
# empty list, will hold color value 
# corresponding to x 
col =[] 
  
for i in range(0, len(L_array)): 
    if L_array[i] == 1: 
        col.append('blue')   
    else: 
        col.append('magenta')  
  
for i in range(len(L_array)): 
      
    # plotting the corresponding x with y  
    # and respective color 
    plt.scatter(X_array[i], Y_array[i], c = col[i], s = 10, 
                linewidth = 0) 
      

print("print out plot. The K values will be chosen base on the cluster of the two different groups")
print("Group one cluster center/K is (1,7) and group negative one is (1,2)")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Blue for group 1\nand Magenta for group -1')
plt.legend()
plt.show() 


X_holder = X_array

Y_holder = Y_array

# The Least Mean Sqaure and Gaussian Function
for i in range(0, len(L_array)):
    # For phi one
    #Least Mean Sqaure or Euclidean distance
    X1  = math.sqrt((pow((1-X_array[i]),2)) + (pow((7-Y_array[i]),2)))
    #Gaussian function
    X2  = math.exp( -1 * pow(X1,2) )
    X_holder[i] = X2
    #Same for phi two    
    Y1 = math.sqrt((pow((1-X_array[i]),2)) + (pow((2-Y_array[i]),2)))
    Y2 = math.exp( -1 * pow(Y1,2) )
    Y_holder[i] = Y2


    
for i in range(0, len(L_array)): 
    if L_array[i] == 1: 
        col.append('blue')   
    else: 
        col.append('magenta')  
  
for i in range(len(L_array)): 
      
    # plotting the corresponding x with y  
    # and respective color 
    plt.scatter(X_holder[i], Y_holder[i], c = col[i], s = 10, 
                linewidth = 0) 
      

print("print out plot linearly separable graph")
plt.axvline(x=0.07)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Blue for group 1\nand Magenta for group -1')
plt.legend()
plt.show() 



# In[ ]:




