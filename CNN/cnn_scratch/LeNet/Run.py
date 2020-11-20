#importing required packages
import numpy as np
from architectures import Architectures

#input
np.random.seed(0)
x=np.random.uniform(low=0,high=1,size=(3,32,32))
y=np.array([[0.2],[0.2],[0.2],[0.2],[0.2],[0],[0],[0],[0],[0]])

emp=Architectures.LeNet(x,y)
print(emp)
