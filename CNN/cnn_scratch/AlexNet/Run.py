#importing required packages
import numpy as np
from architectures import Architectures

#input
np.random.seed(0)
x=np.random.uniform(low=0,high=1,size=(3,227,227))

emp=Architectures.AlexNet(x)
print(emp)