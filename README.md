# Black-Box-Optimization-in-Simulation-Based-Design

This is the repository of our Spring 2024-IE492 project, Black-Box Optimization in Simulation-Based Design. Here, you can find the code samples and files used in sampling, surrogate modeling, AI optimizations and more.

![image](https://github.com/HaticeSerraHakyemez/Black-Box-Optimization-in-Simulation-Based-Design/assets/81290256/51f35062-639a-402a-a67d-f2cce0ee79f6)

You can provide the function you want to optimize, the sampling method used at each iteration of sequential sampling, and the model of your choosing to be used as a surrogate model to train on the samples by importing the bbo_optimizer and utils folder. An example usage of our code can be seen below:

 ```
import bbo_optimizer
import utils.samplers as samplers
import utils.models as models

bbo_optimizer.optimize("Bukin", samplers.LatinHypercubeSampling, models.RandomForestModel)
 ```
