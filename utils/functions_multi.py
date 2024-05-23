import numpy as np

function_ranges = {
    "Sum of Different Powers_6": [(-1, 1)]*6,
    "Sum of Different Powers_8": [(-1, 1)]*8,
    "Sum of Different Powers_10": [(-1, 1)]*10,
    "Sum of Different Powers_12": [(-1, 1)]*12,
    
    "Trid_2": [(-4, 4), (-4, 4)],
    "Trid_5": [(-25,25)]*5,
    "Trid_10": [(-100,100)]*10,
    "Trid_15": [(-225,225)]*15,
    
    "Dixon Price_3": [(-10, 10)]*3,
    "Dixon Price_5": [(-10, 10)]*5,
    "Dixon Price_7": [(-10, 10)]*7,
    "Dixon Price_9": [(-10, 10)]*9,
    
    "Colville": [(-10, 10), (-10, 10), (-10, 10), (-10, 10)]
}

function_mins = {
    
    "Sum of Different Powers_6": [0],
    "Sum of Different Powers_8": [0],
    "Sum of Different Powers_10": [0],
    "Sum of Different Powers_12": [0],
    "Trid_2": [-2],
    "Trid_5": [-30],
    "Trid_10": [-210],
    "Trid_15": [-665],
   
    "Dixon Price_3": [0],
    "Dixon Price_5": [0],
    "Dixon Price_7": [0],
    "Dixon Price_9": [0],
    
    "Colville": [0]
}

function_averages = {
    "Sum of Different Powers_6": [1.217857136],
    "Sum of Different Powers_8": [1.4289682],
    "Sum of Different Powers_10": [1.603210678219104],
    "Sum of Different Powers_12": [1.751562316],
    
   
    "Trid_2": [12.66607],
    "Trid_5": [1046.96796],
    "Trid_10": [33343.7383], 
    "Trid_15": [253137.30580],
    
    "Dixon Price_3": [24134.32355],
    "Dixon Price_5": [80367.55730980658],
    "Dixon Price_7": [168730.9450],
    "Dixon Price_9": [289238.2449],
    
    "Colville": [387103.1860940835]
}


def sum_of_different_powers_6(x):
    d = len(x)
    result = 0
    for i in range(d):
        result += np.abs(x[i])**(i+2)
    return result

def sum_of_different_powers_8(x):
    d = len(x)
    result = 0
    for i in range(d):
        result += np.abs(x[i])**(i+2)
    return result
def sum_of_different_powers_10(x):
    d = len(x)
    result = 0
    for i in range(d):
        result += np.abs(x[i])**(i+2)
    return result
def sum_of_different_powers_12(x):
    d = len(x)
    result = 0
    for i in range(d):
        result += np.abs(x[i])**(i+2)
    return result

def trid_2(x):
    d = len(x)
    return (x[0] - 1)**2 + (x[1] - 1)**2 - x[0] * x[1]

def trid_5(x):
    d = len(x)
    result = 0
    for i in range(d):
        result += (x[i] - 1)**2
    for i in range(1, d):
        result -= x[i]*x[i-1]
    return result


def trid_10(x):
    d = len(x)
    result = 0
    for i in range(d):
        result += (x[i] - 1)**2
    for i in range(1, d):
        result -= x[i]*x[i-1]
    return result


def trid_15(x):
    d = len(x)
    result = 0
    for i in range(d):
        result += (x[i] - 1)**2
    for i in range(1, d):
        result -= x[i]*x[i-1]
    return result


def dixon_price_3(x):
    return (x[0] - 1)**2 + np.sum([i * (2*x[i]**2 - x[i-1])**2 for i in range(1, len(x))])
def dixon_price_5(x):
    return (x[0] - 1)**2 + np.sum([i * (2*x[i]**2 - x[i-1])**2 for i in range(1, len(x))])
def dixon_price_7(x):
    return (x[0] - 1)**2 + np.sum([i * (2*x[i]**2 - x[i-1])**2 for i in range(1, len(x))])
def dixon_price_9(x):
    return (x[0] - 1)**2 + np.sum([i * (2*x[i]**2 - x[i-1])**2 for i in range(1, len(x))])



def colville(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2 + 90*(x[3] - x[2]**2)**2 + (1 - x[2])**2 + \
           10.1*((x[1] - 1)**2 + (x[3] - 1)**2) + 19.8*(x[1] - 1)*(x[3] - 1)