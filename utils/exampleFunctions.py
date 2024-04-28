import numpy as np

# Many Local Minima
def ackley(x):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - \
           np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))) + np.e + 20

def bukin(x):
    return 100 * np.sqrt(np.abs(x[1] - 0.01 * x[0]**2)) + 0.01 * np.abs(x[0] + 10)

def cross_in_tray(x):
    return -0.0001 * (np.abs(np.sin(x[0]) * np.sin(x[1]) * np.exp(np.abs(100 - np.sqrt(x[0]**2 + x[1]**2) / np.pi))) + 1)**0.1

def drop_wave(x):
    return -(1 + np.cos(12 * np.sqrt(x[0]**2 + x[1]**2))) / (0.5 * (x[0]**2 + x[1]**2) + 2)

def eggholder(x):
    return -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[0]/2 + (x[1] + 47)))) - x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47))))

def gramacy_lee(x):
    return np.sin(10*np.pi*x) / (2*x) + (x-1)**4

def griewank(x):
    return 1 + (1/4000)*np.sum(x**2) - np.prod(np.cos(x/np.sqrt(np.arange(1, x.size + 1))))

def holder_table(x):
    return -np.abs(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.abs(1 - np.sqrt(x[0]**2 + x[1]**2) / np.pi)))

def levy(x):
    return np.sum((1 + (x - 1) / 4) * (np.sin(np.pi * (1 + (x - 1) / 4)))**2) + (np.sin(2 * np.pi * (1 + (x[0] - 1) / 4)))**2

def levy13(x):
    w = 1 + (x - 1) / 4
    return (np.sin(np.pi * w[0]))**2 + np.sum((w[:-1] - 1)**2 * (1 + 10 * (np.sin(np.pi * w[:-1] + 1))**2)) + \
           (w[-1] - 1)**2 * (1 + (np.sin(2 * np.pi * w[-1]))**2)

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def schaffer2(x):
    return 0.5 + ((np.sin(x[0]**2 - x[1]**2))**2 - 0.5) / ((1 + 0.001 * (x[0]**2 + x[1]**2))**2)

def schaffer4(x):
    return 0.5 + ((np.cos(np.sin(np.abs(x[0]**2 - x[1]**2))))**2 - 0.5) / ((1 + 0.001 * (x[0]**2 + x[1]**2))**2)

def schwefel(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def shubert(x):
    return np.prod([i * np.sin((i + 1) * x[0] + i) for i in range(1, len(x) + 1)]) * \
           np.prod([i * np.sin((2 * i) * x[1] + i) for i in range(1, len(x) + 1)])

# Bowl-Shaped
def booth(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

def matyas(x):
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

def mccormick(x):
    return np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1

def power_sum(x, b=[8, 18, 44, 114, 274]):
    return np.sum((x - np.array(b))**2)

def zakharov(x):
    return np.sum(x**2) + (0.5 * np.sum(x))**2 + (0.5 * np.sum(x))**4

# Valley-Shaped
def three_hump_camel(x):
    return 2*x[0]**2 - 1.05*x[0]**4 + (x[0]**6)/6 + x[0]*x[1] + x[1]**2

def six_hump_camel(x):
    return (4 - 2.1*x[0]**2 + (x[0]**4)/3) * x[0]**2 + x[0]*x[1] + (-4 + 4*x[1]**2) * x[1]**2

def dixon_price(x):
    return (x[0] - 1)**2 + np.sum([i * (2*x[i]**2 - x[i-1])**2 for i in range(1, len(x))])

def rosenbrock(x):
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# Steep Ridges/Drops
def de_jong5(x):
    return np.sum(x**2)

def easom(x):
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi)**2 - (x[1] - np.pi)**2)

def michalewicz(x, m=10):
    return -np.sum(np.sin(x) * (np.sin(np.arange(1, len(x) + 1) * x**2 / np.pi)**(2*m)))

# Other
def beale(x):
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2

def branin(x, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    return a*(x[1] - b*x[0]**2 + c*x[0] - r)**2 + s*(1 - t)*np.cos(x[0]) + s

def colville(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2 + 90*(x[3] - x[2]**2)**2 + (1 - x[2])**2 + \
           10.1*((x[1] - 1)**2 + (x[3] - 1)**2) + 19.8*(x[1] - 1)*(x[3] - 1)