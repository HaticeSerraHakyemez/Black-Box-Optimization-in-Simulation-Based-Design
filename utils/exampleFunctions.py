import numpy as np

function_ranges = {
    "Ackley": [(-32.768, 32.768), (-32.768, 32.768)],
    "Bukin": [(-15, -5), (-3, 3)],
    "Cross-in-Tray": [(-10, 10), (-10, 10)],
    "Drop-Wave": [(-5.12, 5.12), (-5.12, 5.12)],
    "Eggholder": [(-512, 512), (-512, 512)],
    "Gramacy & Lee": [(0.5, 2.5), (0.5, 2.5)],
    "Griewank": [(-600, 600), (-600, 600)],
    "Holder Table": [(-10, 10), (-10, 10)],
    "Levy": [(-10, 10), (-10, 10)],
    "Rastrigin": [(-5.12, 5.12), (-5.12, 5.12)],
    "Schaffer Function N. 2": [(-100, 100), (-100, 100)],
    "Schaffer Function N. 4": [(-100, 100), (-100, 100)],
    "Schwefel": [(-500, 500), (-500, 500)],
    "Shubert": [(-10, 10), (-10, 10)],
    "Bohachevsky": [(-100, 100), (-100, 100)],
    "Perm Function": [(0, 1)],
    "Rotated Hyper-Ellipsoid": [(-65.536, 65.536), (-65.536, 65.536)],
    "Sphere": [(-5.12, 5.12), (-5.12, 5.12)],
    "Sum of Different Powers": [(-1, 1)],
    "Sum Squares": [(-10, 10), (-10, 10)],
    "Trid": [(-10, 10), (-10, 10)],
    "Booth": [(-10, 10), (-10, 10)],
    "Matyas": [(-10, 10), (-10, 10)],
    "McCormick": [(-1.5, 4), (-3, 4)],
    "Power Sum": [(0, 1)],
    "Zakharov": [(-5, 10), (-5, 10)],
    "Three-Hump Camel": [(-5, 5), (-5, 5)],
    "Six-Hump Camel": [(-3, 3), (-2, 2)],
    "Dixon-Price": [(-10, 10), (-10, 10)],
    "Rosenbrock": [(-2.048, 2.048), (-2.048, 2.048)],
    "De Jong Function N. 5": [(-65.536, 65.536), (-65.536, 65.536)],
    "Easom": [(-100, 100), (-100, 100)],
    "Beale": [(-4.5, 4.5), (-4.5, 4.5)],
    "Branin": [(-5, 10), (0, 15)],
    "Colville": [(-10, 10), (-10, 10), (-10, 10), (-10, 10)],
    "Forrester": [(0, 1)],
    "Goldstein-Price": [(-2, 2), (-2, 2)],
    "Hartmann 3-D": [(0, 1), (0, 1)],
    "Hartmann 4-D": [(0, 1), (0, 1)],
    "Hartmann 6-D": [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)],
    "Perm": [(0, 1)],
    "Powell": [(-4, 5), (-4, 5), (-4, 5), (-4, 5)],
    "Shekel": [(0, 10)],
    "Styblinski-Tang": [(-5, 5), (-5, 5)]
}


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
    return 1 + (1/4000)*np.sum(x**2) - np.prod(np.cos(x/np.sqrt(np.arange(0, x.size))))

def holder_table(x):
    return -np.abs(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.abs(1 - np.sqrt(x[0]**2 + x[1]**2) / np.pi)))

def levy(x):
    return np.sum((1 + (x - 1) / 4) * (np.sin(np.pi * (1 + (x - 1) / 4)))**2) + (np.sin(2 * np.pi * (1 + (x[0] - 1) / 4)))**2

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def schaffer2(x):
    return 0.5 + ((np.sin(x[0]**2 - x[1]**2))**2 - 0.5) / ((1 + 0.001 * (x[0]**2 + x[1]**2))**2)

def schaffer4(x):
    return 0.5 + ((np.cos(np.sin(np.abs(x[0]**2 - x[1]**2))))**2 - 0.5) / ((1 + 0.001 * (x[0]**2 + x[1]**2))**2)

def schwefel(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def shubert(x):
    return np.prod([i * np.cos((i + 1) * x[0] + i) for i in range(1, 5 + 1)]) * \
           np.prod([i * np.cos((i + 1) * x[1] + i) for i in range(1, 5 + 1)])

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
    return (x[0] - 1)**2 + np.sum([i * (2*x[i]**2 - x[i-1])**2 for i in range(0, len(x))])

# Steep Ridges/Drops
def de_jong5(x):
    return np.sum(x**2)

def easom(x):
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi)**2 - (x[1] - np.pi)**2)

# Other
def beale(x):
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2

def branin(x, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    return a*(x[1] - b*x[0]**2 + c*x[0] - r)**2 + s*(1 - t)*np.cos(x[0]) + s

def colville(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2 + 90*(x[3] - x[2]**2)**2 + (1 - x[2])**2 + \
           10.1*((x[1] - 1)**2 + (x[3] - 1)**2) + 19.8*(x[1] - 1)*(x[3] - 1)