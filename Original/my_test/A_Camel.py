import math


def Camel1D(x,a):
    return math.log(0.5/(a*math.sqrt(math.pi)) * (math.exp(-(x-1./3)*(x-1/3.)/(a*a)) + math.exp(-(x-2./3)*(x-2/3.)/(a*a))))

def CamelND(x,a=0.1,n=1):
    sum1 = 0
    sum2 = 0
    denom = 1
    for i in range(n):
        sum1 += -(x[i]-1./3)*(x[i]-1/3.)/(a*a)
        sum2 += -(x[i]-2./3)*(x[i]-2/3.)/(a*a)
        denom *= (a*math.sqrt(math.pi))
    sum1 = math.exp(sum1)
    sum2 = math.exp(sum2)
    sum = math.log((0.5/denom) * (sum1+sum2))
    return sum

def tfCamelND(x,a=0.1,n=1):
    sum1 = 0
    sum2 = 0
    denom = 1
    for i in range(n):
        sum1 += -(x[i]-1./3)*(x[i]-1/3.)/(a*a)
        sum2 += -(x[i]-2./3)*(x[i]-2/3.)/(a*a)
        denom *= (a*math.sqrt(math.pi))
    sum1 = K.exp(sum1)
    sum2 = K.exp(sum2)
    sum = (0.5/denom) * (sum1+sum2)
    #print(sum)
    return sum

