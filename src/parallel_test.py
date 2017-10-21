'''
Created on Oct 21, 2017

@author: Prateek Kolhar
'''
from processing import Pool
import numpy as np
import math 
import timeit



l = 100000000
x = np.zeros(l)
def f(i):
    i*i
    
def g():
    
    start = timeit.default_timer()
    p = Pool(10)    
    stop =timeit.default_timer()
    print "thread creation time="+str(stop-start)
    start = timeit.default_timer()
    p.map(f,range(2*l))
    stop =timeit.default_timer()
    print "parallel time="+str(stop-start)
def h():
    start = timeit.default_timer()
    for i in xrange(2*l):
        f(i)
    stop =timeit.default_timer()
    print "serial time="+str(stop-start)
  
if __name__ == '__main__':  
    g()
    h()

