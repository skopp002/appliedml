import time
import matplotlib.pyplot as plt
import numpy as np

def fibrec(n=0):
     if n == 0:
        return 1
     elif n == 1:
        return 1
     else:
        return fibrec(n-1)+fibrec(n-2)



def fibfor(length=0):
    # Program to generate the Fibonacci sequence in Python
    #If length is 0, return 1 else return fibonacci sequence that far
    # The first two values
    x = 0
    y = 1
    # Condition to check if the length has a valid input
    if length <= 1:
        return (x)
    else:
        for i in range(length):
            z = x + y
            # Modify values
            x = y
            y = z
        return z


def timetrack(n=0):
    startRec = time.time()
    fibresult = fibrec(n)
    print("Recursive response", fibresult)
    endRec = time.time()
    print("Recursive method completed")
    startfor = time.time()
    forres = fibfor(n)
    print("Value for for is ", forres)
    endfor = time.time()
    print("Time taken for recursive method execution is ", (endRec - startRec))
    print("Time taken for loop based method is ", (endfor - startfor))

def performancechart(nval=30):
    r1=[]
    f1=[]
    r2=[]
    f2=[]
    for i in range(nval):
        rstart=time.time()
        fibrec(i)
        rend=time.time()
        r1.append(i)
        r2.append(rend - rstart)
        fstart=time.time()
        fibfor(i)
        fend=time.time()
        f1.append(i)
        f2.append(fend - fstart)

    return (r1,r2, f1, f2)



timetrack(10)
nvalForGraph=30
(r1,r2,f1,f2) = performancechart(nvalForGraph)
plt.plot(r1,r2)
plt.show()
x=np.linspace(0,10,nvalForGraph)
plt.figure()
plt.subplot(2,2,1)
plt.ylabel("timetaken")
plt.plot(x,r2,'ro')
plt.title('recursion time taken')
plt.subplot(2,2,2)
plt.plot(x,f2,'go')
plt.title('for time taken')
plt.show()


plt.show()