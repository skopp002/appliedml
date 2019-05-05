import matplotlib.pyplot as plt
import pdb
import numpy as np

class Needle(object):
    def __init__(self, needlelength):
        self.l = needlelength
        self.theta = np.random.uniform(0, 2 / (np.pi))
        self.x = (needlelength / 2) * np.sin(self.theta)

    def __str__(self):
         print("Needle - ",self.l , "(x = ",self.x, " theta = ", self.theta,")" )

# To define a needle, we need the length of the needle and the angle at which it falls on the grid
# The angle is theta which varies from 0 - Pi/2. The reason being all other angles can be converted to the 0 - 90 degrees
# with the symmetry of the grid in play.
# To make inspection simple, lets use needle length <10 for short needles.
class ShortNeedle(Needle):
   def __init__(self):
       l = np.random.uniform(3, 10, 1)
       Needle.__init__(self, l)


class LongNeedle(Needle):
    def __init__(self):
        l = np.random.uniform(10,15,1)
        Needle.__init__(self, l)

class Grid(object):
        def __init__(self,stripwidth=10):
            self.t = stripwidth

        def getNeedle(self, needle):
            """Will take a needle instance and give back the length and strip width"""
            if needle not in self.needles:
                raise ValueError("No such needle on the grid!")
            return self.needles[needle]


def simulate_needlethrow(number_of_needles, Needletype, choice):
    """
    """
    if(choice == 1):
        crossing = 0
        no_crossing = 0
        sheet = Grid()
        needlelength = 0 # Just to make the variable accessible outside
        for i in range(number_of_needles):
            needle = Needletype()
            needlelength = needle.l
            #Reference http://mathworld.wolfram.com/BuffonsNeedleProblem.html, for a longer needle, the probability is
            #2l/tpi - 2t/pi(sqrt(l^2 - t^2) + t/sin^(-1)(t/l)+1
            if (needle.x < np.random.uniform(1,sheet.t/2,1)):
              crossing = crossing + 1
                 #print(" There is a crossing on ", needle.x, " with ", needle.theta)
            else:
               no_crossing = no_crossing + 1
        P = crossing / number_of_needles
        #print("Number of crossing needles " , crossing , " Number of non-crossing needles ", no_crossing)
        return  (2 * needlelength )/(P * sheet.t) #π = 2l/Pt
    elif (choice == 2):
        crossing = 0
        no_crossing = 0
        l = int(input("enter needlelength"))
        t = int(input("enter stripwidth"))
        sheet = Grid(t)
        for i in range(number_of_needles):
            needle = Needletype(l)
            needlelength = needle.l
            if (needle.x < np.random.uniform(0,sheet.t/2,1)):
                crossing = crossing + 1
                #print(" There is a crossing on ", needle.x, " with ", needle.theta)
            else:
                no_crossing = no_crossing + 1
        P = crossing / number_of_needles
        pi_est = (2 * needlelength )/(P * sheet.t)
        print("Number of crossing needles ", crossing, " Number of non-crossing needles ", no_crossing, " pi is ", pi_est)
        return  pi_est #π = 2l/Pt



def test_simulation(simulations, NeedleType, choice):
    """
    Some sanity checks on the simulations
    """
    pi_est=[]
    noOfSim = []
    for i in range(1000, simulations):
        pi_est.append(simulate_needlethrow(i, NeedleType, choice))
        noOfSim.append(i)
    return (pi_est, noOfSim)


#pdb.set_trace()
if __name__ == '__main__':
  choice = int(input("Choose 1 - for General or 2 - for Specific"))
  simulations = 3000
  if(choice == 1):
     (pi_short, shortsimarr) = test_simulation(simulations, ShortNeedle, choice)
     pi_short_mean = np.mean(pi_short)
     (pi_long, longsimarr) = test_simulation(simulations, LongNeedle, choice)
     pi_long_mean = np.mean(pi_long)
     print("Mean with short needle ", pi_short_mean, " Mean with LongNeedle ", pi_long_mean )
     plt.figure()
     plt.subplot(2,2,1)
     plt.plot(np.array(shortsimarr),np.array(pi_short))
     plt.show()
     plt.subplot(2,2,2)
     plt.plot(np.array(longsimarr),np.array(pi_long))
     plt.show()
  elif(choice == 2):
     test_simulation(simulations, Needle, choice)
