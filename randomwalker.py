class position(object):
    def __init__(self, x, y):
        """x,y are float type"""
        # assigning the initial position
        self.x = x
        self.y = y

    def move(self,dx,dy):
        """dx,dy are float type: function to make a new position object at the new coordinates moved by (dx, dy)"""
        return position(self.x+dx, self.y+dy)

    def findX(self):
        """Give the x coordinate of the object"""
        return self.x

    def findY(self):
        """Give the y coordinate of the object"""
        return self.y

    def distance(self, other):
        """other is an object from position class: function will calculate their relative distance between self, and other"""
        delta_x = self.x - other.findX()
        delta_y = self.y - other.findY()
        return (delta_x**2+delta_y**2)**0.5


    def __str__(self):
        return "({},{})".format(self.x, self.y)
# we are going to pass this class to another classes below

class walker(object):
    def __init__(self, name= None):
        """assume name is a string"""
        self.name = name

    def __str__(self):
        if self.name != None:
            return self.name
        return "Unkown"
# Here we are going to make two types of walker:
# Normal walker: which has no preference for any directions.
# Biased walker: which has some bias toward a particular direction. (in our case in y direction)
import random

class Normal_walker(walker):
    def take_step(self):
        """Taking a random choice out of all the possible moves"""
        choices_of_steps = [(0,1), (1,0), (0,-1), (-1,0)]
        return random.choices(choices_of_steps)[0]

class Biased_walker(walker):
    """Taking a random choice out of all the possible moves"""
    def take_step(self):
        choices_of_steps = [(0,1.5), (1,0), (0,-0.5), (-1,0)]
        return random.choices(choices_of_steps)[0]

# Notice that we have the same name for take_step methods under different sub-classes of walker which is different when the class is different.
#
# Now we need to define a class for the space that we need to put the walkers in:
class Space(object):
    def __init__(self):
        self.walkers={}

    def addWalker(self, walker, pos):
        """Takes a walker and position class and will add it to our dictionary of walkers, if the walker does not already exist"""
        if walker in self.walkers:
            raise ValueError("Walker already exist")
        else:
            self.walkers[walker]=pos

    def getPos(self, walker):
        """Will take a walker class and give back the position class assigned to it"""
        if walker not in self.walkers:
            raise ValueError("No such Walker exist in our space!")
        return self.walkers[walker]

    def moveWalker(self, walker):
        """Take a walker class and dependent on what subclass was chosen in defining the walker, takes step"""
        if walker not in self.walkers:
            raise ValueError("No such Walker exist in our space!")
        Delta_x, Delta_y = walker.take_step()
        # moving the walker to new position (class)
        self.walkers[walker] = self.walkers[walker].move(Delta_x, Delta_y)


# Now that we built up our position, walker, and Space we can make a random walk:

def walk(space, walker, number_of_steps, log_pos=False):
    """ function for performing a random walk for a given walker
    INPUT:
    -------
          space is from Space cls
          walker is from Walker cls
          number_of_steps is integer>=0

    OUTPUT:
    -------
          IF log_pos == False:
                        Function will produce the distance between starting
                        position of the walker and the last location.

          IF log_pass == True:
                        Function will produce a list of all the positions
                        walker was during the walk.

    """
    # Find the initial postion of the walker in the space
    starting_position = space.getPos(walker)

    # Move the walker in the space
    save_all_pos = []
    for i in range(number_of_steps):
        pos_=space.getPos(walker)
        if log_pos:
            save_all_pos.append((pos_.findX(), pos_.findY()))
        space.moveWalker(walker)
    if log_pos:
        return save_all_pos
    return starting_position.distance(space.getPos(walker))
# In the following we are going to define a function to perform severel random walks:

def simulate_walks(number_of_steps, number_of_simulations, walker_class_type, origin=position(0,0)):
    """
    This is function that runs simulation for given variables:

    INPUT:
        number_of_steps: How many step the walker should take
        number_of_simulations: How many simulation to run
        walker_class_type: The type of walker class (a subclass of walker)
        origin: Should be an instance of the class position

    Output:
        A list of distances from origins
    """
    our_walker = walker_class_type("walker_1")
    distances=[]
    for i in range(number_of_simulations):
        space = Space()
        space.addWalker(our_walker, origin)
        distances.append(walk(space, our_walker, number_of_steps))
    return distances


def test_simulation(walk_length_array, number_of_simulations, walker_class_type):
    """
    Some sanity checks on the simulations
    """
    for walk_length in walk_length_array:
        _distances_ = simulate_walks(walk_length, number_of_simulations, walker_class_type)
        print(walker_class_type.__name__, " random walk of {} steps".format(walk_length), " After {} simulations".format(number_of_simulations))
        print(" Mean= {}".format(round(sum(_distances_)/len(_distances_),4)))
        print(" Max= {}".format(round(max(_distances_), 4)))
        print(" Min= {}".format(round(min(_distances_),4)))


test_simulation([0,1,2, 10**3, 10**5], 100, Biased_walker)