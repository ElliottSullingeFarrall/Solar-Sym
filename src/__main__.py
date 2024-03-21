from scipy.constants import G

class Body:
    def __init__(self, x, v, r, m):
        self.x = x
        self.v = v
        self.r = r
        self.m = m

class System:
    def __init__(self, bodies):
        self.bodies = bodies