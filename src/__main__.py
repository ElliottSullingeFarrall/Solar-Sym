from astroquery.jplhorizons import Horizons
from astropy.time import Time

from tqdm import tqdm

from scipy.constants import G

from numpy import array
from numpy.linalg import norm

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

BODY_SCALE = 250

DAY = 24 * 60 * 60

TIME_START = '2022-01-01 00:00:00'
TIME_END  = 1000 * DAY
TIME_STEP = 60
NUM_STEPS = int(TIME_END / TIME_STEP)

CANVAS_SIZE = 250e9
STEPS_PER_FRAME = 5000

class Body:
    def __init__(self, name, colour, x, v, r, m):
        self.name = name
        self.colour = colour

        self.x = array(x)
        self.v = array(v)
        self.r = r
        self.m = m

    def download_data(self):
        obj = Horizons(id=self.name, location='@sun', id_type=None, epochs=Time(TIME_START, format='iso', scale='utc').jd)
        eph = obj.ephemerides()

        self.x = array([eph['x'][0], eph['y'][0]])
        self.v = array([eph['vx'][0], eph['vy'][0]])
        self.r = eph['radius'][0]
        self.m = eph['mass'][0]

    def force(self, *others):
        return sum(- G * self.m * other.m / norm(self.x - other.x)**3 * (self.x - other.x) for other in others)

class System:
    def __init__(self, *bodies):
        self.t = 0
        self.bodies = {body.name: body for body in bodies}

    def step(self): # Forward Euler
        for body in self.bodies.values():
            body.x_new = body.x + TIME_STEP * body.v
            body.v_new = body.v + TIME_STEP * body.force(*(other for other in self.bodies.values() if other is not body)) / body.m

        self.t += TIME_STEP
        for body in self.bodies.values():
            body.x = body.x_new
            body.v = body.v_new

    def draw(self):
        R = max(norm(body.x) for body in self.bodies.values())

        fig, ax = plt.subplots()
        fig.set_facecolor('black')
        ax.set_facecolor('black')
        ax.set_xlim(-CANVAS_SIZE, +CANVAS_SIZE)
        ax.set_ylim(-CANVAS_SIZE, +CANVAS_SIZE)
        ax.axis('off')

        unit_convert = lambda s: ax.transData.transform([s,0])[0] - ax.transData.transform([0,0])[0] 

        scatter = ax.scatter(
            [body.x[0] for body in self.bodies.values()],
            [body.x[1] for body in self.bodies.values()],
            [unit_convert(body.r * BODY_SCALE) for body in self.bodies.values()],
            [body.colour for body in self.bodies.values()]
        )
        time_label = ax.text(0.02, 0.95, '', color='white', transform=ax.transAxes)

        def update(frame):
            for _ in range(STEPS_PER_FRAME):
                self.step()

            scatter.set_offsets(list(zip(
                [body.x[0] for body in self.bodies.values()],
                [body.x[1] for body in self.bodies.values()]
            )))
            time_label.set_text(f'DAY {round(self.t / DAY)}')

            return scatter, time_label
        
        anim = FuncAnimation(fig, update, tqdm(range(int(NUM_STEPS / STEPS_PER_FRAME))), interval=0.1, blit=True, cache_frame_data=False)
        anim.save('simulation.gif', writer='pillow')


if __name__ == '__main__':
    sun     = Body(
        name = 'sun',
        colour = 'yellow',
        x = [0, 0],
        v = [0, 0],
        r = 696340e3,
        m = 1.989e30
    )
    mercury = Body(
        name = 'mercury',
        colour = 'grey',
        x = [0, 57909175e3],
        v = [47000, 0],
        r = 2439.7e3,
        m = 3.285e23
    )
    venus   = Body(
        name = 'venus',
        colour = 'orange',
        x = [0, 108208930e3],
        v = [35000, 0],
        r = 6051.8e3,
        m = 4.867e24
    )
    earth   = Body(
        name = 'earth',
        colour = 'blue',
        x = [0, 149598261e3],
        v = [30000, 0],
        r = 6371e3,
        m = 5.972e24
    )
    mars    = Body(
        name = 'mars',
        colour = 'red',
        x = [0, 227939100e3],
        v = [24000, 0],
        r = 3389.5e3,
        m = 6.39e23
    )
    jupiter = Body(
        name = 'jupiter',
        colour = 'orange',
        x = [0, 778547200e3],
        v = [13000, 0],
        r = 69911e3,
        m = 1.898e27
    )
    saturn  = Body(
        name = 'saturn',
        colour = 'yellow',
        x = [0, 1433449370e3],
        v = [9000, 0],
        r = 58232e3,
        m = 5.683e26
    )
    uranus  = Body(
        name = 'uranus',
        colour = 'cyan',
        x = [0, 2876679082e3],
        v = [6835, 0],
        r = 25362e3,
        m = 8.681e25
    )
    neptune = Body(
        name = 'neptune',
        colour = 'blue',
        x = [0, 4503443661e3],
        v = [5477, 0],
        r = 24622e3,
        m = 1.024e26
    )
    pluto   = Body(
        name = 'pluto',
        colour = 'grey',
        x = [0, 5913520000e3],
        v = [4748, 0],
        r = 1188.3e3,
        m = 1.303e22
    )

    for body in (mercury, venus, earth, mars, jupiter, saturn, uranus, neptune, pluto):
        body.download_data()

    system = System(sun, mercury, venus, earth, mars, jupiter, saturn, uranus, neptune, pluto)
    system.draw()

    