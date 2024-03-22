from astroquery.jplhorizons import Horizons
from astropy.time import Time

from datetime import datetime, timedelta

from tqdm import tqdm

from scipy.constants import day, au, G

from numpy import array
from numpy.linalg import norm

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

BODY_SCALE = 250
CANVAS_SIZE = 2.5e11
STEPS_PER_FRAME = 5000

TIME_END  = 500 * day
TIME_STEP = 60

class Body:
    def __init__(self, colour, r, m, id=None, name=None, x=None, v=None):
        self.colour = colour
        self.r = r
        self.m = m

        if id is not None:
            self.id = id
        elif name is not None and x is not None and v is not None: 
            self.name = name
            self.x = array(x)
            self.v = array(v)
        else:
            raise ValueError('Missing required arguments')
        

    def force(self, *others):
        return sum(- G * self.m * other.m / norm(self.x - other.x)**3 * (self.x - other.x) for other in others)

class System:
    def __init__(self, *bodies):
        self.t = datetime.now()

        for body in bodies:
            if body.id is not None:
                dat = Horizons(id=body.id, location='@SSB', epochs=Time(self.t.isoformat(), format='isot', scale='utc').jd)
                eph = dat.ephemerides()
                vec = dat.vectors()

                body.name = eph['targetname'][0]
                body.x = array([vec['x'][0], vec['y'][0]])      * au
                body.v = array([vec['vx'][0], vec['vy'][0]])    * au / day
        self.bodies = {body.name: body for body in bodies}

    def step(self): # Forward Euler
        for body in self.bodies.values():
            body.x_new = body.x + TIME_STEP * body.v
            body.v_new = body.v + TIME_STEP * body.force(*(other for other in self.bodies.values() if other is not body)) / body.m

        self.t += timedelta(seconds=TIME_STEP)
        for body in self.bodies.values():
            body.x = body.x_new
            body.v = body.v_new

    def draw(self):
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
        date = ax.text(0.02, 0.95, '', color='white', fontsize=8, transform=ax.transAxes)

        def update(frame):
            for _ in range(STEPS_PER_FRAME):
                self.step()

            scatter.set_offsets(list(zip(
                [body.x[0] for body in self.bodies.values()],
                [body.x[1] for body in self.bodies.values()]
            )))
            date.set_text(f'DATE {self.t.strftime("%Y-%m-%d")}') 

            return scatter, date

        anim = FuncAnimation(fig, update, tqdm(range(int(TIME_END / TIME_STEP / STEPS_PER_FRAME))), blit=True, cache_frame_data=False)
        anim.save('simulation.gif', writer='pillow', dpi=450, fps=60)


if __name__ == '__main__':
    sun     = Body(
        id = '10',
        colour = 'yellow',
        r = 695700e3,
        m = 1988500e24
    )
    mercury = Body(
        id = '199',
        colour = 'grey',
        r = 2439.4e3,
        m = 3.302e23
    )
    venus   = Body(
        id = '299',
        colour = 'orange',
        r = 6051.84e3,
        m = 48.685e24
    )
    earth   = Body(
        id = '399',
        colour = 'blue',
        r = 6371.01e3,
        m = 5.97219e24
    )
    mars    = Body(
        id = '499',
        colour = 'red',
        r = 3389.92e3,
        m = 6.4171e23
    )
    jupiter = Body(
        id = '599',
        colour = 'orange',
        r = 69911e3,
        m = 1.89818722e27
    )
    saturn  = Body(
        id = '699',
        colour = 'yellow',
        r = 58232e3,
        m = 5.6834e26
    )
    uranus  = Body(
        id = '799',
        colour = 'cyan',
        r = 25362e3,
        m = 8.681e25
    )
    neptune = Body(
        id = '899',
        colour = 'blue',
        r = 25362e3,
        m = 86.813e24
    )
    pluto   = Body(
        id = '999',
        colour = 'grey',
        r = 1188.3e3,
        m = 1.307e22
    )

    system = System(sun, mercury, venus, earth, mars, jupiter, saturn, uranus, neptune, pluto)
    system.draw()

    