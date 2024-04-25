# Solar System Simulation

We can make a simple model solar system by considering $n$ bodies with masses $m_i$ for $i \in \{ 1, \dots, n \}$ by considering just the gravitational forces that these bodies induce on each other and using this to determine how they will move.

## Newtons Laws

Let $x_i(t)$ denote the position of body $i$ at time $t$. We can start by assuming a planar system and picking some reference time $t = 0$ to simulate from. We will need to know the initial positions $x_i(0)$ and initial velocities $v_i(0)$ for this system to be deterministic.

New tons law of gravitation states that the force that the other bodies exert on body $i$ is
$$F_i = - \sum_{j \ne i} G \frac{m_i m_j}{|x_i - x_j|^3} (x_i - x_j)$$
Here there is an implicit dependence on the time $t$. Here $G$ is [Newtons constant of Gravitation](https://en.wikipedia.org/wiki/Gravitational_constant). Then by Newtons second law, body $i$ has acceleration
$$a_i = \frac{F_i}{m_i}$$

## Simulating Trajectories 

Now that we know how the bodies are accelerating, we can calculate their positions and velocities after some short time $\Delta t$ as
$$x_i(t + \Delta t) = x_i(t) + \Delta t v_i(t)$$
$$v_i(t + \Delta t) = v_i(t) + \Delta t \frac{F_i(t)}{m_i}$$
This is known as the [Forward Euler](https://en.wikipedia.org/wiki/Euler_method#:~:text=In%20mathematics%20and%20computational%20science,with%20a%20given%20initial%20value.) method. 

If we fix some small time increment $\Delta t > 0$ and discretise the time as $t_j = j \Delta t$. Then writing
$$x_i^{(j)} := x_i(t_j)$$
$$v_i^{(j)} := v_i(t_j)$$
we get that
$$x_i^{(j+1)} = x_i^{(j)} + \Delta t v_i^{(j)}$$
$$v_i^{(j+1)} = v_i^{(j)} + \Delta t \frac{F_i^{(j)}}{m_i}$$
We can iterate these calculations to approximate how the bodies will move over time. The smaller we make $\Delta t$, the more accurate the approximation.

## Extensions

We could try doing this in 3D. The formulae will be the same but more thought will be put into how to display the system.
