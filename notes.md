# Taichi Learning Notes

## Working Environment
- **CPU** AMD 7950X
- **GPU** AMD 7900XT
- **OS** Windows 10
- **Python version** 3.10.11

## Implementation of Spring-Damper System
This implementation is a naive way to calculate a spring-damper system on a 1-D line. 
```python
import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

damping_constant = 0.02
spring_constant = 0.2
time_step = 0.5

damping_force = ti.field(dtype=float, shape=())
spring_force = ti.field(dtype=float, shape=())
spring_position = ti.field(dtype=float, shape=())
position = ti.field(dtype=float, shape=())
velocity = ti.field(dtype=float, shape=())
mass = ti.field(dtype=float, shape=())

@ti.kernel
def compute_state():
    force = damping_force[None] + spring_force[None]
    velocity[None] = velocity[None] + force / mass[None]
    position[None] = position[None] + velocity[None] * time_step
    damping_force[None] = - damping_constant * velocity[None]
    spring_force[None] = spring_constant * (spring_position[None] - position[None])


def main():
    gui = ti.GUI("Autodiff gravity")
    spring_position[None] = 0.5
    position[None] = 0.2
    mass[None] = 1.0
    while gui.running:
        compute_state()
        gui.circles(np.array([[position[None], 0.5]], dtype=float), radius=3)
        gui.show()


if __name__ == "__main__":
    main()
```

## Taichi implementation for Fluid Simulation
See main.ipynb