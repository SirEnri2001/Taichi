import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

damping_constant = 0.02
spring_constant = 0.2
time_step = 0.5

damping_force = ti.field(dtype=float, shape=(), needs_grad=True)
spring_force = ti.field(dtype=float, shape=(), needs_grad=True)
force = ti.field(dtype=float, shape=(), needs_grad=True)
spring_position = ti.field(dtype=float, shape=(), needs_grad=True)
position = ti.field(dtype=float, shape=(), needs_grad=True)
velocity = ti.field(dtype=float, shape=(), needs_grad=True)
mass = ti.field(dtype=float, shape=())


@ti.kernel
def compute_forces():
    damping_force[None] = - damping_constant * velocity[None]
    spring_force[None] = spring_constant * (spring_position[None] - position[None])
    force[None] = damping_force[None] + spring_force[None]


@ti.kernel
def compute_state():
    velocity[None] = velocity[None] + force[None] / mass[None]
    position[None] = position[None] + velocity[None] * time_step


def main():
    gui = ti.GUI("Autodiff gravity")
    spring_position[None] = 0.5
    position[None] = 0.2
    mass[None] = 1.0
    while gui.running:
        compute_forces()
        compute_state()
        gui.circles(np.array([[position[None], 0.5]], dtype=float), radius=3)
        gui.show()


if __name__ == "__main__":
    main()
