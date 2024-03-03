import time

import taichi as ti
import taichi.math as tm

ti.init(arch=ti.vulkan)

shape = [512,512]

color_field = ti.Vector.field(3, float, shape=(shape[0], shape[1]))

time_step = 0

@ti.kernel
def set_color(color_field: ti.template(), current_time: ti.f32):
    time_step = abs(tm.fract(current_time/100.0)-0.5)
    color_field.fill(0.0)
    for i, j in color_field:
        # color_field[i, j] = ti.Vector([i / shape[0], j / shape[1], 0])
        color_field[i, j] = ti.Vector([i / shape[0], j / shape[1], time_step])


window = ti.GUI("Taichi", res=(shape[0], shape[1]))


while window.running:
    time_step = time_step + 1
    set_color(color_field, time_step)
    window.set_image(color_field)
    window.show()
