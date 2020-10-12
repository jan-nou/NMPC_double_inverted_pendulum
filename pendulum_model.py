from casadi import *
from scipy import constants

import numpy as np

# constants
m0 = 0.6  # kg, mass of the cart
m1 = 0.2  # kg, mass of the first rod
m2 = 0.2  # kg, mass of the second rod
l1 = 0.5  #m, length of the first rod
l2 = 0.5  #m, length of the second rod

d1 = m0 + m1 + m2
d2 = (m1 / 2 + m2) * l1
d3 = m2 * l2 / 2
d4 = (m1 / 3 + m2) * l1**2
d5 = m2 * l1 * l2 / 2
d6 = m2 / 3 * l2**2

f1 = (m1 / 2 + m2) * l1 * constants.g
f2 = m2 / 2 * l2 * constants.g

# symbolic expressions: state-representation
x_cart = SX.sym('x_cart')
theta = SX.sym('theta', 2)
x_cart_dot = SX.sym('x_cart_dot')
theta_dot = SX.sym('theta_dot', 2)
x = vertcat(x_cart, theta, x_cart_dot, theta_dot)
#print('state vector: {} with shape {}'.format(x, x.shape))
# symbolic expressions: input & output
u = SX.sym('u')
y = SX.sym('y')

# construction of ODEs

#matrix D
D1 = horzcat(d1, d2 * cos(theta[0]), d3 * cos(theta[1]))
D2 = horzcat(d2 * cos(theta[0]), d4, d5 * cos(theta[0] - theta[1]))
D3 = horzcat(d3 * cos(theta[1]), d5 * cos(theta[0] - theta[1]), d6)
D = vertcat(D1, D2, D3)

#matrix C
C1 = horzcat(0, -d2 * sin(theta[0]) * theta_dot[0],
             -d3 * sin(theta[1]) * theta_dot[1])
C2 = horzcat(0, 0, d5 * sin(theta[0] - theta[1]) * theta_dot[1])
C3 = horzcat(0, -d5 * sin(theta[0] - theta[1]) * theta_dot[0], 0)
C = vertcat(C1, C2, C3)

#vector H*u-G
G = vertcat(u, f1 * sin(theta[0]), f2 * sin(theta[1]))

#reduction of order
sz = D.shape
zero_mat = np.zeros((sz))
id_mat = np.eye(sz[0])

A = vertcat(horzcat(id_mat, zero_mat), horzcat(zero_mat, D))
B = vertcat(horzcat(zero_mat, id_mat), horzcat(zero_mat, -C))
F = vertcat(np.zeros((3, 1)), G)

# x_dot = f(x) * x + f(x,u)
eq = inv(A) @ B @ x + inv(A) @ F

#print('ODEs: {}'.format(eq))

# functions describing state-space
systemdynamics = Function("systemdynamics", [x, u], [eq])
output = Function("output", [x], [x])  # we observe the original state

# ODE describing state-sapce
ode = {'x': x, 'ode': eq, 'p': u}

# Energy
J1 = (m1 * (l1 / 2)**2) / 3
J2 = (m2 * (l2 / 2)**2) / 3

h4 = m1 * (l1 / 2)**2 + m2 * l1**2 + J1
h5 = m2 * (l2 / 2) * l1
h6 = m2 * (l2 / 2)**2 + J2
h7 = m1 * (l1 / 2) * constants.g + m2 * l1 * constants.g
h8 = m2 * (l2 / 2) * constants.g

E_up = h7 + h8
E_down = -h7 - h8

T1 = 1 / 2 * m0 * x[3]**2
T2 = 1 / 2 * m1 * ((x[3] + (l1 / 2) * x[4] * cos(x[1]))**2 +
                   ((l1 / 2) * x[4] * sin(x[1]))**2) + 1 / 2 * J1 * x[4]**2
T3 = 1 / 2 * m2 * ((x[3] + l1 * x[4] * cos(x[1]) +
                    (l2 / 2) * x[5] * cos(x[2]))**2 +
                   (l1 * x[4] * sin(x[1]) +
                    (l2 / 2) * x[5] * sin(x[2]))**2) + 1 / 2 * J2 * x[4]**2

T_ges = T1 + T2 + T3

V_ges = 0 + m1 * constants.g * (l1 / 2) * cos(
    x[1]) + m2 * constants.g * (l1 * cos(x[1]) + (l2 / 2) * cos(x[2]))


def distance(xz):
    x_distance = vertcat(x_cart - xz[0], sin(0.5 * (x_cart - xz[1])),
                         sin(0.5 * (x_cart - xz[2])), x_cart_dot - xz[3],
                         fabs(theta_dot - xz[4:5]))

    return x_distance