import math

G = 9.81

PI = math.pi

MC = 1.0 # [kg]
M1 = 0.1
M2 = 0.1
L1 = 0.5
L2 = 0.5
LC1 = L1
LC2 = L2
I1 = 0
I2 = 0
BC = 0.5
B1 = 0.001
B2 = 0.001

TARGETS    = [1.5, -1.5]
THRESHOLD  = 0.3
U_MIN, U_MAX = -50, 50

K_ENERGY   = 5.3
K_CART_P   = 8.0
K_CART_D   = 4.0

LOG_RATE = 1/10

# Initialization values
x0, x_dot0 = 0.0, 0.0
q10, q1dot0 = PI, 1.5
q20, q2dot0 = PI/4, 0