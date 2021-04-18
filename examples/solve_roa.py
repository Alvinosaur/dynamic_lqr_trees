import math
from pydrake.all import Jacobian, MathematicalProgram, Solve, Variables, SymbolicVectorSystem, RegionOfAttraction
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

# Continuous double-integrator dynamics
A = np.array([[0., 1],
              [0., 0]])
B = np.array([[0],
              [1.]])

# LQR costs
Q = np.array([[1., 0.],
            [0., 1.]])
R = np.array([[1.]])

# initial conditions
x0 = np.array([[3, 0.0]]).T
u0 = np.array([0.0])

# goal conditions
xg = np.array([[1.0, 0]]).T

# solve continuous LQR 
S = linalg.solve_continuous_are(A, B, Q, R)
K = linalg.inv(R) @ B.T @ S

# Plotting trajectory
# N = 10
# xs = np.zeros((N+1, 2))
# xs[0, :] = x0.flatten()
# x = np.copy(x0)

# for i in range(1, N+1):
#     u = -K @ (x-xg)
#     x += A @ x + B @ u
#     xs[i, :] = x.flatten()

# time_history = list(range(N+1))

# plt.plot(time_history, xs[:, 0], "-b", label="x1")
# plt.plot(time_history, xs[:, 1], "-g", label="x2")

# xg0_h = [xg[0] for i in range(N+1)]
# xg1_h = [xg[1] for i in range(N+1)]
# plt.plot(time_history, xg0_h, "--b", label="target x1")
# plt.plot(time_history, xg1_h, "--g", label="target x2")

# plt.show()

#################### Method 1 ##########################
prog = MathematicalProgram()
x = prog.NewIndeterminates(2, 1, "x")
xbar = x - xg

# Define the dynamics and Lyapunov function.
xdot = A @ (xbar + xg) + B*(-K @ xbar)
V = (xbar.T @ S @ xbar)[0,0]

# Vdot = Jacobian([V], x).dot(xdot)[0,0]
Vdot = (2*xbar.T @ S @ xdot)[0,0]

rho = prog.NewContinuousVariables(1, "rho")[0]
prog.AddLinearCost(-rho)

h = prog.NewFreePolynomial(Variables(x), 2).ToExpression()
# prog.AddSosConstraint(((V - rho)*h + Vdot))
prog.AddSosConstraint(((V - rho) * (xbar.T @ xbar) - h * Vdot)[0,0])

result = Solve(prog)
print(f"Success: {result.is_success()}, rho = {result.GetSolution(rho)}")

######################### Method 2 #############################
prog = MathematicalProgram()
x = prog.NewIndeterminates(2, 1, "x")
xbar = x - xg
f = A @ x + B*(-K @ xbar)
sys = SymbolicVectorSystem(state=x, dynamics=f)
context = sys.CreateDefaultContext()
V = RegionOfAttraction(system=sys, context=context)
print(f"Success: {result.is_success()}, rho = {result.GetSolution(rho)}")
