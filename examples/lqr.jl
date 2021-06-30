using ControlSystems
using LinearAlgebra # For identity matrix I
# Dynamics: 
# x=Ax+Bu
# y=Cx+Du
h       = 0.1
A       = [1 h; 0 1]
B       = [0 1]' # To handle bug TODO
C       = [1 0]
# define state space model  https://www.mathworks.com/help/control/ref/ss.html
sys     = ss(A,B,C,0, h)
Q       = I
R       = I
# L       = dlqr(A,B,Q,R) # lqr(sys,Q,R) can also be used
S = dare(A, B, Q, R)
K = (B'*S*B + R)\(B'S*A)

# Form control law lambda func (u is a function of t and x), a constant input disturbance 
# is affecting the system from t≧2.5
u(x,t)  = -K*x .+ 1.5(t>=2.5)

t       =0:h:5
x0      = [1,0]
y, t, x, uout = lsim(sys,u,t,x0=x0)
Plots.plot(t,x, lab=["Position" "Velocity"], xlabel="Time [s]")