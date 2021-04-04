using Altro
using TrajectoryOptimization
using StaticArrays, LinearAlgebra
using RobotDynamics
using PyPlot
using DynamicPolynomials
using ControlSystems
using SumOfSquares

# import ProxSDP
import Mosek
import MosekTools

# stay in double integrator for now as long as possible
# only

# DO NOT USE QUADROTOR MODEL IN ROBOTZOO.jl b/c complicated quaternions
# use as simple as possible quadrotor model  
# simple nonlinear quadrotor model 

# specify dt = 1 sec for double integrator, N
# tf is time between those knot points 
# keep time btwn knot points as small as possible 
# dt = tf / (N - 1)

# u0 = SVector{2,Float64}([0.0, 0.001] + rand(2) * 0.01)
# U0 = [u0 for k = 1:N-1]

struct DoubleIntegrator{} <: TrajectoryOptimization.AbstractModel
    Ac::Any
    Ad::Any
    B::Any
    n::Integer  # state space size
    m::Integer  # ctrl space size

    function DoubleIntegrator()
        # continuous
        Ac = [
            zeros(2, 2) I(2)
            zeros(2, 2) zeros(2, 2)
        ]

        # discrete 
        Ad = [
            I(2)     I(2)
            zeros(2, 2) I(2)
        ]
        B = [
            zeros(2, 2)
            I(2)
        ]
        n = 4
        m = 2
        new(Ac, Ad, B, n, m)
    end
end

# continuous dynamics
function RobotDynamics.dynamics(model::DoubleIntegrator, x, u)
    model.Ac * x + model.B * u
end

Base.size(::DoubleIntegrator) = 4, 2
RobotDynamics.state_dim(model::DoubleIntegrator) = 4
RobotDynamics.control_dim(model::DoubleIntegrator) = 2

u_lower_bounds = 1 * [-1, -1]
u_upper_bounds = 1 * [1, 1]
x_lower_bounds = [-0.2, -0.2, -1, -1]
x_upper_bounds = [10.0, 10, 1, 1]
x0 = @SVector [0.1, 0.1, -1.0, -0.5]
xg = @SVector [0.9, 0.9, 0.0, 0.0]

model = DoubleIntegrator()
n, m = size(model)

N = 100
tf = 20.0
dt = 0.2

# discrete dynamics for LQR verification
Ad = model.Ad
Ac = model.Ac
B = model.B
Q = 1.0 * Diagonal([1, 1, 1, 1])
Qf = 10.0 * Diagonal([1, 1, 1, 1])
R = 1 * Diagonal(ones(m))

r = [0.2]
cx = [0.5]
cy = [0.6]
xi = 1
yi = 2
obstacles = TrajectoryOptimization.CircleConstraint(model.n, cx, cy, r, xi, yi)

conSet = ConstraintList(n, m, N)
# xg constraint not really necessary since LQR objective already minimizes distance with xg
# add_constraint!(conSet, GoalConstraint(xg), N:N)

add_constraint!(conSet, obstacles, 1:N)

add_constraint!(
    conSet,
    BoundConstraint(n, m, x_min = x_lower_bounds, x_max = x_upper_bounds,
     u_min=u_lower_bounds, u_max=u_upper_bounds),
    1:N-1,  # do not include Nth knot, the terminal since somehow affects controls, which aren't defined at the terminal knot
)
plt.clf()
obj = LQRObjective(Q, R, Qf, xg, N)
prob = Problem(model, obj, xg, tf, x0 = x0, constraints = conSet)

altro = ALTROSolver(prob);
solve!(altro);
X = states(altro)
U = controls(altro)


xs = [x[1] for x in X]
ys = [x[2] for x in X]
scatter(xs, ys, c = :orange, s = 10)
for i = 1:length(cx)
    plt.gcf().gca().add_artist(plt.Circle((cx[i], cy[i]), r[i], fill = false))
end

# function Dare(A, B, Q, Qf, N)
#     S = [zeros(4,4) for i = 1:N]  # AKA P
#     K = [zeros(2,4) for i = 1:N-1]
#     S[end] = Qf
#     for k = (N-1):(-1):1 
#         # A and B are the dynamics jacobians evaluated at the current state and control of the trajectory
#         # x = X[k]
#         # u = U[k]
#         # RobotDynamics.jl can get the jacobians. In the case of the linear double integrator
#         # the A and B are just constant 
#         K[k] = (R + B'*S[k+1]*B)\(B'*S[k+1]*A)
#         S[k] = Q + A'*S[k+1]*A - A'S[k+1]*B*K[k]
#     end
#     return K, S
# end


# K, S = Dare(Ad, B, Q, Qf, N)

# x_vals = zeros(N, n)
# x_vals[1, :] = x0
# u_vals = zeros(N-1, m)
# for i = 1:N-1
#     u_vals[i,:] = -K[i]*(x_vals[i,:] - X[i])
#     x_vals[i+1,:] = model.Ad*x_vals[i,:] + B*u_vals[i,:]
# end
# scatter(x_vals[:, 1], x_vals[:, 2], c = :blue, s = 10)

# xg2 = @SVector [0.2, 0.9, 0, 0]
# obj = LQRObjective(Q, R, Qf, xg2, N)
# prob = Problem(model, obj, xg2, tf, x0 = x0, constraints = conSet)

# altro = ALTROSolver(prob);
# solve!(altro);
# X = states(altro)
# U = controls(altro)

# xs = [x[1] for x in X]
# ys = [x[2] for x in X]
# scatter(xs, ys, c = :red, s = 10)
# for i = 1:length(cx)
#     plt.gcf().gca().add_artist(plt.Circle((cx[i], cy[i]), r[i], fill = false))
# end

# scatter(x0[1:1], x0[2:2], c = :red, s = 12)
# scatter(xg[1:1], xg[2:2], c = :green, s = 12)

function f(x, u)
    return Ac*x + B*u
end

@polyvar x[1:4] u[1:2]

# for xt in X 

# this should go in a loop, trying to get to each waypoint along trajectory
# xdot = f(xbar, -K*xbar)
ug = U[end]
xt = xg   # X[i]
ut = zeros(m)  # U[i]
xbar = vec(x) - xt

S = care(Ac, B, Q, R)
K = R\B'*S

# ubar = u - ut
# no need for Taylor approx here, already have a closed form linear expression
xdot = f(vec(x) + xg, ug - K*xbar)


J = xbar'*S*xbar  # candidate Lyapunov function
# Poylnomial type docs: https://github.com/JuliaAlgebra/TypedPolynomials.jl/blob/f2781765d5307c629b84b65147f9faec743778eb/src/types.jl#L76
Jdot = 2 * xbar' * S * xdot

# Vdot = diff(V,x)*xdot;
# (x'*x)*(V - rho) +  L*Vdot

# https://jump.dev/SumOfSquares.jl/latest/variables/#Nonnegative-polynomial-variables

# possibly choose another basis?? https://jump.dev/SumOfSquares.jl/latest/variables/#Choosing-a-polynomial-basis
# this is supposedly faster than normal basis?

Nm = 2  # (polynomial h must be of sufficiently large order to counteract higher order terms in Jdot)
X = monomials(x, 0:Nm)
# or SOSModel
solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => true)
model = SOSModel(solver)  # 
@variable(model, h, SOSPoly(X))  # Poly, creates p: a polynomial 
@variable(model, ρ)
@constraint(model, h >= 0)  # enforce nonnegativity constraint
@constraint(model, ρ >= 0)  # enforce nonnegativity constraint
@constraint(model, -1 *(h * Jdot + (x' * x) * (ρ - J)) >= 0)
@objective(model, Max, ρ)
optimize!(model)
@show termination_status(model)

# end


# xT = zeros(n,1);
# u0 = zeros(m,1);
# Q = eye(n);
# R = 10*eye(m);
# f0 = @(t,x,u) f(x(1:3),x(4:6),u);  # Populate this with your dynamics.
# [K0,S0,rho0] = ti_poly_lqr_roa(@(x,u) f0(0,x,u),xT,u0,Q,R);
# S0 = 1.01*S0/rho0;

# # Fixed point of the system 
# x0 = zeros()

# xbar = x;
# x = xbar+x0;
# rho = ti_poly_roa(xbar,f(x,u0-K*xbar),...
#                     xbar'*S*xbar);
    

plt.savefig("/home/alvin/research/dynamic_lqr_trees/examples/results.png")

