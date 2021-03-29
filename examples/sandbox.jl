using Altro
using TrajectoryOptimization
using StaticArrays, LinearAlgebra
using RobotDynamics
using PyPlot

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
    A::Any
    B::Any
    n::Integer  # state space size
    m::Integer  # ctrl space size

    function DoubleIntegrator()
        A = [
            zeros(2, 2) I(2)
            zeros(2, 2) zeros(2, 2)
        ]
        B = [
            zeros(2, 2)
            I(2)
        ]
        n = 4
        m = 2
        new(A, B, n, m)
    end
end


function RobotDynamics.dynamics(model::DoubleIntegrator, x, u)
    model.A * x + model.B * u
end

Base.size(::DoubleIntegrator) = 4, 2
RobotDynamics.state_dim(model::DoubleIntegrator) = 4
RobotDynamics.control_dim(model::DoubleIntegrator) = 2

lower_bounds = [0.0, 0.0, -0.1, -0.1]
upper_bounds = [1.0, 1.0, 0.1, 0.1]
start = @SVector [0.1, 0.1, 0.0, 0.0]
goal = @SVector [0.9, 0.9, 0.0, 0.0]

model = DoubleIntegrator()
n, m = size(model)

N = 50
tf = 20.0
dt = 0.2

Q = 1.0 * Diagonal(@SVector ones(n))
Qf = 10.0 * Diagonal(@SVector ones(n))
R = 1.0 * Diagonal(@SVector ones(m))
obj = LQRObjective(Q, R, Qf, goal, N)

r = [0.2, 0.1]
x = [0.5, 0.6]
y = [0.6, 0.4]
xi = 1
yi = 2
obstacles = TrajectoryOptimization.CircleConstraint(model.n, x, y, r, xi, yi)

conSet = ConstraintList(n, m, N)
# goal constraint not really necessary since LQR objective already minimizes distance with goal
# add_constraint!(conSet, GoalConstraint(goal), N:N)

add_constraint!(conSet, obstacles, 1:N)

add_constraint!(
    conSet,
    BoundConstraint(n, m, x_min = lower_bounds, x_max = upper_bounds),
    1:N-1,  # do not include Nth knot, the terminal since somehow affects controls, which aren't defined at the terminal knot
)

prob = Problem(model, obj, goal, tf, x0 = start, constraints = conSet)

altro = ALTROSolver(prob);
solve!(altro);
X = states(altro)
U = controls(altro)

plt.clf()
xs = [x[1] for x in X]
ys = [x[2] for x in X]
scatter(xs, ys, c = :blue, s = 10)
for i = 1:length(x)
    plt.gcf().gca().add_artist(plt.Circle((x[i], y[i]), r[i], fill = false))
end
plt.savefig("/home/alvin/research/dynamic_lqr_trees/examples/results.png")

