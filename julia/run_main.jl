using LinearAlgebra
using StaticArrays
using PyPlot
const Vec2f = SVector{2,Float64}
const Vec4f = SVector{4,Float64}

include("src/geometry/geometry.jl")
include("src/world.jl")
include("src/rrt.jl")
include("src/dynamics/double_integrator.jl")


obstacle_set = [
    Sphere([0.2, 0.3], 0.1),
    Sphere([0.5, 0.5], 0.1),
    Sphere([0.8, 0.3], 0.1),
    Sphere([0.8, 0.6], 0.1),
    Sphere([0.2, 0.7], 0.1),
    Sphere([0.2, 0.7], 0.1),
]

# define bounds on state space
state_str = ["x", "y", "vx", "vy"]
lower_bounds = @SVector [0.0, 0.0, -1, -1]
upper_bounds = @SVector [1.0, 1.0, 1, 1]

world = World(lower_bounds, upper_bounds, obstacle_set)


model = DoubleIntegrator()
start = @SVector [0.1, 0.1, 0.0, 0.0]
goal = @SVector [0.9, 0.9, 0.0, 0.0]
max_iters = 6000
visualize = true

rrt = RRT(start, goal, model, world)
solved, path = Search(rrt, max_iters, visualize)

if solved
    show(rrt)
    for i=2:length(path)
        show_trajectory(rrt, path[i-1], path[i], :blue, 1.5)
    end
else
    print("Failed to solve!")
end
savefig("./fig/final.png")



