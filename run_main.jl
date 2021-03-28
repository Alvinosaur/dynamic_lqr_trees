using LinearAlgebra
using StaticArrays
using PyPlot
const Vec2f = SVector{2, Float64}
const Vec4f = SVector{4, Float64}

include("src/geometry/geometry.jl")
include("src/world.jl")
include("src/rrt.jl")


obstacle_set = [Rectangle([0.2, 0.3], 0.2, 0.2),
                Rectangle([0.5, 0.5], 0.2, 0.3),
                Rectangle([0.8, 0.3], 0.2, 0.1), 
                Rectangle([0.8, 0.6], 0.15, 0.2),
                Rectangle([0.2, 0.7], 0.1, 0.4),
                Rectangle([0.2, 0.7], 0.1, 0.4)]

# define bounds on state space
state_str = ["x", "y"]
lower_bounds = @SVector [0.0, 0.0]
upper_bounds = @SVector [1.0, 1.0]

world = World(lower_bounds, upper_bounds, obstacle_set)

# state_str = ["x", "y", "vx", "vy"]
# lower_bounds = @SVector [0.0, 0.0, -0.5, -0.5]
# upper_bounds = @SVector [1.0, 1.0, 0.5, 0.5]
# s_init = Vec4f([0.1, 0.1, 0.0, 0.0])   # x, y, dx, dy
# s_goal = Vec4f([0.9, 0.9, 0.0, 0.0])
start = @SVector [0.1, 0.1]
goal = @SVector [0.9, 0.9]
max_dist = 0.2
radius = 0.1
max_iters = 6000
visualize = true

rrt = RRT(start, goal, world, max_dist=max_dist, radius=radius)
solved, path = Search(rrt, max_iters, visualize)

if solved
    show(rrt)
    local prev = start
    for next in path
        show_trajectory(rrt, prev, next, 20, :blue, 1.5)
        prev = next
    end
    show_trajectory(rrt, prev, goal, 20, :blue, 1.5)
else
    print("Failed to solve!")
end
savefig("./fig/final.png")



