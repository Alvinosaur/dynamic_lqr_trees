using RecursiveArrayTools
using DataStructures
using StaticArrays
using LinearAlgebra
using TrajectoryOptimization
using Altro
const TO = TrajectoryOptimization
# include("world.jl")
mutable struct RRT
    start::Any
    goal::Any
    model::Any
    world::World # simulation world config
    constraints::Any  # static and dynamic constraints
    vertices::Any
    edges::Any
    Q
    Qf
    R
    N::Integer
    tf
    max_dist
    radius

    function RRT(start, goal, model, world)
        vertices = [goal]  # search backwards from goal to start
        edges = [[]]
        N = 50  # fixed number of knots 
        tf = 5.0  # fixed time 
        constraints = gen_static_constraints(world, model, N)

        # State/Dynamics constraints
        add_constraint!(
            constraints,
            BoundConstraint(model.n, model.m, x_min = world.lower_bounds, x_max = world.upper_bounds),
            1:N-1,  # do not include Nth knot, the terminal since somehow affects controls, which aren't defined at the terminal knot
        )

        Q = Diagonal(@SVector [1, 1, 1e-3, 1e-3])
        Qf = 10 * Q
        R = Diagonal(@SVector ones(model.m))
        
        max_dist = 0.2
        radius = 0.1
        new(
            start,
            goal,
            model,
            world,
            constraints,
            vertices,
            edges,
            Q,
            Qf, 
            R,
            N,
            tf,
            max_dist,
            radius
        )

    end
end

function SampleState(this::RRT)
    while (true)
        x =
            this.world.lower_bounds +
            (this.world.upper_bounds - this.world.lower_bounds) .* rand(this.model.n)
        if isValid(this.world, x)
            return x
        end
    end
end

function StackArrays(arr_of_arr)
    # https://discourse.julialang.org/t/very-best-way-to-concatenate-an-array-of-arrays/8672/2
    arr_of_arr = VectorOfArray(arr_of_arr) 
    return convert(Array, arr_of_arr)'  # N x n
end

function GenerateTrajectory(this::RRT, start, goal)
    obj = LQRObjective(this.Q, this.R, this.Qf, goal, this.N)
    prob = Problem(this.model, obj, goal, this.tf, x0 = start, constraints = this.constraints, integration=RK3)
    # initial_controls!(prob, [@SVector rand(model.m) for k = 1:N-1])
    # rollout!(prob) 

    altro = ALTROSolver(prob)
    solve!(altro);

    X = StackArrays(states(altro))
    U = StackArrays(controls(altro))
    return X, U
end


function Distance(this::RRT, traj)
    # TODO: This is a horrible way of estimating distance btwn two points
    # assumes we can solve optimal traj between two states, which defeats the whole purpose of LQR Trees since this should be a hard task done piecewise
    diff = traj[1:end-1, 1:2] - traj[2:end, 1:2]
    # https://github.com/JuliaLang/julia/issues/34830#issuecomment-589942875
    piecewise_dist = sum(map(norm, eachslice(diff, dims=1)))
    return piecewise_dist
end

function EuclideanDistance(this::RRT, x1, x2)
    return norm(x1 - x2)
end



function Nearest(this::RRT, x)
    min_dist = Inf
    nn = nothing
    ni = nothing
    for i = 1:length(this.vertices)
        # trajectory from new point to vertex since stabilize towards goal
        # traj, _ = GenerateTrajectory(this, x, this.vertices[i])
        dist = EuclideanDistance(this, x, this.vertices[i])
        if dist < min_dist
            min_dist = dist
            nn = this.vertices[i]
            ni = i
        end
    end
    return nn, ni
end

function NewVertex(this::RRT; cur, target)
    # vec = target - cur
    if !isValid(this.world, cur) || !isValid(this.world, target)
        return nothing, false, nothing, nothing
    else
        traj, U = GenerateTrajectory(this, cur, target)
        return traj[1, :], true, traj, U
        # dist = Distance(this, traj)
        # capped_dist = min(dist, this.max_dist)
        # t = this.N * capped_dist / dist 
        # # https://stackoverflow.com/questions/40520131/convert-float-to-int-in-julia-lang
        # t = min(this.N-1, max(1, trunc(Int, t)))  # convert to int
        # println("Current: ", cur)
        # println("Target: ", target)
        # println("next: ", traj[t, :])
        # println("traj: ", traj[1:t, :])
        # return traj[t, :], true, traj[1:t, :], U[1:t, :]
    end
end

function AddVertex(this::RRT, x)
    push!(this.vertices, x)
    push!(this.edges, [])
    return length(this.vertices)
end

function AddEdge(this::RRT, x1_idx, x2_idx, traj, U, dist)
    push!(this.edges[x1_idx], (dist, x2_idx, traj, U))
    push!(this.edges[x2_idx], (dist, x1_idx, traj, U))
end

function LookupTraj(this::RRT, x1_idx::Integer, x2_idx::Integer)
    for (_, ni, traj, U) in this.edges[x1_idx]
        if ni == x2_idx
            return traj
        end
    end
end

function show_trajectory(this::RRT, x1::Array{Float64}, x2::Array{Float64}, c_ = :gray, linewidth_ = 0.5)
    traj, U = GenerateTrajectory(this, x1, x2)
    @views plot(traj[:, 1], traj[:, 2], c = c_, linewidth = linewidth_)
end

function show_trajectory(this::RRT, x1_idx::Integer, x2_idx::Integer, c_ = :gray, linewidth_ = 0.5)
    traj = LookupTraj(this, x1_idx, x2_idx)
    @views plot(traj[:, 1], traj[:, 2], c = c_, linewidth = linewidth_)
end

function show_trajectory(this::RRT, traj::Adjoint{Float64,Array{Float64,2}}, c_ = :gray, linewidth_ = 0.5)
    @views plot(traj[:, 1], traj[:, 2], c = c_, linewidth = linewidth_)
end

function Search(this::RRT, max_iters, visualize)
    iter = 0
    done = false
    while (!done)
        iter += 1
        x = SampleState(this)
        nearest, nearest_idx = Nearest(this, x)
        # TODO: WARNING THIS SHOULD BE cur=x, target=nearest for LQR trajectories since searching backwards towards goal
        # for LQR Trees, there is no notion of stepping along path 
        # so we can just solve optimal traj from cur=x to target=nearest 
        # but for holonomic test version, need to branch from existing
        # to new node and stop when hit obstacle.
        new_vert, is_expanded, traj, U = NewVertex(this, cur = x, target = nearest)
        if !is_expanded
            continue
        end
        new_vert_idx = AddVertex(this, new_vert)
        cost = Distance(this, traj)
        AddEdge(this, nearest_idx, new_vert_idx, traj, U, cost)

        if visualize
            close()
            show(this)
            savefig("fig/" * string(iter) * ".png")
        end

        # Search backwards from goal to start
        # TODO: WARNING THIS SHOULD BE cur=start, target=new_vert for LQRTree
        traj_from_start, U_from_start = GenerateTrajectory(this, start, new_vert)
        start_dist = Distance(this, traj_from_start)
        if start_dist < 2 * this.radius
            start_idx = AddVertex(this, start)
            AddEdge(this, new_vert_idx, start_idx, traj_from_start, U_from_start, start_dist)
            done = true
            break
        end

        if iter >= max_iters
            break
        end
    end
    return done, BuildPath(this)
end

function BuildPath(this::RRT)
    ϵ = 1
    goal_idx = 1
    start_idx = length(this.vertices)
    open = [(0.0, goal_idx)]  # cost, idx = goal_idx
    open = BinaryHeap(Base.By(first), open)
    closed = []
    successor = Dict{Int64,Int64}()
    G = Dict{Int64,Float16}()
    G[goal_idx] = 0
    while length(open) > 0
        cur_g, cur_idx = pop!(open)
        if cur_idx == start_idx
            break
        end
        if cur_idx in closed
            continue
        end

        push!(closed, cur_idx)

        neighbors = this.edges[cur_idx]
        for (trans_cost, next_idx, _, _) in neighbors
            new_g = cur_g + trans_cost

            if get(G, next_idx, nothing) == nothing || new_g < G[next_idx]
                G[next_idx] = new_g
                traj, _ = GenerateTrajectory(this, start, this.vertices[next_idx])
                h = Distance(this, traj)
                f = new_g + ϵ * h
                push!(open, (f, next_idx))
                successor[next_idx] = cur_idx
            end
        end
    end

    path = [start_idx]
    cur_idx = start_idx
    while cur_idx != goal_idx
        cur_idx = successor[cur_idx]
        push!(path, cur_idx)
    end
    return path
end


function show(this::RRT)
    println("drawing...")
    show(this.world)
    N = length(this.vertices)
    mat = zeros(2, N)
    for idx = 1:N
        mat[:, idx] = this.vertices[idx][1:2]
    end
    # idxset_open = findall(this.bool_open)
    # idxset_closed = findall(this.bool_closed)
    # idxset_unvisit = findall(this.bool_unvisit)  
    # idxset_tree = setdiff(union(idxset_open, idxset_closed), [1])
    scatter(mat[1, 1], mat[2, 1], c = :blue, s = 10, zorder = 100)
    scatter(mat[1, end], mat[2, end], c = :blue, s = 10, zorder = 101)
    println(mat[:, end])
    scatter(mat[1, 2:end], mat[2, 2:end], c = :gray, s = 2)
    scatter(mat[1, 2:end], mat[2, 2:end], c = :gray, s = 2)
    #scatter(mat[1, idxset_unvisit], mat[2, idxset_unvisit], c=:orange, s=5)
    for idx = 1:N
        neighbors = this.edges[idx]
        for (_, next_idx, _, _) in neighbors
            show_trajectory(this, idx, next_idx)
        end
    end

    scatter(mat[1, 1], mat[2, 1], c = :blue, s = 20, zorder = 100)
    scatter(mat[1, end], mat[2, end], c = :blue, s = 20, zorder = 101)

    xlim(this.world.lower_bounds[1] - 0.05, this.world.upper_bounds[1] + 0.05)
    ylim(this.world.lower_bounds[2] - 0.05, this.world.upper_bounds[2] + 0.05)
    println("finish drawing")
end
