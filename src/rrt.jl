using DataStructures
using StaticArrays
using LinearAlgebra
using TrajectoryOptimization
const TO = TrajectoryOptimization

mutable struct RRT
    start
    goal
    S::Int8  # size of state space
    world::World # simulation world config
    vertices
    edges
    max_dist
    radius

    function RRT(start, goal, world; max_dist, radius)
        vertices = [goal]  # search backwards from goal to start
        edges = [[]]
        new(start, goal, size(start, 1), world, vertices, edges, max_dist, radius)
    end
end

function SampleState(this::RRT)
    while (true)
        x = this.world.lower_bounds + (
            this.world.upper_bounds - this.world.lower_bounds).*rand(this.S)
        if isValid(this.world, x)
            return x
        end
    end
end

function Distance(this::RRT, x1, x2)
    return norm(x1 - x2)
end

function Nearest(this::RRT, x)
    min_dist = Inf
    nn = nothing
    ni = nothing
    for i = 1:length(this.vertices)
        dist = Distance(this, this.vertices[i], x)
        if dist < min_dist
            min_dist = dist
            nn = this.vertices[i]
            ni = i
        end
    end
    return nn, ni
end

function NewVertex(this::RRT; cur, target)
    vec = target - cur 
    dist = Distance(this, cur, target)
    vec = vec / dist
    dist = min(dist, this.max_dist)
    N = 10
    new_vert = nothing
    is_expanded = false
    for t = 1:N
        x = cur + vec * dist * (t / N)
        if isValid(this.world, x)
            new_vert = x
            is_expanded = true
        else
            break
        end
    end
    return new_vert, is_expanded
end

function AddVertex(this::RRT, vert)
    push!(this.vertices, vert)
    push!(this.edges, [])
    return length(this.vertices)
end

function AddEdge(this::RRT, vert1_idx, vert2_idx, dist)
    push!(this.edges[vert1_idx], (dist, vert2_idx))
    push!(this.edges[vert2_idx], (dist, vert1_idx))
end

function gen_trajectory(this::RRT, x1, x2, N)
    vec = x2 - x1 
    dist = Distance(this, x2, x1)
    vec = vec / dist
    wpts = zeros(2, N+1)
    for t = 0:N
        x = x1 + vec * dist * (t / N)
        wpts[:, t+1] = x
    end
    return wpts
end

function show_trajectory(this::RRT, x1, x2, N_split = 20, c_ = :gray, linewidth_ = 0.5)
    wpts =  gen_trajectory(this, x1, x2, N_split)
    @views plot(wpts[1, :], wpts[2, :], c=c_, linewidth = linewidth_)
end

function Search(this::RRT, max_iters, visualize)
    iter = 0
    done = false
    while(!done)
        iter += 1
        x = SampleState(this)
        nearest, nearest_idx = Nearest(this, x)
        # TODO: WARNING THIS SHOULD BE cur=x, target=nearest for LQR trajectories since searching backwards towards goal
        # for LQR Trees, there is no notion of stepping along path 
        # so we can just solve optimal traj from cur=x to target=nearest 
        # but for holonomic test version, need to branch from existing
        # to new node and stop when hit obstacle.
        new_vert, is_expanded = NewVertex(this, cur=nearest, target=x)
        if !is_expanded
            continue
        end
        new_vert_idx = AddVertex(this, new_vert)
        cost = Distance(this, nearest, new_vert)
        AddEdge(this, nearest_idx, new_vert_idx, cost)
        
        # Search backwards from goal to start
        start_dist = Distance(this, new_vert, start)
        if start_dist < 2 * this.radius
            start_idx = AddVertex(this, start)
            AddEdge(this, new_vert_idx, start_idx, start_dist)
            done = true
            break
        end

        if iter >= max_iters
            break
        end

        if visualize
            close()
            show(this)
            savefig("fig/"*string(iter)*".png")
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
        for (trans_cost, next_idx) in neighbors
            new_g = cur_g + trans_cost
            
            if get(G, next_idx, nothing) == nothing || new_g < G[next_idx]
                G[next_idx] = new_g
                h = Distance(this, this.vertices[next_idx], start)
                f = new_g + ϵ*h
                push!(open, (f, next_idx))
                successor[next_idx] = cur_idx
            end
        end
    end
    
    path = []
    cur_idx = start_idx
    # path doesn't include start, ends with goal
    while cur_idx != goal_idx
        cur_idx = successor[cur_idx]
        push!(path, this.vertices[cur_idx])
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
    scatter(mat[1, 1], mat[2, 1], c=:blue, s=10, zorder = 100)
    scatter(mat[1, end], mat[2, end], c=:blue, s=10, zorder = 101)
    scatter(mat[1, 2:end], mat[2, 2:end], c=:gray, s=2)
    scatter(mat[1, 2:end], mat[2, 2:end], c=:gray, s=2)
    #scatter(mat[1, idxset_unvisit], mat[2, idxset_unvisit], c=:orange, s=5)
    for idx = 1:N
        neighbors = this.edges[idx]
        for (_, next_idx) in neighbors 
            show_trajectory(this, this.vertices[idx], this.vertices[next_idx])
        end
    end

    scatter(mat[1, 1], mat[2, 1], c=:blue, s=20, zorder = 100)
    scatter(mat[1, end], mat[2, end], c=:blue, s=20, zorder = 101)

    xlim(this.world.lower_bounds[1]-0.05, this.world.upper_bounds[1]+0.05)
    ylim(this.world.lower_bounds[2]-0.05, this.world.upper_bounds[2]+0.05)
    println("finish drawing")
end