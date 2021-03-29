using TrajectoryOptimization

mutable struct World
    lower_bounds::Any
    upper_bounds::Any
    state_size::Int
    Pset::Vector{Union{Polygonic,GeoParametric}}  # Obstacles
    function World(lower_bounds, upper_bounds, Pset)
        new(lower_bounds, upper_bounds, size(upper_bounds, 1), Pset)
    end
end

@inline function isValid(this::World, s_q)
    # check if the sampled point is inside the world"
    all(this.lower_bounds .< s_q) || return false
    all(s_q .< this.upper_bounds) || return false

    # check if point is not already sampled (within some radius of another point)
    for P in this.Pset
        isInside(P, Vec2f(s_q[1:2])) && return false
    end
    return true
end

# @inline function isValid(this::World, q_set)
#     # check validity for multiple points.
#     # will be used for piecewize path consited of multiple points
#     for q in q_set
#         !isValid(this, q) && return false
#     end
#     return true
# end

@inline function isIntersect(this::World, q1, q2)
    for P in this.Pset
        isIntersect(P, q1, q2) && return true
    end
    return false
end


function show(this::World)
    p1 = [this.lower_bounds[1], this.lower_bounds[2]]
    p2 = [this.lower_bounds[1], this.upper_bounds[2]]
    p3 = [this.upper_bounds[1], this.upper_bounds[2]]
    p4 = [this.upper_bounds[1], this.lower_bounds[2]]
    plot([p1[1], p2[1]], [p1[2], p2[2]], "k-")
    plot([p2[1], p3[1]], [p2[2], p3[2]], "k-")
    plot([p3[1], p4[1]], [p3[2], p4[2]], "k-")
    plot([p4[1], p1[1]], [p4[2], p1[2]], "k-")
    for P in this.Pset
        show(P)
    end
end

# custom function signature for TrajectoryOptimization package
function combine_constraints(cons1::ConstraintList, cons2::ConstraintList)
    @assert check_dims(cons2, cons1.n, cons1.m) "2nd constraints list not consistent with 1st list n=$(cons1.n) and m=$(cons1.m)"

    N1 = length(cons1.p)  # holds true num of constraints, even if not added yet
    N2 = length(cons2.p)
    new_cons = constraints = ConstraintList(cons1.n, cons1.m, N1 + N2)
    for i = 1:N1
        add_constraint!(new_cons, cons1.constraints[i], i)
    end

    for j = 1:N2
        add_constraint!(new_cons, cons2.constraints[j], j + N1)
    end
    return new_cons
end


function gen_static_constraints(this::World, model)
    # NOTE: This only works with circular constraints 
    constraints = ConstraintList(model.n, model.m, length(world.Pset))
    for i = 1:length(this.Pset)
        xc = SA[this.Pset[i].c[1]]
        yc = SA[this.Pset[i].c[2]]
        radius = SA[this.Pset[i].r]
        xi = 1  # index into state vector to get x
        yi = 2  # index into state vector to get y
        con = CircleConstraint(model.n, xc, yc, radius, xi, yi)
        add_constraint!(constraints, con, i)
    end
    return constraints
end