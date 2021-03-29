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


function gen_static_constraints(this::World, model, N)
    # NOTE: This only works with circular constraints 
    constraints = ConstraintList(model.n, model.m, N)
    for i = 1:length(this.Pset)
        xc = this.Pset[i].c[1]
        yc = this.Pset[i].c[2]
        radius = this.Pset[i].r
        xi = 1  # index into state vector to get x
        yi = 2  # index into state vector to get y
        con = CircleConstraint(model.n, [xc], [yc], [radius], xi, yi)
        
        # add constraint to all N knot points
        add_constraint!(constraints, con, 1:N)
    end
    return constraints
end