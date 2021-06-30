# Status messages
REACHED::Int = 0   # reached target node
TRAPPED::Int = 1   # unable to extend at all to new point
ADVANCED::Int = 2  # partially extend to new point

start = [0, 4.5]
goal = [10.3, 6.7]

XMAX = 20
XMIN = -20
DX = 1.0
YMIN = -15
YMAX = 15
DY = 1.0


function NearestNeighbor(p, nodes)
    return argmin(norm(p - nodes))
end


function IsValid(p, map)
    xi = p[1] ÷ DX
    yi = p[2] ÷ DY
    in_bounds = (XMIN < xi <= XMAX) && (YMIN < yi <= YMAX)
    return in_bounds && map[xi+1, yi+1] == 1
end


function Extend(target, nodes, map, max_dist, step, is_forward)
    nn = NearestNeighbor(p=target, nodes=nodes)
    if is_forward
        status, new_p = ExtendHelper(p1=nn, p2=target, map=map, max_dist=max_dist, step=step)
    else
        status, new_p = ExtendHelper(p1=target, p2=nn, map=map, max_dist=max_dist, step=step)
    end
    if status != TRAPPED
        nodes = [nodes; new_p]
    end
    return status, new_p
end


function ExtendHelper(p1, p2, map, max_dist, step)
    """
    Attempts to connect p1 to p2 (direction may matter for dynamics). Have optional max distance
    of an edge.
    Returns REACHED if able to fully connect.
    Returns ADVANCED if partially connects.
    Returns TRAPPED if unable to even partially connect.
    """
    dist = norm(p2 - p1)
    vec12 = (p2 - p1) / dist
    if dist > max_dist
        dist = max_dist
        partial_extend = true
    else
        partial_extend = false
    end

    N = dist ÷ step
    if N < 2
        new_p = copy(p2)
        return REACHED, new_p
    end

    was_extended = false
    new_p = copy(p1)
    for i = 1:N
        new_p += step * vec12
        if !IsValid(new_p, map)
            if was_extended
                return ADVANCED, new_p
            else
                return TRAPPED, new_p
            end
        end
        was_extended = true
    end

    if partial_extend
        return ADVANCED, new_p
    else
        new_p = copy(p2)
        return ADVANCED, new_p
    end
end

function RRTCHelper(target, tree1, tree2, map, max_dist, step, is_forward)
    status, new_p = Extend(target=target, nodes=tree1,
        map=map, max_dist=max_dist, step=step, is_forward=is_forward)
    
    if status != TRAPPED
        status2 = Connect(target=new_p, nodes=tree2,
        map=map, max_dist=max_dist, step=step, is_forward=!is_forward)
        return status2
    else
        return TRAPPED
    end
end

function RRTC()
    max_iters = 5000
    step = 0.2
    map = zeros(YMAX - YMIN - 1, XMAX - XMIN - 1)
    
end

"""
Thoughts:
- in Extend() function, need parameter stating whether extending from
start or goal, since this affects direction of extension:
    from start: p2 = newly sampled node
    from goal: p2 = nearest neighbor in tree

- for simple ND space, use KDTree for fast NN search, but
for dynamics, how to define nearest neighbor?
"""
