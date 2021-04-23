using ControlSystems
using StaticArrays
using LinearAlgebra
const TO = TrajectoryOptimization

function verify_traj(wpts::Vec4f[], world)

    
    A = [1 0; 0 1]
    B = [0;1]
    S = dare(A, B, Q, R)
    K = (B'*S*B + R)\(B'S*A)

end

function V(x, S)
    return x'Sx
end
