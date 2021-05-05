using StaticArrays
using LinearAlgebra
using TrajectoryOptimization
using RobotDynamics
const TO = TrajectoryOptimization

struct QuadrotorFlat{} <: TO.AbstractModel
    A::Any
    B::Any
    n::Integer  # state space size
    m::Integer  # ctrl space size

    function QuadrotorFlat()
        A = [
            zeros(3, 3) I(3) zeros(3, 1)
            zeros(4, 7)
        ]
        B = [
            zeros(3, 4)
            I(4)
        ]
        n = 7
        m = 4
        new(A, B, n, m)
    end
end

function RobotDynamics.dynamics(model::QuadrotorFlat, x, u)
    model.A * x + model.B * u
end

Base.size(::QuadrotorFlat) = 7, 4
RobotDynamics.state_dim(model::QuadrotorFlat) = 7
RobotDynamics.control_dim(model::QuadrotorFlat) = 4