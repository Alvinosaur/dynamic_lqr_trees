using StaticArrays
using LinearAlgebra
using TrajectoryOptimization
using RobotDynamics
const TO = TrajectoryOptimization



struct DoubleIntegrator{} <: TO.AbstractModel
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