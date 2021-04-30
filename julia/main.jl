#!/usr/bin/env julia
using PyCall
@pyimport scipy

using RobotOS
@rosimport mavros_msgs.msg: AttitudeTarget
@rosimport geometry_msgs.msg: PoseStamped, TwistStamped, Vector3
@rosimport nav_msgs.msg: Path
@rosimport mavros_msgs.srv: CommandBool
rostypegen()
# using .mavros_msgs.msg
# using .geometry_msgs.msg
# using .nav_msgs.msg
# using .mavros_msgs.srv

using LinearAlgebra
using StaticArrays
using Printf
using Revise
using Debugger

include("geometry/geometry.jl")
include("rrt_src/world.jl")
include("rrt_src/rrt.jl")
include("dynamics/quadrotor_flat.jl")

mutable struct Controller
    vel_msg
    pose_msg
    num_obs
    obstacle_paths
    obs_radii
    pos_control_pub 
    att_control_pub
    arm_service

    function Controller(num_obs, obs_radii, pos_control_pub, att_control_pub, arm_service)
        obstacle_paths = [nothing for i = 1:num_obs]
        new(nothing, [nothing,], num_obs, obstacle_paths, obs_radii, pos_control_pub, att_control_pub, arm_service)
         
    end
end


function vel_callback(msg::geometry_msgs.msg.TwistStamped, controller::Controller)
    controller.vel_msg = msg
end

function pose_callback(msg::geometry_msgs.msg.PoseStamped, controller::Controller)
    controller.pose_msg[1] = msg
end

function obstacle_callback(msg::nav_msgs.msg.Path, controller::Controller, i)
    controller.obstacle_paths[i] = "hello"
    println(i, controller.obstacle_paths[i] == nothing)
    
    println(i, controller.obstacle_paths[i] == nothing)
end

function create_pose_msg(x)
    target_raw_pose = PoseStamped()
    target_raw_pose.header.stamp = RobotOS.now()
    yaw = x[end]
    q = scipy.spatial.transform.Rotation.from_euler("zyx", [yaw, 0, 0], degrees=false).as_quat()

    # target_raw_pose.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
    target_raw_pose.pose.position.x = x
    target_raw_pose.pose.position.y = y
    target_raw_pose.pose.position.z = z

    target_raw_pose.pose.orientation.x = q[1]
    target_raw_pose.pose.orientation.y = q[2]
    target_raw_pose.pose.orientation.z = q[3]
    target_raw_pose.pose.orientation.w = q[4]

    return target_raw_pose
end

function check_connection(controller; max_iters=10)
    for i = 1:max_iters
        if ! ((controller.pose_msg[1] == nothing) || 
            (controller.vel_msg == nothing) || 
            (controller.obstacle_paths[1] == nothing))
            break
        else
            println("Waiting for initialization.")
            println(controller.pose_msg[1] == nothing)
            println(controller.vel_msg == nothing)
            println(controller.obstacle_paths[1] == nothing)
            rossleep(0.5)
        end
    end

    return ! ((controller.pose_msg[1] == nothing) || 
                (controller.vel_msg == nothing) || 
                (controller.obstacle_paths[1] == nothing))
end

function takeoff(controller, target_pose; max_iters=100)
    reached_target = false
    is_armed = false
    for i = 1:max_iters
        println("Taking off. Current height: ", controller.pose_msg.pose.position.z)
        publish(controller.pose_pub, create_pose_msg(target_pose))
        is_armed = controller.arm_srv(true)
        reached_target = abs(controller.pose_msg.pose.position.z - 
                            self.takeoff_height) < 1e-4
        if reached_target && is_armed
            break
        end
        rossleep(0.2)
    end
    return reached_target && is_armed
end


function run_search(start, goal, new_obstacles)
    obstacles = []
    for (center, radius) in new_obstacles
        push!(obstacles, Sphere(center, radius))
    end
    world = World(lower_bounds, upper_bounds, obstacles)

    max_iters = 6000
    visualize = true

    rrt = RRT(start, goal, model, world, N=200, tf=10.0)
    solved, path, ctrl_traj = Search(rrt, max_iters, visualize)
    return solved, path, ctrl_traj
end


function loop(controller)
    if !check_connection(controller)
        println("Failed to connect!")
        return
    end

    loop_rate = Rate(10)  # 10hz -> dt = 0.1sec
    x0 = [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0]
    xg = [10.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0]

    # run rrt search (temporary assume all obstacles static)
    # define bounds on state space
    state_str = ["x", "y", "z", "vx", "vy", "vz", "ψ"]
    lower_bounds = @SVector [-1, -1, 0, -2, -2, -2, -pi]
    upper_bounds = @SVector [7, 7, 5, 2, 2, 2, pi]
    model = QuadrotorFlat()
    new_osbtacles = [
        ((controller.obstacle_paths[i].pose.position.x,
        controller.obstacle_paths[i].pose.position.y,
        controller.obstacle_paths[i].pose.position.z), controller.obs_radii[i])  for i = 1:controller.num_obs]

    # TODO: don't just execute U, rather keep as piecewise and run LQR for each piecewise to get the linear feedback control
    solved, X, U = run_search(x0, goal, new_obstacles)

    if !solved
        println("Failed to solve trajectory!")
        return 
    end

    if !takeoff(controller, x0)
        println("Failed to takeoff!")
        return
    end

    i = 1
    while !is_shutdown() && i <= size(X,1)
        x = X[i, :]
        publish(controller.pose_pub, create_pose_msg(x))
        rossleep(loop_rate)
        i += 1
    end
end

function main()
    num_obs = 5
    obs_radii = [1.0, 0.6, 0.3, 0.5, 0.4]  # hardcoded, defined in main.world
    init_node("controller")

    # Publishers
    pos_control_pub = Publisher{geometry_msgs.msg.PoseStamped}("mavros/setpoint_position/local", queue_size=10)
    att_control_pub = Publisher{mavros_msgs.msg.AttitudeTarget}("mavros/setpoint_raw/attitude", queue_size=10)

    # Services
    arm_service = ServiceProxy{mavros_msgs.srv.CommandBool}("/mavros/cmd/arming")

    # High level flight controller 
    controller = Controller(num_obs, obs_radii, pos_control_pub, att_control_pub, arm_service)

    # Subscribers
    local_vel_sub = Subscriber{geometry_msgs.msg.TwistStamped}("/mavros/local_position/velocity_local", 
        vel_callback, (controller,), queue_size=10)
    local_pose_sub = Subscriber{geometry_msgs.msg.PoseStamped}("/mavros/local_position/pose", 
        pose_callback, (controller,), queue_size=10)
    temp = Subscriber{nav_msgs.msg.Path}("/dynamic_lqr_trees/obs_path_0", 
    obstacle_callback, (controller,0+1), queue_size=10)
    obstacle_path_subs = [
        Subscriber{nav_msgs.msg.Path}("/dynamic_lqr_trees/obs_path_$(@sprintf("%d", i))", 
        obstacle_callback, (controller,i+1), queue_size=10) for i = 0:num_obs-1]

    

    loop(controller)
end

# if ! isinteractive()
#     main()
# end

main()