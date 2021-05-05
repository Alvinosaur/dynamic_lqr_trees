#!/usr/bin/env julia
using PyCall
@pyimport math
@pyimport numpy

using RobotOS
@rosimport mavros_msgs.msg: AttitudeTarget, Thrust
@rosimport geometry_msgs.msg: PoseStamped, TwistStamped, Vector3
@rosimport nav_msgs.msg: Path
@rosimport mavros_msgs.srv: CommandBool, SetMode
rostypegen()
# using .mavros_msgs.msg
# using .geometry_msgs.msg
# using .nav_msgs.msg
# using .mavros_msgs.srv

using SciPy
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
    N
    dt
    vel_msg
    pose_msg
    num_obs
    obstacle_paths
    obs_radii
    pose_control_pub 
    att_control_pub
    thrust_control_pub
    arm_service
    offboard_service


    function Controller(N, dt, num_obs, obs_radii, pose_control_pub, att_control_pub, thrust_control_pub, arm_service, offboard_service)
        obstacle_paths = Array{Union{Nothing, nav_msgs.msg.Path}}(nothing, num_obs)
        new(N, dt, nothing, nothing, num_obs, obstacle_paths, obs_radii, pose_control_pub, att_control_pub, thrust_control_pub, arm_service, offboard_service)
         
    end
end

function vel_callback(msg::geometry_msgs.msg.TwistStamped, controller::Controller)
    controller.vel_msg = msg
end

function pose_callback(msg::geometry_msgs.msg.PoseStamped, controller::Controller)
    controller.pose_msg = msg
end

function obstacle_callback(msg::nav_msgs.msg.Path, controller::Controller, i)
    controller.obstacle_paths[i] = msg
end

function create_pose_msg(x)
    target_raw_pose = geometry_msgs.msg.PoseStamped()
    target_raw_pose.header.stamp = RobotOS.now()
    yaw = x[end]
    q = SciPy.spatial.transform.Rotation.from_euler("zyx", [yaw, 0, 0], degrees=false).as_quat()

    # target_raw_pose.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
    target_raw_pose.pose.position.x = x[1]
    target_raw_pose.pose.position.y = x[2]
    target_raw_pose.pose.position.z = x[3]

    target_raw_pose.pose.orientation.x = q[1]
    target_raw_pose.pose.orientation.y = q[2]
    target_raw_pose.pose.orientation.z = q[3]
    target_raw_pose.pose.orientation.w = q[4]

    return target_raw_pose
end

function create_path(traj, dt; frame="world")
    poses = []
    cur_tstamp = RobotOS.now()
    for i in range(traj.shape[0])
        tstamp = cur_tstamp + RobotOS.Duration(dt * i)
        p = geometry_msgs.msg.PoseStamped()

        p.pose.position = Point(traj[i, 1], traj[i, 2], traj[i, 3])

        p.header.frame_id = frame
        p.header.stamp = tstamp
        poses.append(p)
    end

    path = nav_msgs.msg.Path()
    path.header.frame_id = frame
    path.header.stamp = cur_tstamp
    path.poses = poses
    return path

end

function check_connection(controller; max_iters=10)
    for i = 1:max_iters
        if ! ((controller.pose_msg == nothing) || 
            (controller.vel_msg == nothing) || 
            (controller.obstacle_paths[1] == nothing))
            break
        else
            println("Waiting for initialization.")
            println(controller.pose_msg == nothing)
            println(controller.vel_msg == nothing)
            println(controller.obstacle_paths[1] == nothing)
            sleep(0.5)
        end
    end

    return ! ((controller.pose_msg == nothing) || 
                (controller.vel_msg == nothing) || 
                (controller.obstacle_paths[1] == nothing))
end

function takeoff(controller, target_pose; max_iters=100)
    reached_target = false
    is_armed = false
    takeoff_height = target_pose[3]
    for i = 1:max_iters
        println("Taking off. Current height: ", controller.pose_msg.pose.position.z)
        publish(controller.pose_control_pub, create_pose_msg(target_pose))
        is_armed = controller.arm_service(mavros_msgs.srv.CommandBoolRequest(true))
        is_offboard = controller.offboard_service(mavros_msgs.srv.SetModeRequest(220, "OFFBOARD"))

        reached_target = abs(controller.pose_msg.pose.position.z - 
                            takeoff_height) < 1e-1
        if reached_target
            break
        end
        sleep(0.2)
    end
    return reached_target
end

function run_search(start, goal, new_obstacles, N, dt)
    obstacles = []
    for (center, radius) in new_obstacles
        push!(obstacles, Sphere(center, radius))
    end
    world = World(lower_bounds, upper_bounds, obstacles)

    max_iters = 6000
    visualize = true

    tf = N * dt
    rrt = RRT(start, goal, model, world, N=N, tf=tf)
    solved, path, ctrl_traj = Search(rrt, max_iters, visualize)
    return solved, path, ctrl_traj
end


function loop(controller, path_pub)
    if !check_connection(controller)
        println("Failed to connect!")
        return
    end
    println("Connected!")

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
        ((controller.obstacle_paths[i].poses[1].pose.position.x,
        controller.obstacle_paths[i].poses[1].pose.position.y,
        controller.obstacle_paths[i].poses[1].pose.position.z), controller.obs_radii[i])  for i = 1:controller.num_obs]

    # TODO: don't just execute U, rather keep as piecewise and run LQR for each piecewise to get the linear feedback control
    # println("Running search...")
    # solved, rrt = run_search(x0, goal, new_obstacles, controller.N, controller.dt)
    # _, X, U = BuildPath(rrt)
    # println("Finished!")

    # if !solved
    #     println("Failed to solve trajectory!")
    #     return 
    # end
    X = numpy.load("state_traj.npy")
    U = numpy.load("ctrl_traj.npy")

    X_viz_path = create_path(X, controller.dt; frame="world")

    if !takeoff(controller, x0)
        println("Failed to takeoff!")
        return
    end

    i = 1
    while !is_shutdown() && i <= size(U, 1)
        # visualize generated path
        publish(path_pub, X_viz_path)

        u_traj = U[i, :, :]
        println(size(u_traj, 1))
        x_traj = X[i, :, :]
        for j = 1:size(u_traj, 1)
            cur_q = [controller.pose_msg.pose.orientation.x,
            controller.pose_msg.pose.orientation.y,
            controller.pose_msg.pose.orientation.z,
            controller.pose_msg.pose.orientation.w]

            # thrust, phid, thetad, psid = inverse_dyn(cur_q, x_traj[i, :], u_traj[i, :])
            # publish(controller.thrust_control_pub, create_pose_msg(x))
            publish(controller.pose_control_pub, create_pose_msg(x_traj[j, :]))
            rossleep(loop_rate)
        end
        i += 1
    end
end

function main()
    N, dt = 10, 0.1
    num_obs = 4
    obs_radii = [0.3, 0.6, 0.7, 0.5]  # hardcoded, defined in main.world
    init_node("controller")

    # Publishers
    pose_control_pub = Publisher{geometry_msgs.msg.PoseStamped}("mavros/setpoint_position/local", queue_size=10)
    # att_control_pub = Publisher{mavros_msgs.msg.AttitudeTarget}("mavros/setpoint_attitude/attitude", queue_size=10)
    att_control_pub = Publisher{mavros_msgs.msg.PoseStamped}("mavros/setpoint_attitude/attitude", queue_size=10)
    thrust_control_pub = Publisher{mavros_msgs.msg.Thrust}("mavros/setpoint_attitude/thrust", queue_size=10)
    path_pub = Publisher{nav_msgs.msg.Path}("/dynamic_lqr_trees/drone_traj", queue_size=10)

    # Services
    arm_service = ServiceProxy{mavros_msgs.srv.CommandBool}("/mavros/cmd/arming")
    offboard_service = ServiceProxy{mavros_msgs.srv.SetMode}("/mavros/set_mode")

    # High level flight controller 
    controller = Controller(N, dt, num_obs, obs_radii, pose_control_pub, att_control_pub, thrust_control_pub, arm_service, offboard_service)

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

    

    loop(controller, path_pub)
end

# if ! isinteractive()
#     main()
# end

main()