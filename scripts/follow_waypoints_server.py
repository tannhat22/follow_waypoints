#!/usr/bin/env python3

import rospy
import actionlib
import numpy as np
import math
import tf2_ros

from actionlib_msgs.msg import GoalStatus
from follow_waypoints.msg import (
    FollowWaypointsAction,
    FollowWaypointsGoal,
    FollowWaypointsResult,
    FollowWaypointsFeedback,
)
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseFeedback
from std_msgs.msg import Float32, Bool

# from sensor_msgs.msg import BatteryState
from geometry_msgs.msg import Pose, PoseArray
from tf.transformations import euler_from_quaternion

# from amr_v3_autodocking.msg import AutoDockingGoal, AutoDockingAction


class FollowWaypointsServer:
    _feedback = FollowWaypointsFeedback()
    _result = FollowWaypointsResult()

    def __init__(self):

        # move_base Action Client
        self.move_base_client = actionlib.SimpleActionClient(
            "/move_base", MoveBaseAction
        )

        self.logwarn("Connecting to move base...")
        self.move_base_client.wait_for_server()
        self.loginfo("Connected to move base!")

        # Parameters:
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.robot_frame = rospy.get_param("~robot_frame", "base_footprint")
        self.distance_tolerance = rospy.get_param("~distance_tolerance", 3.0)
        self.max_speed = rospy.get_param("~max_speed", 0.7)
        self.action_name = rospy.get_param("~action_name", "/follow_waypoints_server")
        self.find_closest_point = rospy.get_param("~find_closest_point", False)
        self.step_distance = rospy.get_param("~step_distance", 1.0)
        self.debug = rospy.get_param("~debug", False)
        self.update_frequency = rospy.get_param("~update_frequency", 10.0)

        # Listen to Transfromation
        self.__tfBuffer = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self.__tf_listener = tf2_ros.TransformListener(self.__tfBuffer)
        self.rate = rospy.Rate(self.update_frequency)

        # FollowWaypoint action server
        self._as = actionlib.SimpleActionServer(
            self.action_name,
            FollowWaypointsAction,
            execute_cb=self.execute_cb,
            auto_start=False,
        )

        self._as.start()

        # Publishers, Subcribers:
        self.speed_limit_publisher = rospy.Publisher(
            "speed_limit_lane", Float32, queue_size=1, latch=True
        )
        self.pose_array_publisher = rospy.Publisher(
            "/waypoints", PoseArray, queue_size=1, latch=True
        )
        self.run_rsc_publisher = rospy.Publisher(
            "/run_rs_controller", Bool, queue_size=1, latch=True
        )

    def execute_cb(self, goal: FollowWaypointsGoal):
        # Check is charging will undock first:
        # try:
        #     batteryMsg = rospy.wait_for_message('/battery_state', BatteryState, timeout=2.0)
        #     if batteryMsg.power_supply_status == BatteryState.POWER_SUPPLY_STATUS_CHARGING:
        #         autodockClient = actionlib.SimpleActionClient('/amr/autodock_action', AutoDockingAction)
        #         autodockClient.wait_for_server()
        #         autodockGoal = AutoDockingGoal()
        #         autodockGoal.dock_pose.header.frame_id = self.map_frame
        #         autodockGoal.dock_pose.header.stamp = rospy.Time.now()
        #         autodockGoal.mode = autodockGoal.MODE_UNDOCK
        #         autodockClient.send_goal(autodockGoal)
        #         autodockClient.wait_for_result()
        #         autodockResult = autodockClient.get_result()
        #         if not autodockResult.is_success:
        #             self._result.is_success = False
        #             self._as.set_aborted(self._result, "Aborting on goal because can't 'UNDOCK'")
        #             return
        #     # batteryMsg = rospy.wait_for_message("/battery_state", BatteryState, timeout=2.0)
        # except:
        #     self._result.is_success = False
        #     self._as.set_aborted(self._result, "Aborting on goal because can't check 'IS CHARGING'")
        #     return

        poses = goal.target_poses.poses
        obey_speed_limit = list(goal.obey_approach_speed_limit)
        speed_limit = list(goal.approach_speed_limit)
        if self.debug:
            print(f"Follow_waypoints: Receive request with {len(poses)} waypoints")
            print(
                f"Follow_waypoints: {[[poses[i].position, obey_speed_limit[i], speed_limit[i]] for i in range(len(poses))]}"
            )
        if len(poses) == 0:
            self._result.is_success = False
            self._as.set_aborted(
                self._result, "Aborting on goal because it was sent with an empty poses"
            )
            return

        """
        Navigating robot through pose list on map
        """
        waypoints = []
        waypoints_follow_path = []
        arr_points = []
        obey_speed_limit_filter = []
        speed_limit_filter = []

        amr_tf = self.get2DPose()
        amrPoseMsg = Pose()
        amrPoseMsg.position.x = amr_tf[0]
        amrPoseMsg.position.y = amr_tf[1]

        for i in range(len(poses)):
            if i < (len(poses) - 1):
                if self.distance(poses[i], poses[i + 1]) == 0.0:
                    if obey_speed_limit[i]:
                        obey_speed_limit[i + 1] = obey_speed_limit[i]
                        speed_limit[i + 1] = speed_limit[i]
                    elif obey_speed_limit[i + 1]:
                        obey_speed_limit[i] = obey_speed_limit[i + 1]
                        speed_limit[i] = speed_limit[i + 1]

                if self.distance(amrPoseMsg, poses[i]) > self.distance_tolerance:
                    arr_points.append(
                        [
                            poses[i].position.x,
                            poses[i].position.y,
                            poses[i].orientation.z,
                            poses[i].orientation.w,
                        ]
                    )
                    obey_speed_limit_filter.append(obey_speed_limit[i])
                    speed_limit_filter.append(speed_limit[i])

            else:
                arr_points.append(
                    [
                        poses[i].position.x,
                        poses[i].position.y,
                        poses[i].orientation.z,
                        poses[i].orientation.w,
                    ]
                )
                obey_speed_limit_filter.append(obey_speed_limit[i])
                speed_limit_filter.append(speed_limit[i])

        poses_arr = np.array(arr_points)

        # amr_tf = self.get2DPose()
        x, y, rz, rw = amr_tf

        amr_pose = np.array([x, y, rz, rw])

        if self.find_closest_point:
            # Calculate distance between two arrays
            distances = np.linalg.norm(poses_arr - amr_pose, axis=1)

            # Find closest array in poses_arr respective to amr_pose
            closest_index = np.argmin(distances)

            if (len(poses_arr) - 1) > closest_index:
                if distances[closest_index] < np.linalg.norm(
                    poses_arr[closest_index + 1] - poses_arr[closest_index], axis=0
                ):
                    closest_index += 1

            target_poses_filter = poses_arr[closest_index:]
            obey_speed_limit_filter = obey_speed_limit_filter[closest_index:]
            speed_limit_filter = speed_limit_filter[closest_index:]

        else:
            target_poses_filter = poses_arr

        # if (len(obey_speed_limit) > 1
        #     and obey_speed_limit[-2]):
        #     obey_speed_limit[-1] = True
        #     speed_limit[-1] = speed_limit[-2]

        for point in target_poses_filter:
            pose = Pose()
            pose.position.x = float(point[0])
            pose.position.y = float(point[1])
            pose.position.z = 0.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = float(point[2])
            pose.orientation.w = float(point[3])
            waypoints.append(pose)

        # Them cac waypoints de di theo lane:
        for i in range(len(waypoints) - 1):
            waypoints_follow_path.append(waypoints[i])
            waypoints_follow_path += self.find_points_on_line(
                waypoints[i], waypoints[i + 1], self.step_distance
            )
        waypoints_follow_path.append(waypoints[len(waypoints) - 1])

        self.logwarn(f"Start moving robot through {len(waypoints)} waypoints!")
        self.run_rsc_publisher.publish(True)

        is_reach_goal = True
        distance = 10
        while not rospy.is_shutdown():
            if self._as.is_preempt_requested():
                self.loginfo("%s: Preempted" % self.action_name)
                self.move_base_client.cancel_all_goals()
                self._result.is_success = False
                self._as.set_preempted(self._result, "Preempted on goal from client")
                return

            if is_reach_goal:
                self.sendMoveBaseGoal(
                    waypoints_follow_path[0],
                    obey_speed_limit_filter[0],
                    speed_limit_filter[0],
                )
                is_reach_goal = False

            else:
                if self.move_base_client.get_state() == GoalStatus.ABORTED:
                    self._result.is_success = False
                    self._as.set_aborted(
                        self._result,
                        "Aborting on goal because 'move_base_client' was ABORTED",
                    )
                    return

                elif self.move_base_client.get_state() == GoalStatus.PREEMPTED:
                    self._result.is_success = False
                    self._as.set_preempted(
                        self._result,
                        "Preempted on goal because 'move_base_client' was PREEMPTED",
                    )
                    return

                if len(waypoints_follow_path) == 1:
                    if self.move_base_client.get_state() == GoalStatus.SUCCEEDED:
                        self.loginfo("Goal is SUCCESS!")
                        self._result.is_success = True
                        self._as.set_succeeded(self._result)
                        return

                else:
                    if distance > self.distance_tolerance:
                        curr_pose = self.get2DPose()
                        x, y, _, _ = curr_pose
                        distance = math.sqrt(
                            pow(waypoints_follow_path[0].position.x - x, 2)
                            + pow(waypoints_follow_path[0].position.y - y, 2)
                        )

                    else:
                        is_reach_goal = True
                        distance = 10
                        if self.distance(waypoints[0], waypoints_follow_path[0]) == 0.0:
                            waypoints.pop(0)
                            obey_speed_limit_filter.pop(0)
                            speed_limit_filter.pop(0)
                        waypoints_follow_path.pop(0)

            self.handle_feedback(waypoints, obey_speed_limit_filter, speed_limit_filter)
            if self.debug:
                self.pubWaypointList(waypoints_follow_path)
            self.rate.sleep()

    def handle_feedback(
        self, waypoints: list, obey_speed_limit: list, speed_limit: list
    ):
        # self._feedback.base_position = feedback.base_position
        self._feedback.remain_target_poses.header.frame_id = self.map_frame
        self._feedback.remain_target_poses.header.stamp = rospy.Time.now()
        self._feedback.remain_target_poses.poses = waypoints
        # self._feedback.remain_target_follow_poses = waypoints_follow_path
        self._feedback.remain_obey_approach_speed_limit = obey_speed_limit
        self._feedback.remain_approach_speed_limit = speed_limit
        self._as.publish_feedback(self._feedback)

    def distance(self, point1: Pose, point2: Pose):
        return math.sqrt(
            (point2.position.x - point1.position.x) ** 2
            + (point2.position.y - point1.position.y) ** 2
        )

    def find_points_on_line(self, start: Pose, end: Pose, step: float):
        if self.distance(start, end) <= step:
            return []
        point1 = (start.position.x, start.position.y)
        point2 = (end.position.x, end.position.y)
        if point2[0] == point1[0] and point2[1] == point1[1]:
            return []
        # Chuyển điểm thành vector
        v1 = np.array(point1)
        v2 = np.array(point2)

        # Tính vector hướng
        direction = v2 - v1

        # Tính chiều dài của đoạn thẳng
        line_length = np.linalg.norm(direction)

        # Chia vector hướng thành vector đơn vị
        unit_direction = direction / line_length

        # Tính số điểm cần tạo trên đoạn thẳng
        num_points = math.floor(line_length / step)

        # Tạo danh sách các điểm trên đoạn thẳng
        points_on_line = []
        for i in range(1, num_points):
            v = v1 + i * unit_direction * step
            point = Pose()
            point.position.x = v[0]
            point.position.y = v[1]
            point.position.z = 0
            point.orientation = start.orientation
            points_on_line.append(point)
        return points_on_line

    def get2DPose(self):
        """
        Take 2D Pose
        """
        try:
            trans = self.__tfBuffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                rospy.Time.now(),
                timeout=rospy.Duration(1.0),
            )

            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z

            rx = trans.transform.rotation.x
            ry = trans.transform.rotation.y
            rz = trans.transform.rotation.z
            rw = trans.transform.rotation.w

            orientation = euler_from_quaternion([rx, ry, rz, rw])

            return x, y, orientation[2], rw

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            self.logwarn(f"Failed lookup: {self.robot_frame}, from {self.map_frame}")
            return None

    def pubWaypointList(self, waypoints):
        """Helper method to publish the waypoints that should be followed."""
        try:
            self.pose_array_publisher.publish(self.toPoseArray(waypoints))
            return True
        except:
            return False

    # helper methods
    def toPoseArray(self, waypoints):
        """Publish waypoints as a pose array so that you can see them in rviz."""
        poses = PoseArray()
        poses.header.frame_id = self.map_frame
        poses.poses = [pose for pose in waypoints]
        return poses

    def sendMoveBaseGoal(
        self, pose: Pose, obey_speed_limit: bool = False, speed_limit: float = None
    ):
        """Assemble and send a new goal to move_base"""
        msg_speed = Float32()
        if obey_speed_limit:
            msg_speed.data = speed_limit

        else:
            msg_speed.data = self.max_speed
        if self.debug:
            print("Obey speed: ", obey_speed_limit)
            print("speed_limit: ", speed_limit)
        self.speed_limit_publisher.publish(msg_speed)

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.map_frame
        goal.target_pose.pose.position = pose.position
        goal.target_pose.pose.orientation = pose.orientation

        self.move_base_client.send_goal(goal)
        # self.move_base_client.send_goal(goal, done_cb=None, active_cb=None, feedback_cb=self.handle_feedback)

    def loginfo(self, msg: str):
        msg_out = rospy.get_name() + ": " + msg
        rospy.loginfo(msg_out)

    def logwarn(self, msg: str):
        msg_out = rospy.get_name() + ": " + msg
        rospy.logwarn(msg_out)


if __name__ == "__main__":
    rospy.init_node("follow_waypoints")
    try:
        follow_waypoints = FollowWaypointsServer()
        follow_waypoints.logwarn("Follow waypoints server node is running!")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
