#!/usr/bin/env python3

import rospy
import actionlib
import numpy as np
import math
import tf2_ros

from actionlib_msgs.msg import GoalStatus
from follow_waypoints.msg import FollowWaypointsAction, FollowWaypointsGoal, FollowWaypointsResult, FollowWaypointsFeedback
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseFeedback
from std_msgs.msg import Float32
from geometry_msgs.msg import Pose, PoseArray
from tf.transformations import euler_from_quaternion

class FollowWaypointsServer():
    _feedback = FollowWaypointsFeedback()
    _result = FollowWaypointsResult()

    def __init__(self):

        # move_base Action Client
        self.move_base_client = actionlib.SimpleActionClient("/move_base", MoveBaseAction)
        
        self.logwarn("Connecting to move base...")
        self.move_base_client.wait_for_server()
        self.loginfo("Connected to move base!")

        # Parameters:
        self.map_frame          = rospy.get_param("~map_frame", "map")
        self.robot_frame        = rospy.get_param("~robot_frame", "base_footprint")
        self.distance_tolerance = rospy.get_param("~distance_tolerance", 3.0)
        self.max_speed          = rospy.get_param("~max_speed", 0.7)
        self.action_name        = rospy.get_param("~action_name", "/follow_waypoints_server")
        self.find_closest_point = rospy.get_param("~find_closest_point", False)
        self.update_frequency   = rospy.get_param("~update_frequency", 5.0)

        # Listen to Transfromation
        self.__tfBuffer = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self.__tf_listener = tf2_ros.TransformListener(self.__tfBuffer)
        self.rate = rospy.Rate(self.update_frequency)

        # FollowWaypoint action server
        self._as = actionlib.SimpleActionServer(self.action_name, FollowWaypointsAction,
                                                        execute_cb=self.execute_cb, auto_start=False)
        
        self._as.start()
        
        # Publishers, Subcribers:
        self.speed_limit_publisher = rospy.Publisher("speed_limit_lane", 
                                                    Float32, queue_size=1, latch=True)
        self.pose_array_publisher = rospy.Publisher("/waypoints", 
                                                    PoseArray, queue_size=1, latch=True)
        

    def execute_cb(self, goal: FollowWaypointsGoal):
        poses_array = goal.target_poses.poses
        obey_speed_limit = list(goal.obey_approach_speed_limit)
        speed_limit = list(goal.approach_speed_limit)
        if len(poses_array) == 0:
            self._result.is_success = False
            self._as.set_aborted(self._result, "Aborting on goal because it was sent with an empty poses")
            return

        """
        Navigating robot through pose list on map
        """
        waypoints = []
        arr_points = []
        for i in poses_array:
            arr_points.append([i.position.x, i.position.y,
                               i.orientation.z, i.orientation.w])

        pose_arr = np.array(arr_points)

        amr_tf = self.get2DPose()
        x, y, rz, rw = amr_tf

        amr_pose = np.array([x, y, rz, rw])

        if self.find_closest_point:
            # Calculate distance between two arrays
            distances = np.linalg.norm(pose_arr - amr_pose, axis=1)

            # Find closest array in pose_arr respective to amr_pose
            closest_index = np.argmin(distances)

            if (len(amr_pose) - 1) > closest_index :
                if distances[closest_index] < np.linalg.norm(pose_arr[closest_index+1] - pose_arr[closest_index], axis=0):
                    closest_index += 1

            try:
                target_poses_filter = pose_arr[closest_index:]
                obey_speed_limit = obey_speed_limit[closest_index:]
                speed_limit = speed_limit[closest_index:]
            except:
                target_poses_filter = pose_arr[-1]
                obey_speed_limit = obey_speed_limit[-1]
                speed_limit = speed_limit[-1]
        
        else:
            target_poses_filter = pose_arr

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
        
        self.logwarn(f"Start moving robot through {len(waypoints)} waypoints!")

        is_reach_goal = True
        distance = 10
        while not rospy.is_shutdown():
            if self._as.is_preempt_requested():
                self.loginfo('%s: Preempted' % self.action_name)
                self.move_base_client.cancel_all_goals()
                self._result.is_success = False
                self._as.set_preempted(self._result, "Preempted on goal from client")
                return
            
            if is_reach_goal:
                self.sendMoveBaseGoal(waypoints[0],
                                      obey_speed_limit[0],
                                      speed_limit[0])
                is_reach_goal = False

            else:
                if (self.move_base_client.get_state() == GoalStatus.ABORTED):
                    self._result.is_success = False
                    self._as.set_aborted(self._result, "Aborting on goal because 'move_base_client' was ABORTED")
                    return
                
                elif (self.move_base_client.get_state() == GoalStatus.PREEMPTED):
                    self._result.is_success = False
                    self._as.set_preempted(self._result, "Preempted on goal because 'move_base_client' was PREEMPTED")
                    return

                if len(waypoints) == 1:
                    if self.move_base_client.get_state() == GoalStatus.SUCCEEDED:
                        self.loginfo("Goal is SUCCESS!")
                        self._result.is_success = True
                        self._as.set_succeeded(self._result)
                        return
                    
                else:
                    if distance > self.distance_tolerance:
                        curr_pose = self.get2DPose()
                        x, y, _, _ = curr_pose
                        distance = math.sqrt(pow(waypoints[0].position.x - x, 2) + pow(waypoints[0].position.y - y, 2))
                    
                    else:
                        is_reach_goal = True
                        distance = 10
                        waypoints.pop(0)
                        obey_speed_limit.pop(0)
                        speed_limit.pop(0)
            
            self.handle_feedback(waypoints, obey_speed_limit, speed_limit)
            self.pubWaypointList(waypoints)
            self.rate.sleep()


    def handle_feedback(self, waypoints: list, obey_speed_limit: list, speed_limit: list):
        # self._feedback.base_position = feedback.base_position
        self._feedback.remain_target_poses.header.frame_id = self.map_frame
        self._feedback.remain_target_poses.header.stamp = rospy.Time.now()
        self._feedback.remain_target_poses.poses = waypoints
        self._feedback.remain_obey_approach_speed_limit = obey_speed_limit
        self._feedback.remain_approach_speed_limit = speed_limit
        self._as.publish_feedback(self._feedback)
    
    def get2DPose(self):
        """
        Take 2D Pose
        """
        try:
            trans = self.__tfBuffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                rospy.Time.now(), timeout=rospy.Duration(1.0))
            
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z
        
            rx = trans.transform.rotation.x
            ry = trans.transform.rotation.y
            rz = trans.transform.rotation.z
            rw = trans.transform.rotation.w

            orientation = euler_from_quaternion([rx, ry, rz, rw])

            return x, y, orientation[2], rw

        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
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
    

    def sendMoveBaseGoal(self, pose: Pose, obey_speed_limit: bool = False , speed_limit: float = None):
        """Assemble and send a new goal to move_base"""
        msg_speed = Float32()
        if obey_speed_limit:
            msg_speed.data = speed_limit
            
        else:
            msg_speed.data = self.max_speed
        self.speed_limit_publisher.publish(msg_speed)

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.map_frame
        goal.target_pose.pose.position = pose.position
        goal.target_pose.pose.orientation = pose.orientation

        self.move_base_client.send_goal(goal)
        # self.move_base_client.send_goal(goal, done_cb=None, active_cb=None, feedback_cb=self.handle_feedback)

    def loginfo(self, msg: str):
        msg_out = rospy.get_name() + ': ' + msg
        rospy.loginfo(msg_out)

    def logwarn(self, msg: str):
        msg_out = rospy.get_name() + ': ' + msg
        rospy.logwarn(msg_out)


if __name__== '__main__':
    rospy.init_node('follow_waypoints')
    try:
        follow_waypoints = FollowWaypointsServer()
        follow_waypoints.logwarn("Follow waypoints server node is running!")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass