#!/usr/bin/env python3

import numpy
import random
import sys

import geometry_msgs.msg
import moveit_msgs.msg
import moveit_msgs.srv
import rospy
import tf
import moveit_commander
from urdf_parser_py.urdf import URDF
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Transform
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

def convert_to_message(T):
    t = geometry_msgs.msg.Pose()
    position = tf.transformations.translation_from_matrix(T)
    orientation = tf.transformations.quaternion_from_matrix(T)
    t.position.x = position[0]
    t.position.y = position[1]
    t.position.z = position[2]
    t.orientation.x = orientation[0]
    t.orientation.y = orientation[1]
    t.orientation.z = orientation[2]
    t.orientation.w = orientation[3]        
    return t

class MoveArm(object):

    def __init__(self):

        #Loads the robot model, which contains the robot's kinematics information
        self.num_joints = 0
        self.joint_names = []
        self.joint_axes = []
        self.robot = URDF.from_parameter_server()
        self.base = self.robot.get_root()
        self.get_joint_info()

        # Wait for moveit IK service
        rospy.wait_for_service("compute_ik")
        self.ik_service = rospy.ServiceProxy('compute_ik',  moveit_msgs.srv.GetPositionIK)
        print("IK service ready")

        # Wait for validity check service
        rospy.wait_for_service("check_state_validity")
        self.state_valid_service = rospy.ServiceProxy('check_state_validity',  
                                                      moveit_msgs.srv.GetStateValidity)
        print("State validity service ready")

        # MoveIt parameter
        robot_moveit = moveit_commander.RobotCommander()
        self.group_name = robot_moveit.get_group_names()[0]

        #Subscribe to topics
        rospy.Subscriber('/joint_states', JointState, self.get_joint_state)
        rospy.Subscriber('/motion_planning_goal', Transform, self.motion_planning)
        self.current_obstacle = "None"
        rospy.Subscriber('/obstacle', String, self.get_obstacle)

        #Set up publisher
        self.pub = rospy.Publisher('/joint_trajectory', JointTrajectory, queue_size=10)

    '''This callback provides you with the current joint positions of the robot 
     in member variable q_current.'''
    def get_joint_state(self, msg):
        self.q_current = []
        for name in self.joint_names:
            self.q_current.append(msg.position[msg.name.index(name)])

    '''This callback provides you with the name of the current obstacle which
    exists in the RVIZ environment. Options are "None", "Simple", "Hard",
    or "Super". '''
    def get_obstacle(self, msg):
        self.current_obstacle = msg.data

    '''This is the callback which will implement your RRT motion planning.
    You are welcome to add other functions to this class (i.e. an
    "is_segment_valid" function will likely come in handy multiple times 
    in the motion planning process and it will be easiest to make this a 
    seperate function and then call it from motion planning). You may also
    create trajectory shortcut and trajectory sample functions if you wish, 
    which will also be called from the motion planning function.''' 
    #distance between two 
    def dist_two(self,q_1,q_2):
        dist = numpy.linalg.norm(numpy.subtract(q_1,q_2))
        return dist
    
    #trajectory sample function    
    def traj_sample(self,q_start,q_2,space, boolean):
        #dist between positions
        distance = self.dist_two(q_start, q_2)
        #are we closer than one segment?
        if distance < space:
                return []
        num_seg = numpy.ceil(numpy.abs(distance/space))
        next_step = []
        
        for i in range(0, len(q_start)):
                next_step.append((q_2[i]-q_start[i])/num_seg)
        sample_array = []
        q_old = q_start
        #check if valid without adding to array
        if boolean:
                for i in range(0, int(num_seg)):
                        q_new = numpy.add(q_old, numpy.asarray(next_step))
                        if not self.is_state_valid(q_new):
                                return False
                        q_old = q_new
                return True
        #return valid sample array
        else:
                for i in range(0, int(num_seg)):
                        q_new = numpy.add(q_old, numpy.asarray(next_step))
                        sample_array.append(q_new)
                        q_old = q_new
                return sample_array
    
    #check segment 
    def is_segment_valid(self,q_start,q_goal):
        seg = 0.1 
        if self.dist_two(q_start, q_goal) < seg:
                return True
        return self.traj_sample(q_start, q_goal, seg, True)
   
    #find shortest path 
    def traj_shortcut(self,q_array):
        segment = q_array[0]
        p = 0 
        q_new_array = [segment]
        for i in range(0, len(q_array)):
                if i == len(q_array)-1:
                        q_new_array.append(q_array[i])
                elif not self.is_segment_valid(segment, q_array[i]):
                        segment = q_array[i-1]
                        p = i-1
                        q_new_array.append(segment)
        return q_new_array
        
        
    def motion_planning(self, ee_goal):
        print("Starting motion planning")
    ########INSERT YOUR RRT MOTION PLANNING HERE##########
        #initialize
        result_traj = JointTrajectory()
        result_traj.points = []
        result_traj.joint_names = self.joint_names
        q_c = self.q_current
        
        #perform ik to get goal configuration in joint space 
        x_tran = tf.transformations.translation_matrix((ee_goal.translation.x,ee_goal.translation.y,ee_goal.translation.z))
        x_rot = tf.transformations.quaternion_matrix((ee_goal.rotation.x,ee_goal.rotation.y,ee_goal.rotation.z,ee_goal.rotation.w))
        T_goal = (numpy.dot(x_tran, x_rot))
        q_goal = (self.IK(T_goal))
        
        #implement rrt algorithm 
        #start with current position
        tree_path = [q_c]
        
        #do we have valid tree?
        if self.is_segment_valid(q_c,q_goal):
                q_link = self.traj_sample(q_c,q_goal, 0.5, False)
                tree_path.extend(q_link)
                
        #no then find and build random
        else:
                rand_path = [RRTBranch([], q_c)]
                r = q_c
                branch_total = 1
                
                #while loop to reach configuration
                while not self.is_segment_valid(r, q_goal):
                        r = -1*numpy.pi + numpy.dot(2*numpy.pi, numpy.random.rand(self.num_joints))
                        dist = rand_path[0].q
                        p = 0
                        
                        #finding next point
                        for i in range(0,len(rand_path)):
                                if self.dist_two(rand_path[i].q, r) < self.dist_two(dist, r):
                                        dist = rand_path[i].q
                                        p = i
                        delta_q = 0.1*numpy.subtract(dist,r)/self.dist_two(r,dist)
                        r = r + delta_q
                        
                        #are there obstacles in new segment
                        if self.is_segment_valid(rand_path[p].q, r):
                                branch_total += 1 
                                
                                #we add to branch 
                                rand_path.append(RRTBranch(rand_path[p], r))
                        else:
                                r = q_c
                                
                #continue to expand path 
                rand_path.append(RRTBranch(rand_path[len(rand_path)-1], q_goal))
                last_branch = rand_path[len(rand_path)-1]
                
                #build woking path 
                tree_path = [q_goal]
                while last_branch != []:
                        tree_path[0:0] = [last_branch.q]
                        last_branch = last_branch.parent
                
                #shortcut path and use new array
                tree_path_cut = self.traj_shortcut(tree_path)
                tree_path = [q_c]
                for i in range(0,len(tree_path_cut)-1):
                        q_link = self.traj_sample(tree_path_cut[i], tree_path_cut[i+1], 0.5, False)
                        tree_path.extend(q_link)
                        
        #get full path in right position and send
        for q in tree_path:
                new_point = JointTrajectoryPoint()
                new_point.positions = q 
                result_traj.points.append(new_point)
                
        self.pub.publish(result_traj)
        ######################################################

    """ This function will perform IK for a given transform T of the end-effector.
    It returns a list q[] of values, which are the result positions for the 
    joints of the robot arm, ordered from proximal to distal. If no IK solution 
    is found, it returns an empy list.
    """
    def IK(self, T_goal):
        req = moveit_msgs.srv.GetPositionIKRequest()
        req.ik_request.group_name = self.group_name
        req.ik_request.robot_state = moveit_msgs.msg.RobotState()
        req.ik_request.robot_state.joint_state.name = self.joint_names
        req.ik_request.robot_state.joint_state.position = numpy.zeros(self.num_joints)
        req.ik_request.robot_state.joint_state.velocity = numpy.zeros(self.num_joints)
        req.ik_request.robot_state.joint_state.effort = numpy.zeros(self.num_joints)
        req.ik_request.robot_state.joint_state.header.stamp = rospy.get_rostime()
        req.ik_request.avoid_collisions = True
        req.ik_request.pose_stamped = geometry_msgs.msg.PoseStamped()
        req.ik_request.pose_stamped.header.frame_id = self.base
        req.ik_request.pose_stamped.header.stamp = rospy.get_rostime()
        req.ik_request.pose_stamped.pose = convert_to_message(T_goal)
        req.ik_request.timeout = rospy.Duration(3.0)
        res = self.ik_service(req)
        q = []
        if res.error_code.val == res.error_code.SUCCESS:
            q = res.solution.joint_state.position
        return q

    '''This is a function which will collect information about the robot which
       has been loaded from the parameter server. It will populate the variables
       self.num_joints (the number of joints), self.joint_names and
       self.joint_axes (the axes around which the joints rotate)'''
    def get_joint_info(self):
        link = self.robot.get_root()
        while True:
            if link not in self.robot.child_map: break
            (joint_name, next_link) = self.robot.child_map[link][0]
            current_joint = self.robot.joint_map[joint_name]
            if current_joint.type != 'fixed':
                self.num_joints = self.num_joints + 1
                self.joint_names.append(current_joint.name)
                self.joint_axes.append(current_joint.axis)
            link = next_link


    """ This function checks if a set of joint angles q[] creates a valid state,
    or one that is free of collisions. The values in q[] are assumed to be values
    for the joints of the KUKA arm, ordered from proximal to distal. 
    """
    def is_state_valid(self, q):
        req = moveit_msgs.srv.GetStateValidityRequest()
        req.group_name = self.group_name
        req.robot_state = moveit_msgs.msg.RobotState()
        req.robot_state.joint_state.name = self.joint_names
        req.robot_state.joint_state.position = q
        req.robot_state.joint_state.velocity = numpy.zeros(self.num_joints)
        req.robot_state.joint_state.effort = numpy.zeros(self.num_joints)
        req.robot_state.joint_state.header.stamp = rospy.get_rostime()
        res = self.state_valid_service(req)
        return res.valid


'''This is a class which you can use to keep track of your tree branches.
It is easiest to do this by appending instances of this class to a list 
(your 'tree'). The class has a parent field and a joint position field (q). 
You can initialize a new branch like this:
RRTBranch(parent, q)
Feel free to keep track of your branches in whatever way you want - this
is just one of many options available to you.'''
class RRTBranch(object):
    def __init__(self, parent, q):
        self.parent = parent
        self.q = q


if __name__ == '__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_arm', anonymous=True)
    ma = MoveArm()
    rospy.spin()

