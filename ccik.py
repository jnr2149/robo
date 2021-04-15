#!/usr/bin/env python3

import math
import numpy
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Transform
from cartesian_control.msg import CartesianCommand
from urdf_parser_py.urdf import URDF
import random
import tf
from threading import Thread, Lock


'''This is a class which will perform both cartesian control and inverse
   kinematics'''
class CCIK(object):
    def __init__(self):
    #Load robot from parameter server
        self.robot = URDF.from_parameter_server()

    #Subscribe to current joint state of the robot
        rospy.Subscriber('/joint_states', JointState, self.get_joint_state)

    #This will load information about the joints of the robot
        self.num_joints = 0
        self.joint_names = []
        self.q_current = []
        self.joint_axes = []
        self.get_joint_info()

    #This is a mutex
        self.mutex = Lock()

    #Subscribers and publishers for for cartesian control
        rospy.Subscriber('/cartesian_command', CartesianCommand, self.get_cartesian_command)
        self.velocity_pub = rospy.Publisher('/joint_velocities', JointState, queue_size=10)
        self.joint_velocity_msg = JointState()

        #Subscribers and publishers for numerical IK
        rospy.Subscriber('/ik_command', Transform, self.get_ik_command)
        self.joint_command_pub = rospy.Publisher('/joint_command', JointState, queue_size=10)
        self.joint_command_msg = JointState()

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

    '''This is the callback which will be executed when the cartesian control
       recieves a new command. The command will contain information about the
       secondary objective and the target q0. At the end of this callback, 
       you should publish to the /joint_velocities topic.'''
    def get_cartesian_command(self, command):
        self.mutex.acquire()
        #--------------------------------------------------------------------------
        #FILL IN YOUR PART OF THE CODE FOR CARTESIAN CONTROL HERE

        
       
 
        #Initalize 
        delta_x = numpy.zeros((6,1))

        self.joint_velocity_msg.name = self.joint_names
        joint_values = self.q_current
       
        #forward kinematics 
        joint_transforms, b_T_ee_c = self.forward_kinematics(joint_values)
        
        #compute b_T_ee desired 
        x_tran2 = tf.transformations.translation_matrix((command.x_target.translation.x,command.x_target.translation.y,command.x_target.translation.z))
        x_rot2 = tf.transformations.quaternion_matrix((command.x_target.rotation.x,command.x_target.rotation.y,command.x_target.rotation.z,command.x_target.rotation.w))
        b_T_ee_d = (numpy.dot(x_tran2, x_rot2))
        
        eec_T_eed = numpy.dot(numpy.linalg.inv(b_T_ee_c), b_T_ee_d)
        
        
        #first three rows
        delta_x[0:3,0] = eec_T_eed[0:3,3]
        #last three rows
        angle, axis = self.rotation_from_matrix(eec_T_eed)
        delta_x[3:6,0] = numpy.transpose(numpy.dot(axis, angle))

        
        #scale Vee to 0.1m/s and 1 rad/s
        if delta_x[0] > 0.1 or delta_x[1] > 0.1 or delta_x[2] > 0.1 or numpy.linalg.norm((delta_x[0], delta_x[1], delta_x[2])) >0.1:
                norm10 = 10*numpy.linalg.norm((delta_x[0],delta_x[1],delta_x[2]))
                delta_x[0] = delta_x[0]/norm10
                delta_x[1] = delta_x[1]/norm10
                delta_x[2] = delta_x[2]/norm10
        
        if delta_x[3] > 1 or delta_x[4] > 1 or delta_x[5] > 1 or numpy.linalg.norm((delta_x[3], delta_x[4], delta_x[5])) >0.1:
                norm10 = numpy.linalg.norm((delta_x[3],delta_x[4],delta_x[5]))
                delta_x[3] = delta_x[3]/norm10
                delta_x[4] = delta_x[4]/norm10
                delta_x[5] = delta_x[5]/norm10
                
        #desired veloctiy vee
        gain_p = 1 
        v_ee = numpy.dot(gain_p, delta_x)

        #Jacobian and pseudoinverse 
        J = self.get_jacobian(b_T_ee_c, joint_transforms)
        Jpsinv = numpy.linalg.pinv(J, 0.01)
        
        #where q_dot = Jpsinv*Vee
        q_dot = numpy.dot(Jpsinv, v_ee)


        #secondary objective 
        if command.secondary_objective:
                delta_x_sec = numpy.zeros((self.num_joints, 1))
                delta_x_sec[0,0] = command.q0_target - self.q_current[0]
                gain_ps = 3 
                qdot_sec = numpy.dot(gain_ps, delta_x_sec)

        #compute qnull 
                j_null = numpy.identity(self.num_joints) - numpy.dot(numpy.linalg.pinv(J), J)
                qdot_jnull = numpy.dot(j_null,qdot_sec)

        #add to previously computed solution
                q_dot = q_dot + qdot_jnull

        #scale resulting qdot 1 rad/s 
        if any(i>1 for i in q_dot):
                q_dot = numpy.divide(q_dot, numpy.linalg.norm(q_dot))

        self.joint_velocity_msg.name = self.joint_names
        self.joint_velocity_msg.velocity = q_dot
        self.velocity_pub.publish(self.joint_velocity_msg)
        
        #--------------------------------------------------------------------------
        self.mutex.release()

    '''This is a function which will assemble the jacobian of the robot using the
       current joint transforms and the transform from the base to the end
       effector (b_T_ee). Both the cartesian control callback and the
       inverse kinematics callback will make use of this function.
       Usage: J = self.get_jacobian(b_T_ee, joint_transforms)'''
    def get_jacobian(self, b_T_ee, joint_transforms):
        J = numpy.zeros((6,self.num_joints))
        #--------------------------------------------------------------------------
        #FILL IN YOUR PART OF THE CODE FOR ASSEMBLING THE CURRENT JACOBIAN HERE
        #find skew symmetric
        def s_matrix(w):
                s = numpy.zeros((3,3))
                s[0,1] = -w[2]
                s[0,2] = w[1]
                s[1,0] = w[2]
                s[1,2] = -w[0]
                s[2,0] = -w[1]
                s[2,1] = w[0]
                return s
               
            
        v_j = numpy.zeros((6,6)) 
        
       
        for i in range(0,self.num_joints):  
        #joint to ee transform 
                j_T_ee = (numpy.dot(numpy.linalg.inv(joint_transforms[i]), b_T_ee))
          
        #calculate ee_R_j
                ee_R_j = numpy.linalg.inv(j_T_ee)[0:3, 0:3]
                j_T_ee_t = j_T_ee[0:3,3]
        
        #find velocity of J
                v_j[0:3,0:3] = v_j[3:6,3:6] = ee_R_j
                v_j[0:3,3:6] = (-1)*numpy.dot((ee_R_j), s_matrix(j_T_ee_t))
                 
        
                J[:, i] = numpy.dot(v_j[:,3:6], self.joint_axes[i])
                
                
       
        #--------------------------------------------------------------------------
        return J

    '''This is the callback which will be executed when the inverse kinematics
       recieve a new command. The command will contain information about desired
       end effector pose relative to the root of your robot. At the end of this
       callback, you should publish to the /joint_command topic. This should not
       search for a solution indefinitely - there should be a time limit. When
       searching for two matrices which are the same, we expect numerical
       precision of 10e-3.'''
    def get_ik_command(self, command):
        self.mutex.acquire()
        #--------------------------------------------------------------------------
        #FILL IN YOUR PART OF THE CODE FOR INVERSE KINEMATICS HERE
        
    
        #initial 
        
        delta_x = numpy.zeros((6,1))
        delta_q = numpy.ones((self.num_joints,1))
        i = 0
        
        #rand q_c
        while i < 3:
                q = numpy.random.rand((self.num_joints),1)
                first = rospy.get_time()
         
        #timeout of 10 sec and runs three times 
                while rospy.get_time() - first <10 and not all(numpy.abs(j) < 0.001 for j in delta_q):
        #fwk to get new values
                        joint_transforms, b_T_ee_c = self.forward_kinematics(q)
        
        #calculate delta_x
                        x_tran2 = tf.transformations.translation_matrix((command.translation.x,command.translation.y,command.translation.z))
                        x_rot2 = tf.transformations.quaternion_matrix((command.rotation.x,command.rotation.y,command.rotation.z,command.rotation.w))
                        b_T_ee_d = (numpy.dot(x_tran2, x_rot2))
        
                        eec_T_eed = numpy.dot(numpy.linalg.inv(b_T_ee_c), b_T_ee_d)
                        delta_x[0:3,0] = eec_T_eed[0:3,3]
                        angle, axis = self.rotation_from_matrix(eec_T_eed)
                        delta_x[3:6,0] = numpy.transpose(numpy.dot(axis, angle))
        
        #find Jacobian and psuedoinverse
                        J = self.get_jacobian(b_T_ee_c, joint_transforms)
                        Jpsinv = numpy.linalg.pinv(J, 0.01)
       
        
        #calculate delta_q
                        delta_q = numpy.dot(Jpsinv, delta_x)
        
        
        #add q and delta q
                        q += delta_q
                 
                if all(numpy.abs(j) < 0.001 for j in delta_q):
                        i = 3 
                else:
                        i += 1
                             
                        
        self.joint_command_msg.name = self.joint_names
        self.joint_command_msg.position = q
        self.joint_command_pub.publish(self.joint_command_msg)
        
        #--------------------------------------------------------------------------
        self.mutex.release()

    '''This function will return the angle-axis representation of the rotation
       contained in the input matrix. Use like this: 
       angle, axis = rotation_from_matrix(R)'''
    def rotation_from_matrix(self, matrix):
        R = numpy.array(matrix, dtype=numpy.float64, copy=False)
        R33 = R[:3, :3]
        # axis: unit eigenvector of R33 corresponding to eigenvalue of 1
        l, W = numpy.linalg.eig(R33.T)
        i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
        if not len(i):
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
        axis = numpy.real(W[:, i[-1]]).squeeze()
        # point: unit eigenvector of R33 corresponding to eigenvalue of 1
        l, Q = numpy.linalg.eig(R)
        i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
        if not len(i):
            raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
        # rotation angle depending on axis
        cosa = (numpy.trace(R33) - 1.0) / 2.0
        if abs(axis[2]) > 1e-8:
            sina = (R[1, 0] + (cosa-1.0)*axis[0]*axis[1]) / axis[2]
        elif abs(axis[1]) > 1e-8:
            sina = (R[0, 2] + (cosa-1.0)*axis[0]*axis[2]) / axis[1]
        else:
            sina = (R[2, 1] + (cosa-1.0)*axis[1]*axis[2]) / axis[0]
        angle = math.atan2(sina, cosa)
        return angle, axis

    '''This is the function which will perform forward kinematics for your 
       cartesian control and inverse kinematics functions. It takes as input
       joint values for the robot and will return an array of 4x4 transforms
       from the base to each joint of the robot, as well as the transform from
       the base to the end effector.
       Usage: joint_transforms, b_T_ee = self.forward_kinematics(joint_values)'''
    def forward_kinematics(self, joint_values):
        joint_transforms = []

        link = self.robot.get_root()
        T = tf.transformations.identity_matrix()

        while True:
            if link not in self.robot.child_map:
                break

            (joint_name, next_link) = self.robot.child_map[link][0]
            joint = self.robot.joint_map[joint_name]

            T_l = numpy.dot(tf.transformations.translation_matrix(joint.origin.xyz), tf.transformations.euler_matrix(joint.origin.rpy[0], joint.origin.rpy[1], joint.origin.rpy[2]))
            T = numpy.dot(T, T_l)

            if joint.type != "fixed":
                joint_transforms.append(T)
                q_index = self.joint_names.index(joint_name)
                T_j = tf.transformations.rotation_matrix(joint_values[q_index], numpy.asarray(joint.axis))
                T = numpy.dot(T, T_j)

            link = next_link
        return joint_transforms, T #where T = b_T_ee

    '''This is the callback which will recieve and store the current robot
       joint states.'''
    def get_joint_state(self, msg):
        self.mutex.acquire()
        self.q_current = []
        for name in self.joint_names:
            self.q_current.append(msg.position[msg.name.index(name)])
        self.mutex.release()


if __name__ == '__main__':
    rospy.init_node('cartesian_control_and_IK', anonymous=True)
    CCIK()
    rospy.spin()
