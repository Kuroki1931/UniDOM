#include <ros/ros.h>
#include <actionlib/server/simple_action_server.h>
#include <xarm_gripper/MoveAction.h>
#include <xarm_api/xarm_ros_client.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/joint_state_interface.h>
#include <hardware_interface/robot_hw.h>
#include <controller_manager/controller_manager.h>
#include <control_msgs/GripperCommandAction.h>

// Please run "export ROS_NAMESPACE=/xarm" first

namespace xarm_control
{
    class GripperInterface : public hardware_interface::RobotHW
    {
    protected:
        ros::NodeHandle nh_;
        actionlib::SimpleActionServer<control_msgs::GripperCommandAction> as_; // NodeHandle instance must be created before this line. Otherwise strange error occurs.
        std::string action_name_;
        // create messages that are used to published feedback/result
        control_msgs::GripperCommandFeedback feedback_;
        control_msgs::GripperCommandResult result_;

        xarm_api::XArmROSClient xarm;

    private:
        hardware_interface::JointStateInterface jnt_state_interface;
        double cmd;
        double pos;
        double vel;
        double eff;

    public:
        GripperInterface(std::string name,std::string joint_prefix="") : as_(nh_, name, boost::bind(&GripperInterface::executeCB, this, _1), false),
                                             action_name_(name)
        {
            xarm.init(nh_);
            as_.start();

            // hardware interfaceの設定
            hardware_interface::JointStateHandle state_handle_a(joint_prefix+"drive_joint", &pos, &vel, &eff);
            jnt_state_interface.registerHandle(state_handle_a);
            registerInterface(&jnt_state_interface);
        }
        ~GripperInterface(){};
        void executeCB(const control_msgs::GripperCommandGoalConstPtr &goal);
        void read();
        void write();
    };

    void GripperInterface::executeCB(const control_msgs::GripperCommandGoalConstPtr &goal)
    {
        ros::Rate r(10);
        int ret = 0;
        const int pulse_speed = 2000; // 今は簡単のためにgripperの開閉速度を一定にする
        int target_pulse = (int)((0.85 - goal->command.position) * 1000);
        feedback_.position = 0.0;

        // publish info to the console for the user
        ROS_INFO("Executing, creating GripperAction ");

        // start executing the action
        if (xarm.gripperConfig(pulse_speed))
        {
            ROS_INFO("%s: Aborted, not ready", action_name_.c_str());
            as_.setAborted();
            return;
        }

        if (xarm.gripperMove(target_pulse))
        {
            ROS_INFO("%s: Aborted, not ready", action_name_.c_str());
            as_.setAborted();
            return;
        }

        float fdb_pulse = 0;
        int fdb_err = 0;
        float curr_pos = 0.0;
        ret = xarm.getGripperState(&fdb_pulse, &fdb_err);

        while (!ret && fabs(fdb_pulse - target_pulse) > 10)
        {
            // TODO:Gripperの状態(stalled,reached_goal)をちゃんとfeedbackとして返すようにする
            curr_pos =  0.85 - fdb_pulse / 1000.0;
            feedback_.position = curr_pos;
            feedback_.stalled = false;
            feedback_.reached_goal = false;

            as_.publishFeedback(feedback_);

            if (as_.isPreemptRequested() || !ros::ok())
            {
                ROS_INFO("%s: Preempted", action_name_.c_str());
                // set the action state to preempted
                xarm.gripperMove(fdb_pulse);
                as_.setPreempted();
                return;
            }

            r.sleep();

            ret = xarm.getGripperState(&fdb_pulse, &fdb_err);
        }

         curr_pos =  0.85 - fdb_pulse / 1000.0;
        if (!ret)
        {
            feedback_.position = curr_pos;
            result_.reached_goal = true;
            // TODO:Gripperの状態(stalled,reached_goal)をちゃんとfeedbackとして返すようにする
            result_.stalled = false; //(fdb_err);
            ROS_INFO("%s: Succeeded, err_code: %d", action_name_.c_str(), fdb_err);
            as_.setSucceeded(result_);
        }

        else
        {
            feedback_.position =  curr_pos;
            result_.reached_goal = false;
            // TODO:Gripperの状態(stalled,reached_goal)をちゃんとfeedbackとして返すようにする
            result_.stalled = false; //(fdb_err);
            ROS_INFO("%s: Failed, ret = %d, err_code: %d", action_name_.c_str(), ret, fdb_err);
            as_.setAborted();
        }
    }
    void GripperInterface::read()
    {
        float fdb_pulse = 0;
        int fdb_err = 0;
        xarm.getGripperState(&fdb_pulse, &fdb_err);
        pos = 0.85 - fdb_pulse / 1000;
        vel = 0;
        eff = 0;
    }
    void GripperInterface::write()
    {
        // TODO:状態をhardware interfaceを通じて行う場合は、ここで実装する
    }

}

int main(int argc, char **argv)
{

    ros::init(argc, argv, "gripper_control_node");

    //joint_prefixを取得
    std::string joint_prefix="";
    ros::NodeHandle pnh("~");
    pnh.getParam("joint_prefix",joint_prefix);
    

    xarm_control::GripperInterface gripper("xarm_gripper",joint_prefix);
    controller_manager::ControllerManager cm(&gripper);

    ros::AsyncSpinner spinner(1);
    spinner.start();

    ros::Time t = ros::Time::now();
    ros::Rate rate(10);

    while (ros::ok())
    {
        ros::Duration d = ros::Time::now() - t;
        ros::Time t = ros::Time::now();
        gripper.read();
        cm.update(t, d);
        gripper.write();
        rate.sleep();
    }
    return 0;
}