#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <tf/transform_listener.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <math.h>
#include <pcl/filters/passthrough.h>


class VoxelizedPointCloudPublisher{
    public:
        VoxelizedPointCloudPublisher();
        void MaskedPointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);

    private:
        ros::NodeHandle nh;
        ros::NodeHandle private_nh;

        tf::TransformListener tflistener;

        ros::Subscriber masked_point_cloud_sub;

        ros::Publisher downsampled_cloud_pub;

        pcl::PointCloud<pcl::PointXYZ>::Ptr masked_point_cloud;
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud;
        pcl::PointCloud<pcl::PointXYZ>::Ptr transforming_cloud;
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtering_cloud;

        std::string target_frame;
        double leaf_size;

        void Downsample(
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out,
                float leaf_size);
        void publish_clouds(void);

        void TransformPCL(
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out);

        void SavePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in);

        void FilterCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in);

};

VoxelizedPointCloudPublisher::VoxelizedPointCloudPublisher()
:private_nh("~")
{
    //get leaf_size from parameter server
    private_nh.getParam("/leaf_size", leaf_size);
    std::cout << "leaf_size: " << leaf_size << std::endl;

    // param
    private_nh.param("target_frame", target_frame, {"eye_on_hand_camera_color_optical_frame"});
    // private_nh.param("leaf_size", leaf_size, 0.0185);//*1.2);//0.0216); //0.025*1.4);
    // private_nh.param("leaf_size", leaf_size, 0.1);

    // subscriber
    masked_point_cloud_sub = nh.subscribe("/masked_point_cloud", 1, &VoxelizedPointCloudPublisher::MaskedPointCloudCallback, this);

    // publisher
    downsampled_cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/downsampled_cloud", 1);

    // ds_c_pub = nh.advertise<sensor_msgs::PointCloud2>("/ds_c", 1);

    masked_point_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    downsampled_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    transforming_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    // ds_c_pub = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

}

void VoxelizedPointCloudPublisher::MaskedPointCloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
{
    sensor_msgs::PointCloud pc_in;
    sensor_msgs::PointCloud pc_trans;
    sensor_msgs::PointCloud2 pc2_out;
    sensor_msgs::convertPointCloud2ToPointCloud(*msg, pc_in);
    float leaf_size;
    int max_pc = 150;
    int min_pc = 120;

    int leaf_change_counts = 0;
    float leaf_change_step;
    bool adjust_leaf_size = false;

    try{
        tflistener.waitForTransform(target_frame, msg->header.frame_id, msg->header.stamp, ros::Duration(1.0));
        tflistener.transformPointCloud(target_frame, msg->header.stamp, pc_in, msg->header.frame_id, pc_trans);
        sensor_msgs::convertPointCloudToPointCloud2(pc_trans, pc2_out);
        pcl::fromROSMsg(pc2_out, *masked_point_cloud);

        std::cout << "masked point cloud size: " << masked_point_cloud->points.size() << std::endl;
        // Downsample(masked_point_cloud, transforming_cloud);

        // TransformPCL(masked_point_cloud, transforming_cloud);
        FilterCloud(masked_point_cloud);
        //FilterCloud(transforming_cloud);
        private_nh.getParam("/leaf_size", leaf_size);
        std::cout << "leaf_size: " << leaf_size << std::endl;
        Downsample(masked_point_cloud, downsampled_cloud, leaf_size);
        std::cout << "downsampled cloud size: " << downsampled_cloud->points.size() << std::endl;
        // add fine-tuning func for keep the near pointcloud size
        if (adjust_leaf_size) {
            leaf_change_step = 0.005;
            while (downsampled_cloud->points.size() > max_pc or downsampled_cloud->points.size() < min_pc)
            {
                if (downsampled_cloud->points.size() > max_pc)
                {
                    std::cout<< "point cloud size is too large, increase leaf_size" << std::endl;
                    leaf_size += leaf_change_step;
                    std::cout<< leaf_size << std::endl;
                    std::cout<< leaf_change_step << std::endl;
                    leaf_change_counts += 1;
                }
                else if (downsampled_cloud->points.size() < min_pc)
                {
                    std::cout<< "point cloud size is too small, reduce leaf_size" << std::endl;
                    leaf_size -= leaf_change_step;
                    std::cout<< leaf_size << std::endl;
                    std::cout<< leaf_change_step << std::endl;
                    leaf_change_counts += 1;
                }
                Downsample(masked_point_cloud, downsampled_cloud, leaf_size);
                std::cout << "downsampled cloud size: " << downsampled_cloud->points.size() << std::endl;
                if (leaf_change_counts > 100)
                {
                    std::cout << "leaf_change_step is big, change leaf_step_num" << std::endl;
                    leaf_change_step *= 0.9;
                    leaf_change_counts = 0;
                }
                if (downsampled_cloud->points.size() == 1){break;}
            }
            leaf_change_counts = 0;
        }

        SavePointCloud(downsampled_cloud);
        // std::cout << "downsampled cloud size: " << downsampled_cloud->points.size() << std::endl;
        publish_clouds();
    }
    catch(tf::TransformException ex){
        ROS_ERROR("%s",ex.what());
    }
}


void VoxelizedPointCloudPublisher::Downsample(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out,
        float leaf_size)
{
    pcl::VoxelGrid<pcl::PointXYZ> voxelSampler;
    voxelSampler.setInputCloud(cloud_in->makeShared());
    voxelSampler.setLeafSize(leaf_size, leaf_size, leaf_size);
    //voxelSampler.setLeafSize(leaf_size, leaf_size, leaf_size/1.3); //z軸方向の点の数を増やしたらうまく布のシワを認識するかと思ったが、効果なし。
    //voxelSampler.setLeafSize(leaf_size, leaf_size, leaf_size/1.5);
    voxelSampler.filter(*cloud_out);
}


void VoxelizedPointCloudPublisher::publish_clouds(void)
{
    sensor_msgs::PointCloud2 downsampled_cloud_ros;
    pcl::toROSMsg(*downsampled_cloud, downsampled_cloud_ros);
    // downsampled_cloud_ros.header.stamp = ros::Time::now();
    downsampled_cloud_ros.header.frame_id = target_frame;

    downsampled_cloud_pub.publish(downsampled_cloud_ros);
    downsampled_cloud->points.clear();
}

void VoxelizedPointCloudPublisher::SavePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in)
{
    for (int i=0;i< cloud_in->points.size();i++){
        // std::cout << cloud_in->points[i].y << " " << -cloud_in->points[i].x << " " << cloud_in->points[i].z << std::endl;
    }
}

void VoxelizedPointCloudPublisher::FilterCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in)
{
    // Create the filtering object
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (cloud_in);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (-0.3, 0.3);
    //pass.setFilterLimitsNegative (true);
    pass.filter (*cloud_in);
    pcl::PassThrough<pcl::PointXYZ> pass2;
    pass2.setInputCloud (cloud_in);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (-0.45, 0.45);
    pass.filter(*cloud_in);
}

void VoxelizedPointCloudPublisher::TransformPCL(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out)
{
    //**x軸を中心にtheta回転させるコード**
    Eigen::Matrix4f rotation_matrix_x;
    Eigen::Matrix4f rotation_matrix_y;
    Eigen::Matrix4f rotation_matrix_z;
    Eigen::Matrix4f rotation_matrix_trace;
    Eigen::Matrix4f scale_matrix;
    float radx = 0; //3.141592/2 * 0;
    float rady =0;// 3.141592/2 * 1.1;
    float radz = 0;//3.141592/2 * -0.2;
    float radt = 0;
    float scale = 1;//20/1.5;



    //行列を作成する 4x4
    rotation_matrix_x << \
    1,         0,         0, 0, \
    0,  cos(radx), -sin(radx), 0, \
    0,  sin(radx),  cos(radx), 0, \
    0,         0,         0, 1;

    rotation_matrix_y << \
    cos(rady),  0,  sin(rady), 0, \
    0,         1,         0, 0, \
    -sin(rady), 0,  cos(rady), 0, \
    0,         0,         0, 1;

    rotation_matrix_z << \
    cos(radz), -sin(radz), 0, 0, \
    sin(radz),  cos(radz), 0, 0, \
    0,                  0, 1, 0, \
    0,                  0, 0, 1;

    rotation_matrix_trace << \
    1,         0,           0, -47, \
    0, cos(radt), -sin(radt),  10, \
    0, sin(radt),   cos(radt), 15, \
    0,          0,           0, 1;

    scale_matrix << \
    scale, 0, 0, 0, \
    0, scale, 0, 0, \
    0, 0, scale, 0, \
    0, 0, 0,  1;

    //回転

    pcl::transformPointCloud( *cloud_in, *cloud_out, rotation_matrix_y );
    pcl::transformPointCloud( *cloud_out, *cloud_out, rotation_matrix_z );
    pcl::transformPointCloud( *cloud_out, *cloud_out, rotation_matrix_x );
    pcl::transformPointCloud( *cloud_out, *cloud_out, rotation_matrix_trace );
    pcl::transformPointCloud( *cloud_out, *cloud_out, scale_matrix);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "voxelized_point_cloud_publisher");

    VoxelizedPointCloudPublisher voxelized_point_cloud_publisher;
    ros::spin();

    return 0;
}