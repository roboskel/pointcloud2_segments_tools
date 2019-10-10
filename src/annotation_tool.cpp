#include <ctime>
#include <stdio.h>
#include <fstream>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <ros/package.h>
#include <rosbag/view.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/impl/point_types.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <pointcloud_msgs/PointCloud2_Segments.h>

#if defined(WIN32) || defined(_WIN32) 
    #define PATH_SEPARATOR "\\" 
#else 
    #define PATH_SEPARATOR "/" 
#endif 

std::string generateDateTime() {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer [80];

    time (&rawtime);
    timeinfo = localtime (&rawtime);

    strftime (buffer,sizeof(buffer),"%d-%b-%Y_%H-%M-%S",timeinfo);
    return buffer;
}

void processRosbag(std::string rosbag_file, std::string toi, std::string out_topic, ros::NodeHandle n_) {
    rosbag::Bag bag;
    std::string output_csv;
    std::ofstream csv_file;

    try {
        bag.open(rosbag_file,  rosbag::bagmode::Read);
        rosbag::View view(bag);
        if (toi != "") {
            view.addQuery(bag, rosbag::TopicQuery(toi));
        }

        ros::Publisher pub = n_.advertise<sensor_msgs::PointCloud2> (out_topic, 1);

        std::string user_input, new_data;
        uint class_;

        std::string path = ros::package::getPath("pointcloud2_segments_tools");
        std::cout << generateDateTime() << std::endl;
        output_csv = path + PATH_SEPARATOR + generateDateTime() + ".csv";
        std::cout << "Your annotated data file is going to be saved here:" << std::endl << output_csv << std::endl;
        csv_file.open(output_csv);

        std::cout << "Make sure you rviz is running and you have subscribed to " << out_topic << std::endl;
        std::cout << "When you are done, press [enter] to continue..." << std::endl;
        std::getline(std::cin, user_input);

        bool quit = false;

        for (auto m : view) {
            if (quit) {
                break;
            }
            pointcloud_msgs::PointCloud2_Segments::ConstPtr msg = m.instantiate<pointcloud_msgs::PointCloud2_Segments>();
            if (msg != nullptr) {
                for (size_t i=0; i < msg->cluster_id.size(); i++){
                    sensor_msgs::PointCloud2 out_msg;
                    sensor_msgs::PointCloud2 accumulator;
                    pcl::PointCloud<pcl::PointXYZRGB> cloud;
                    pcl::PointCloud<pcl::PointXYZRGB> current_candidate;
                    std::string frame_id = msg->header.frame_id;
                    for (size_t j=0; j < msg->clusters.size(); j++){
                        pcl::PCLPointCloud2 cloud2;
                        pcl_conversions::toPCL( msg->clusters[j] , cloud2);
                        pcl::fromPCLPointCloud2(cloud2, cloud);
                        if (j == i) {
                            for (size_t k=0; k < cloud.points.size(); k++) {
                                cloud.points[k].r = 255;
                                cloud.points[k].g = 0;
                                cloud.points[k].b = 0;
                                pcl::fromPCLPointCloud2(cloud2, current_candidate);
                            }
                        }
                        else {
                            for (size_t k=0; k < cloud.points.size(); k++) {
                                cloud.points[k].r = 0;
                                cloud.points[k].g = 0;
                                cloud.points[k].b = 0;
                            }
                        }
                        pcl::PCLPointCloud2 cloud2_;
                        pcl::toPCLPointCloud2(cloud, cloud2_);
                        pcl_conversions::fromPCL(cloud2_, out_msg);
                        out_msg.header.stamp = ros::Time::now();
                        out_msg.header.frame_id = frame_id;
                        sensor_msgs::PointCloud2 tmp = sensor_msgs::PointCloud2(accumulator);
                        pcl::concatenatePointCloud( out_msg, tmp, accumulator);
                    }
                    pub.publish(accumulator);
                    user_input = "";
                    while(user_input != "s" && user_input != "n" && user_input != "y" && user_input != "q"){
                        std::cout << "Is the red cluster a human? (y)es/(n)o/(s)kip/(q)uit" << std::endl;
                        std::getline(std::cin, user_input);
                    }

                    if (user_input == "q"){
                        quit = true;
                        break;
                    }
                    if (user_input != "s") {
                        new_data = "[";
                        class_ = user_input == "y" ? 1 : 0;
                        for (size_t k=0; k < cloud.points.size(); k++) {
                            new_data += new_data != "[" ? "," : "";
                            new_data += "(" + std::to_string(current_candidate.points[k].x) + "," + std::to_string(current_candidate.points[k].y) + "," + std::to_string(current_candidate.points[k].z) + ")";
                        }
                        new_data += "],";
                        new_data += std::to_string(class_);
                        new_data += "\n";
                        csv_file << new_data;
                        std::cout << new_data << std::endl;
                    }
                }
            }
        }

        csv_file.close();
        bag.close();
    }
    catch (rosbag::BagIOException) {
        std::cout << "Could not find rosbag file." << std::endl;
    }
    catch (...) {
        std::cout << "An unexpected error occured. Please contact the maintainer if the problem persists." << std::endl;
    }
    // Delete the file if there is no data in it
    std::ifstream csv_file_(output_csv);
    if (csv_file_.peek() == std::ifstream::traits_type::eof()) {
        std::remove(output_csv.c_str());
    }
}

int main (int argc, char** argv) {
    ros::init (argc, argv, "pointcloud2_segments_annotation_tool");
    ros::NodeHandle n_;
    pcl::console::setVerbosityLevel(pcl::console:: L_ERROR);

    std::string out_topic, toi, rbf;
    n_.param("pointcloud2_segments_annotation_tool/rosbag_file", rbf, std::string(""));
    // If a topic of interest is not specified, all pointcloud2 segments messages (if any) are going to be processed.
    n_.param("pointcloud2_segments_annotation_tool/topic_of_interest", toi, std::string(""));
    n_.param("pointcloud2_segments_annotation_tool/out_topic", out_topic, std::string("/pointcloud2_segments_annotation_tool/pointcloud2"));

    processRosbag(rbf, toi, out_topic, n_);
}
