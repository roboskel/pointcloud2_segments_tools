// This executable assumes that you have rosbags with a topic that contains only instances of one class.
// Given a folder for each of the (two) classes, it creates a .csv file containing the annotated data.

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

std::vector<std::string> getDirectoryFiles(const std::string& dir) {
    std::vector<std::string> files;
    std::shared_ptr<DIR> directory_ptr(opendir(dir.c_str()), [](DIR* dir){ dir && closedir(dir); });
    struct dirent *dirent_ptr;
    if (!directory_ptr) {
        std::cout << "Error opening : " << std::strerror(errno) << dir << std::endl;
        return files;
    }

    while ((dirent_ptr = readdir(directory_ptr.get())) != nullptr) {
        files.push_back(std::string(dirent_ptr->d_name));
    }
    return files;
}

void processRosbag(std::string rosbag_file, int class_id, std::ofstream& csv_file, std::string toi){
    try {
        rosbag::Bag bag;
        bag.open(rosbag_file,  rosbag::bagmode::Read);
        rosbag::View view(bag);
        if (toi != "") {
            view.addQuery(bag, rosbag::TopicQuery(toi));
        }

        std::string all_data, new_data;

        for (auto m : view) {
            pointcloud_msgs::PointCloud2_Segments::ConstPtr msg = m.instantiate<pointcloud_msgs::PointCloud2_Segments>();
            if (msg != nullptr) {
                for (size_t i=0; i < msg->clusters.size(); i++){
                    new_data = "[";
                    pcl::PointCloud<pcl::PointXYZRGB> cloud;
                    std::string frame_id = msg->header.frame_id;
                    pcl::PCLPointCloud2 cloud2;
                    pcl_conversions::toPCL( msg->clusters[i] , cloud2);
                    pcl::fromPCLPointCloud2(cloud2, cloud);
                    for (size_t k=0; k < cloud.points.size(); k++) {
                        new_data += new_data != "[" ? "," : "";
                        new_data += "(" + std::to_string(cloud.points[k].x) + "," + std::to_string(cloud.points[k].y) + "," + std::to_string(cloud.points[k].z) + ")";
                    }
                    new_data += "],";
                    new_data += std::to_string(class_id);
                    new_data += "\n";
                    all_data += new_data;
                }
            }
        }
        csv_file << all_data;
        bag.close();
    }
    catch (rosbag::BagIOException) {
        std::cout << "Could not find rosbag file." << std::endl;
    }
    catch (...) {
        std::cout << "An unexpected error occured. Please contact the maintainer if the problem persists." << std::endl;
    }
}

void generateDataset(std::string class0_folder, std::string class1_folder, std::string toi0, std::string toi1) {
    std::string output_csv;
    std::ofstream csv_file;

    try {
        std::string path = ros::package::getPath("pointcloud2_segments_tools");
        output_csv = path + PATH_SEPARATOR + generateDateTime() + ".csv";
        std::cout << "Your dataset file is going to be saved here:" << std::endl << output_csv << std::endl;
        csv_file.open(output_csv);
        std::vector<std::string> files = getDirectoryFiles(class0_folder);
        float class_progress = 0;
        for (auto file : files){
            class_progress++;
            if (file.length() > 4 and file.substr(file.length() - 4) == ".bag"){
                processRosbag(class0_folder + file, 0, csv_file, toi0);
            }
            std::cout << "Class 0 progress = " << std::to_string(int(class_progress/files.size()*100)) << " %" << std::endl;
        }
        files = getDirectoryFiles(class1_folder);
        class_progress = 0;
        for (auto file : files){
            class_progress++;
            if (file.length() > 4 and file.substr(file.length() - 4) == ".bag"){
                processRosbag(class1_folder + file, 1, csv_file, toi1);
            }
            std::cout << "Class 1 progress = " << std::to_string(int(class_progress/files.size()*100)) << " %" << std::endl;
        }

        csv_file.close();
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
    ros::init (argc, argv, "pointcloud2_segments_dataset_generator");
    ros::NodeHandle n_;
    pcl::console::setVerbosityLevel(pcl::console:: L_ERROR);

    std::string class0_folder, class1_folder, toi0, toi1;
    n_.param("pointcloud2_segments_dataset_generator/class0_folder_location", class0_folder, std::string(""));
    n_.param("pointcloud2_segments_dataset_generator/class1_folder_location", class1_folder, std::string(""));
    // If a topic of interest is not specified, all pointcloud2 segments messages (if any) are going to be processed.
    n_.param("pointcloud2_segments_dataset_generator/topic_of_interest_for_class0", toi0, std::string(""));
    n_.param("pointcloud2_segments_dataset_generator/topic_of_interest_for_class1", toi1, std::string(""));

    generateDataset(class0_folder, class1_folder, toi0, toi1);
}
