cmake_minimum_required(VERSION 2.8.3)
project(pointcloud2_segments_tools)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  pointcloud_msgs
  pcl_conversions
  laser_filters
  sensor_msgs
  std_msgs
  pcl_ros
  roscpp
  tf
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(annot src/annotation_tool.cpp)
target_link_libraries(annot ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_executable(dataset_generator src/dataset_generator.cpp)
target_link_libraries(dataset_generator ${catkin_LIBRARIES} ${PCL_LIBRARIES})

