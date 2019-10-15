# pointcloud2_segments_tools
A repository for various tools specifically created to handle pointcloud2_segments messages.

## Currently included
* A dummy pointcloud2_segments tool for quick (and dirty) annotations. (`annot.launch`)
* Two launch files for laserscan filtering using the laser_filters/LaserScanBoxFilter. (`laserscan_filter_humans_only.launch` and `laserscan_filter_walls_only.launch`)
* A tool that reads rosbag files and creates a dataset. The tool currently supports only two classes and each rosbag has to contain only one of the two. The filtering launch files provided in this repo could help with this requirement.

