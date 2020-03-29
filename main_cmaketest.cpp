#include<iostream>
#include<librealsense2/rs.hpp>
#include<pcl/point_cloud.h>
#include<pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

using pcl_ptr = pcl::PointCloud<pcl::PointXYZ>::Ptr;

pcl_ptr points_to_pcl(const rs2::points& points)
{
    pcl_ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    auto sp = points.get_profile().as<rs2::video_stream_profile>();
    cloud->width = sp.width();
    cloud->height = sp.height();
    cloud->is_dense = false;
    cloud->points.resize(points.size());
    auto ptr = points.get_vertices();
    for (auto& p : cloud->points)
    {
        p.x = ptr->x;
        p.y = ptr->y;
        p.z = ptr->z;
        ptr++;
    }

    return cloud;
}

int main(){
    std::cout << "Cmake workes and exe created" << std::endl;
    
    rs2::pipeline p;
    rs2::frameset frames;
    rs2::frame fff;
    auto depth = fff.as<rs2::depth_frame>();
    rs2::pointcloud pc;
    rs2::points points;
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_points;
    rs2::config cfg;

    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    cfg.enable_stream(RS2_STREAM_DEPTH, RS2_FORMAT_ANY);
    p.start(cfg);
    frames = p.wait_for_frames();
    depth = frames.get_depth_frame();        
    points = pc.calculate(depth);
    pcl_points = points_to_pcl(points);

    viewer.showCloud(pcl_points);

    while (!viewer.wasStopped ())
    {
        // Block program until frames arrive
        frames = p.wait_for_frames();
        // Try to get a frame of a depth image
        depth = frames.get_depth_frame();        
        points = pc.calculate(depth);
        pcl_points = points_to_pcl(points);
        
        // Get the depth frame's dimensions
        float width = depth.get_width();
        float height = depth.get_height();

        // Query the distance from the camera to the object in the center of the image
        float dist_to_center = depth.get_distance(width / 2, height / 2);

        // Print the distance 
        std::cout << "The camera is facing an object " << dist_to_center << " meters away         \r";
    }

    return EXIT_SUCCESS;
}