
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sstream> // for converting the command line parameter to integer



int main(int argc, char** argv)
{
    ros::init(argc, argv, "camera_publisher");
    ros::NodeHandle 				nh;
   	image_transport::ImageTransport it(nh);
    image_transport::Publisher 		pub = it.advertise("image_raw", 1);

    cv::VideoCapture cap("/dev/v4l/by-id/usb-USB_Developer_Android_20080411-video-index0");
    if(!cap.isOpened())
    {
        ROS_INFO("Can`t open camera!");
        return 1;
    }

    cv::Mat frame;
    sensor_msgs::ImagePtr pImageMsg;
    ros::Rate loop_rate(1);

    while (nh.ok()) {

        cap >> frame;

        if(!frame.empty()) 
        {
            cv_bridge::CvImage opencv_2_ros (std_msgs::Header(), "bgr8", frame);

            

            pub.publish(opencv_2_ros.toImageMsg());
        }

        // loop_rate.sleep();
    }
}