#include <iostream>
#include <opencv2/opencv.hpp>
#include "ImageHandler.cpp"

using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {
    Mat face = imread("../imgs/test1.jpeg");
    if (face.empty()) {
        cout << "Error" << endl;
        return -1;
    }

    int mm = 8;
    int nn = 8;
    int t2[mm * nn];
    cout << "测试一下" << endl;
    ImageHandler handler = ImageHandler();
    //cv::imshow("original", face);
    //waitKey();

    //handler.getFeature(face, t2);

     handler.faceCheck();

    //面部识别代码
    //vector<Point> centers;

    //Mat faceROI = frameGray(face);
    //vector<Rect> eyes;
    
    return 0;
}
