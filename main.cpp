#include <iostream>
#include <opencv2/opencv.hpp>
#include "ImageHandler.cpp"

using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {
    Mat face = imread("../imgs/眼镜1.jpeg");
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

     handler.faceCheck(face);

    
    return 0;
}
