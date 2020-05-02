//
// Created by xysh on 2020/5/2.
//

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
//#include "ImageHandler.h

#define FACE_MODEL			"../opencv-data/haarcascades/haarcascade_frontalface_alt2.xml"
#define EYE_GLASSES_MODEL	"../opencv-data/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
#define EYE_MODEL	"../opencv-data/haarcascade_eye.xml"
#define FACEPHOTO_FACENAME  "./image/result.jpg"
#define DETECT_IMAGE		"../imgs/眼镜1.jpeg"


class  ImageHandler {
   // ImageHandler() {}

    public:
    /**
     * 获取图片的特征
     * @param m
     * @param t
     */
    void getFeature(cv::Mat m, int t[]) {
        int M = m.rows;
        int N = m.cols;
        cv::cvtColor(m, m, cv::COLOR_BGR2GRAY);

        int i,j;
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                printf("%d ", m.at<uchar>(i, j));
            }
            printf("\n");
        }
        printf("开始压缩图片\n");
        int mm = 8;
        int nn = 8;

        int FeatureResult[mm * nn];
        cv::Mat Feature(mm, nn, CV_32FC1, cv::Scalar::all(0));
        printf("压缩第一步\n");
        for (i = 0; i< M; i++) {
            for (j = 0; j < N; j++) {
                Feature.at<int> (i / (M/nn), j / (N / nn)) = Feature.at<int> (i / (M/mm), j / (N / nn)) + m.at<uchar>(i, j);
            }
        }

        printf("图片压缩成功\n");
        //测试：看看缩小后，图像里面的每个像素值是多少
        showImg(Feature);

        printf("M: %d; N: %d \n", M, N);
        printf("M: %d; N: %d \n", mm, nn);
        int rate = (M * N) / (mm * nn);
        printf("压缩倍率为： %d \n", rate);
        for (i=0; i< mm; i++) {
            for (j = 0; j < nn; j++) {
                Feature.at<int>(i, j) = Feature.at<int>(i, j) / rate;
                if (Feature.at<int>(i, j) > 255) { Feature.at<int>(i, j) = 255; }
                printf("%d ", Feature.at<int>(i, j));
            }
            printf("\n");
        }
        printf("end\n");
        printf("图片的深度为： %d \n", Feature.depth());
        //cv::imshow("original", m);

        printf("第2步：简化色彩。当前灰度级为8位，即256色，将其处理为cc色。这里设置cc = 64");
        int cc = 64;
        for (i=0; i< mm; i++) {
            for (j = 0; j < nn; j++) {
                Feature.at<int>(i, j) /= (256/cc);
                printf("%d ", Feature.at<int>(i, j));
            }
            printf("\n");
        }
        printf("第3步：计算图像的平均值");
        double sum, aver;
        sum = 0;
        for (i=0; i< mm; i++) {
            for (j = 0; j < nn; j++) {
                sum += Feature.at<int>(i, j);
            }
            printf("\n");
        }
        aver = sum/(mm * nn);
        printf("图片的平均值为：%d \n", aver);

        printf("第4步：计算像素值与平均值大小关系。同时得到特征矩阵FeatureResult,并返回");
        int flag = 0;
        for (i=0; i< mm; i++) {
            for (j = 0; j < nn; j++) {
                if (Feature.at<int>(i, j) > aver) {
                    t[flag++] = 1;
                } else {
                    t[flag++] = 0;
                }
            }
            printf("\n");
        }
        for (i = 0; i < mm*nn; i++) {
            printf("%d", t[i]);
            printf("\n");
        }
    }

    void showImg(cv::Mat m) {
        int mm = m.rows;
        int nn = m.cols;
        int i,j;
        for (i=0; i< mm; i++) {
            for (j = 0; j < nn; j++) {
                printf("%d ", m.at<int>(i, j));
            }
            printf("\n");
        }
        printf("end\n");
    }

    void detectAndDraw( cv::Mat& img, cv::CascadeClassifier& cascade,
                        cv::CascadeClassifier& nestedCascade, double scale, bool tryflip )
    {
        double t = 0;
        std::vector<cv::Rect> faces, faces2;
        /* 定义七种颜色用于人脸标记 */
        const static cv::Scalar colors[] = {
                cv::Scalar(255,0,0),
                cv::Scalar(255,128,0),
                cv::Scalar(255,255,0),
                cv::Scalar(0,255,0),
                cv::Scalar(0,128,255),
                cv::Scalar(0,255,255),
                cv::Scalar(0,0,255),
                cv::Scalar(255,0,255)
        };

        cv::Mat gray, smallImg;

        /* 因为用的是类haar特征，所以都是基于灰度图像的，这里要转换成灰度图像 */
        cvtColor( img, gray, cv::COLOR_BGR2GRAY );

        /* 将图片缩小，加快检测速度 */
        double fx = 1 / scale;
        /* 将尺寸缩小到1/scale, 用线性插值 */
        resize( gray, smallImg, cv::Size(), fx, fx, cv::INTER_LINEAR );
        /* 直方图均衡 */
        equalizeHist( smallImg, smallImg );

        cv::imshow("smallImg1", smallImg);
        //cv::waitKey(0);

        /* 用来计算算法执行时间 */
        t = (double)cv::getTickCount();

        /*人脸检测
            smallImg：输入的原图
            faces	：表示检测到的人脸目标序列
            1.1		：每次图像尺寸减小的比例为1.1
            2		：每一个目标至少要被检测到3次才算是真的目标
            CV_HAAR_SCALE_IMAGE：表示不是缩放分类器来检测，而是缩放图像
            Size(30, 30) 目标的最大最小尺寸
        */
        cascade.detectMultiScale( smallImg, faces, 1.1, 2, cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30) );
        if( tryflip )
        {
            flip(smallImg, smallImg, 1);
            cascade.detectMultiScale( smallImg, faces2,1.1, 2, 0|cv::CASCADE_SCALE_IMAGE,cv::Size(30, 30) );
            for( std::vector<cv::Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r )
            {
                faces.push_back(cv::Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
            }
        }

        /* 相减为算法执行的时间 */
        t = (double)cv::getTickCount() - t;
        printf( "detection time = %g ms\n", t*1000 / cv::getTickFrequency());

        for ( size_t i = 0; i < faces.size(); i++ ){
            cv::Rect r = faces[i];
            cv::Mat smallImgROI;
            std::vector<cv::Rect> nestedObjects;
            cv::Point center;
            cv::Scalar color = colors[i%8];
            int radius;

            printf("检测到的人脸范围是：%d, %d, %d, %d \n", r.x, r.y, r.width, r.height);

            /* 人脸长宽比例，在0.75-1.3 间画圆，其他范围画矩形 */
            double aspect_ratio = (double)r.width/r.height;
           /* if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
            {
                *//*还原原来尺寸 计算圆心和圆半径 *//*
                center.x = cvRound((r.x + r.width*0.5)*scale);
                center.y = cvRound((r.y + r.height*0.5)*scale);
                radius = cvRound((r.width + r.height)*0.25*scale);
                *//* 画出人脸检测区域 画圆 *//*
                circle( img, center, radius, color, 3, 8, 0 );
            }
            else
            {
                *//* 画出检测区域，画矩形 *//*
                rectangle( img, cv::Point(cvRound(r.x*scale), cvRound(r.y*scale)),
                           cv::Point(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)), color, 3, 8, 0);
            }*/
            rectangle( img, cv::Point(cvRound(r.x*scale), cvRound(r.y*scale)),
                       cv::Point(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)), color, 3, 8, 0);

            /* 检测到人眼，在人脸上画出人眼 */
            if( nestedCascade.empty()){
                continue;
            }

            smallImgROI = smallImg( r );

            /* 人眼检测 */
            nestedCascade.detectMultiScale( smallImgROI, nestedObjects, 1.1, 2, cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30) );
            printf("识别出来的眼睛数量为：%d", nestedObjects.size());
            if (nestedObjects.size() != 2) {
                printf("眼部检测不合格");
                continue;
            }
            cv::Rect eye1 = nestedObjects[0];
            cv::Rect eye2 = nestedObjects[1];


            for ( size_t j = 0; j < nestedObjects.size(); j++ ){
                cv::Rect nr = nestedObjects[j];
                printf("检测到的人眼睛范围是：%d, %d, %d, %d \n", nr.x, nr.y, nr.width, nr.height);
                /*还原原来尺寸 计算圆心和圆半径 */
                center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
                center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
                radius = cvRound((nr.width + nr.height)*0.25*scale);
                /* 画出人眼检测区域 画圆*/
                circle( img, center, radius, color, 3, 8, 0 );

               rectangle(img, cv::Point(cvRound(nr.x*scale), cvRound(nr.y*scale)),
                          cv::Point(cvRound((nr.x + nr.width-1)*scale), cvRound((nr.y + nr.height-1)*scale)), color, 3, 8, 0);
            }
        }
        /* 显示图像 img */
        imshow( "result", img );
    }



    int faceCheck() {
        cv::Mat frame, image;
        bool tryflip;
        cv::CascadeClassifier cascade, nestedCascade;
        double scale = 1.3;

        /* 加载分类器 */
        if ( !nestedCascade.load(EYE_GLASSES_MODEL ) )
        {
            std::cerr << "WARNING: Could not load classifier cascade for nested objects" << std::endl;
        }
        if( !cascade.load(FACE_MODEL ) )
        {
            std::cerr << "ERROR: Could not load classifier cascade" << std::endl;
            return -1;
        }

        /* 加载图片 */
        image = cv::imread(DETECT_IMAGE, 1 );
        if(image.empty())
        {
            std::cout << "Couldn't read iamge" << DETECT_IMAGE  <<  std::endl;

        }

        std::cout << "Detecting face(s) in " << DETECT_IMAGE << std::endl;

        /* 检测人脸及眼睛并画出检测到的区域 */
        if( !image.empty() )
        {
            detectAndDraw( image, cascade, nestedCascade, scale, tryflip );
            cv::waitKey(0);
        }
        return 0;


    }

};


