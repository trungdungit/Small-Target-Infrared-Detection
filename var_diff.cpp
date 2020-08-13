#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;


Mat mat2gray(const Mat& src){
    Mat dst;
    normalize(src, dst, 0.0, 1, NORM_MINMAX, CV_32FC1);
    return dst;
}

Mat my_stdfilt(const Mat& src, const Mat& nhood){
    Point anchor;
    anchor = Point(-1,-1);
    double delta = 0;

    Mat dst;
    float ss;
    ss = sum(nhood)[0];
    Mat src2;
    pow(src, 2, src2);
    float ss2;
    ss2 = ss - 1;
    float ss3;
    ss3 = 1.0f/ss2;
    Mat nhood2;
    nhood2 = nhood*ss3;
    Mat conv1;
    filter2D(src2, conv1, -1, nhood2, anchor, delta, BORDER_DEFAULT);

    float ss4;
    ss4=ss2*ss;
    float ss5=sqrt(ss4);
    float ss6=1.0f / ss5;
    Mat nhood3=nhood*ss6;
    Mat conv2;
    filter2D(src, conv2, -1, nhood3, anchor,delta, BORDER_DEFAULT);
    Mat conv3;
    pow(conv2,2,conv3);
    
    Mat temp1;
    temp1=conv1-conv3;
    
    Mat temp_thresh;
    threshold(temp1, temp_thresh , 0, 1 , THRESH_BINARY );
    temp_thresh.convertTo(temp_thresh, CV_32FC1);
    
    dst=temp_thresh.mul(temp1);
    
    
    return dst;
}

int main(){
    int64 t1 =0, t2 = 0;
    Mat image;
    Mat img;

    img = imread("/home/dungthu/PycharmProjects/ShipDetection/AntiShip/data/selected/good/1_375.png",0);
    img.convertTo(image, CV_32FC1);

    //LoG filter
    Point anchor;
    anchor = Point(-1,-1);

    double delta = 0;
    double time_vardiff;
    double time_temp = 0;

    for (int a=1; a<101; a=a+1){
        t1 = getCPUTickCount();
        int bnhood = 11;
        int tnhood3 = 7;
        int Nb=bnhood^2;
        int N3=tnhood3^2;
        int Ndiff3=Nb-N3;
        Mat mu3;
        Mat mub;

        blur(image, mu3, Size(tnhood3, tnhood3), Point(-1,-1), BORDER_DEFAULT);
        blur(image, mub, Size(bnhood, bnhood), Point(-1,-1), BORDER_DEFAULT);

        Mat mun3;
        Mat temp3;



        mun3=N3*mu3;

        mub=Nb*mub;


        temp3=(mub-mun3)/Ndiff3;
        
        Mat out3;
        out3=mu3-temp3;
        Mat temp_thresh;
        threshold(out3, temp_thresh , 0, 1 , THRESH_BINARY );
        temp_thresh.convertTo(temp_thresh, CV_32FC1);
        
        Mat P_rem;
        P_rem=temp_thresh.mul(out3);

        float mask_e[15][15]={
            {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
            {1,1,0,0,0,0,0,0,0,0,0,0,0,1,1},
            {1,1,0,0,0,0,0,0,0,0,0,0,0,1,1},
            {1,1,0,0,0,0,0,0,0,0,0,0,0,1,1},
            {1,1,0,0,0,0,0,0,0,0,0,0,0,1,1},
            {1,1,0,0,0,0,0,0,0,0,0,0,0,1,1},
            {1,1,0,0,0,0,0,0,0,0,0,0,0,1,1},
            {1,1,0,0,0,0,0,0,0,0,0,0,0,1,1},
            {1,1,0,0,0,0,0,0,0,0,0,0,0,1,1},
            {1,1,0,0,0,0,0,0,0,0,0,0,0,1,1},
            {1,1,0,0,0,0,0,0,0,0,0,0,0,1,1},
            {1,1,0,0,0,0,0,0,0,0,0,0,0,1,1},
            {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
        }; 

        Mat maske_kr1 = Mat(15,15,CV_32FC1, mask_e);
        float mask_i[7][7]={
            {1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1},
        }; 
       Mat maski_krl=Mat(7,7, CV_32FC1,mask_i);
       
       
       float mask_var[7][7]={
            {1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1},
            {1,1,1,0,1,1,1},
            {1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1},
            {1,1,1,1,1,1,1},
        }; 
       Mat maskvar_krl=Mat(7,7, CV_32FC1,mask_var);
       Mat V_i;
       V_i = my_stdfilt(image, maski_krl);

       Mat V_e;
       V_e = my_stdfilt(image, maske_kr1);

       float ss2;
       ss2 = sum(maskvar_krl)[0];

       float ss3;
       ss3 = 1.0f/ss2;

       maskvar_krl = maskvar_krl*ss3;
       
       Mat V2_e;
       filter2D(V_e, V2_e, -1, maskvar_krl, anchor,delta, BORDER_DEFAULT);
    
       Mat M_vard;
       M_vard=V_i-V2_e;
       
       Mat temp_thresh2;
       threshold(M_vard, temp_thresh2, 0,1, THRESH_BINARY);
       temp_thresh2.convertTo(temp_thresh2, CV_32FC1);
       M_vard = M_vard.mul(temp_thresh2);

       Mat out1;
       pow(M_vard, 2, out1);
       Mat out2;
        pow(P_rem,2,out2);
        out1=out1.mul(out2);
        Mat out;
        out=out1.mul(image);

       namedWindow("out");
       imshow("out", mat2gray(out));

        t2 = getCPUTickCount();
        time_vardiff = ((double)(t2 - t1)) / ((double)getTickFrequency());
        time_temp=time_temp+time_vardiff;
        printf("\n var_diff filtering time is: %f", time_vardiff);
    }
    printf("\n var_diff filtering time is: %f\n", time_temp);
    cv::waitKey(0);
    return EXIT_SUCCESS;
}