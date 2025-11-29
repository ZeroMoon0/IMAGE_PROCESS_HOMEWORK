#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;
using namespace std;

const string IMAGE_PATH = std::string(IMAGE_DIR) +"/1.jpg"; //图像的相对路径

// 显示图像函数的封装
void showAndWait(const string& windowName, const Mat& image){
    if (!image.empty()) {
        imshow(windowName, image);
        waitKey(0);
        destroyAllWindows();
    }
}

// 彩色图像变为灰度图像 Gray = 0.299*R + 0.587*G + 0.114*B
void colorToGray(const Mat& image,Mat& gray_image){
    CV_Assert(image.type() == CV_8UC3); //检查图像的类型是否为 CV_8UC3
    gray_image.create(image.size(), CV_8UC1); //创建单通道的灰度图像
    for (int i = 0; i < image.rows; ++i) {
        const Vec3b* p_src = image.ptr<Vec3b>(i); //彩色图像的三通道指针
        uchar* p_dst = gray_image.ptr<uchar>(i); //灰度图像的单通道指针
        for (int j = 0; j < image.cols; ++j) {
            uchar B = p_src[j][0]; //0 代表 Blue
            uchar G = p_src[j][1]; //1 代表 Green
            uchar R = p_src[j][2]; //2 代表 Red
            
            p_dst[j] = saturate_cast<uchar>(0.299 * R + 0.587 * G + 0.114 * B); // 计算灰度值
        }
    }
}

// 灰度图像二值化处理
void binaryThresold(const Mat& gray_image,Mat& binary_image,int threshold){
    binary_image.create(gray_image.size(), gray_image.type()); //创建二值化图像
    for (int i = 0; i < gray_image.rows; ++i) {
        const uchar* p_src = gray_image.ptr<uchar>(i);
        uchar* p_dst = binary_image.ptr<uchar>(i);
        
        for (int j = 0; j < gray_image.cols; ++j) {
            uchar pixel = p_src[j];
            // 判断像素与阈值的大小关系，更大即为255，更小即为0
            if (pixel > threshold)
                p_dst[j] = 255;
            else
                p_dst[j] = 0;
        }
    }
}


//灰度图像的对数变换
void logTransform(const Mat& gray_image, Mat& log_image,float c){
    Mat src_float;
    gray_image.convertTo(src_float, CV_32FC1); // 由于进行对数变换，故转换为 float 类型

    log_image.create(src_float.size(), src_float.type()); //创建图像
    for (int i = 0; i < src_float.rows; ++i) {
        const float* p_src = src_float.ptr<float>(i);
        float* p_dst = log_image.ptr<float>(i);

        for (int j = 0; j < src_float.cols; ++j) {
            float r = p_src[j];
            p_dst[j] = c * log(1 + r); // 对数变换公式，s = c * log(1 + r)，r是像素
        }
    }
    log_image.convertTo(log_image, CV_8UC1); //确保最终在0-255范围
}

//灰度图像的伽马变换
void gammaTransform(const Mat& gray_image, Mat& gamma_image,float gamma){
    Mat src_float;
    gray_image.convertTo(src_float, CV_32FC1, 1.0 / 255.0); // 归一化到 [0, 1]
    gamma_image.create(src_float.size(), src_float.type()); //创建图像
    for (int i = 0; i < src_float.rows; ++i) {
        const float* p_src = src_float.ptr<float>(i);
        float* p_dst = gamma_image.ptr<float>(i);
        for (int j = 0; j < src_float.cols; ++j) {
            float r = p_src[j];
            p_dst[j] = pow(r, gamma); // s = c * r^gamma, c=1
        }
    }
    gamma_image.convertTo(gamma_image, CV_8UC1, 255.0); //转换回 8-bit unsigned char，并乘以 255 恢复范围
}

//彩色图像的补色变换
void colorComplement(const Mat& image, Mat& complement_image){
    CV_Assert(image.type() == CV_8UC3); 
    complement_image.create(image.size(), image.type()); //创建补色图像
    for (int i = 0; i < image.rows; ++i) {
        const Vec3b* p_src = image.ptr<Vec3b>(i);
        Vec3b* p_dst = complement_image.ptr<Vec3b>(i);
        for (int j = 0; j < image.cols; ++j) {
            // 对 B, G, R 三个通道分别进行补色操作,公式为 255 - pixel 
            p_dst[j][0] = 255 - p_src[j][0]; // B
            p_dst[j][1] = 255 - p_src[j][1]; // G
            p_dst[j][2] = 255 - p_src[j][2]; // R
        }
    }
}


int main() {
    // 1. 读取图像
    Mat color_image = imread(IMAGE_PATH);
    if (color_image.empty()) {
        cerr << "错误: 无法读取图像. 请检查路径是否正确: " << IMAGE_PATH << endl;
        return -1;
    }
    // 显示原始彩色图像
    showAndWait("0. Original Color Image", color_image);

    Mat gray_image;
    colorToGray(color_image, gray_image); //得到灰度图像
    showAndWait("1. Grayscale Image", gray_image);

    //2. 灰度图像二值化处理
    Mat binary_img;
    binaryThresold(gray_image, binary_img, 127); 
    showAndWait("2. Binary Threshold (T=127)", binary_img);

    // 3. 灰度图像的对数变换
    Mat log_img_1;
    float c1 = 20.0; // 较小的 c
    logTransform(gray_image, log_img_1, c1);
    showAndWait("3.1 Log Transform (c=20.0)", log_img_1);

    Mat log_img_2;
    float c2 = 255.0 / log(1 + 255.0); // 标准归一化 c
    logTransform(gray_image, log_img_2, c2);
    showAndWait("3.2 Log Transform (c=Standard)", log_img_2);

    Mat log_img_3;
    float c3 = 80.0; // 较大的 c
    logTransform(gray_image, log_img_3, c3);
    showAndWait("3.3 Log Transform (c=60.0)", log_img_3);

    // 4. 灰度图像的伽马变换
    Mat gamma_img_1;
    float gamma1 = 0.4; 
    gammaTransform(gray_image, gamma_img_1, gamma1); 
    showAndWait("4.1 Gamma Transform Gamma=0.4", gamma_img_1);

    Mat gamma_img_2;
    float gamma2 = 2.5; 
    gammaTransform(gray_image, gamma_img_2, gamma2); 
    showAndWait("4.2 Gamma Transform Gamma=2.5", gamma_img_2);

    Mat gamma_img_3;
    float gamma3 = 1.2; 
    gammaTransform(gray_image, gamma_img_3, gamma3); 
    showAndWait("4.3 Gamma Transform Gamma=1.2", gamma_img_3);

    // 5. 彩色图像的补色变换
    Mat complement_img;
    colorComplement(color_image, complement_img);
    showAndWait("5. Color Complement ", complement_img);


    
    return 0;
}