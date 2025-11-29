#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <queue>
#include <vector>

using namespace cv;
using namespace std;

const std::string IMAGE_PATH1 = std::string(IMAGE_DIR) +"/3.png"; //图像的相对路径
const std::string IMAGE_PATH2 = std::string(IMAGE_DIR) +"/7.png"; //图像的相对路径

// 显示图像函数的封装
void showAndWait(const std::string& windowName, const Mat& image){
    if (!image.empty()) {
        imshow(windowName, image);
        waitKey(0);
        destroyAllWindows();
    }
}

//跟踪以p为开始点的轮廓线直到该轮廓线的终点q
Mat iterative_edge_linking(const Mat& weak_and_strong, Mat& strong_edges) {
    queue<Point> q; 
    // 1. 扫描图像 2，将所有非零像素p入队 
    for (int i = 0; i < strong_edges.rows; ++i) {
        for (int j = 0; j < strong_edges.cols; ++j) {
            if (strong_edges.at<uchar>(i, j) == 255) {
                q.push(Point(j, i)); // Point(x, y)
            }
        }
    }
    static const int dx[8] = {-1, -1, -1, 0, 1, 1, 1, 0};
    static const int dy[8] = {-1, 0, 1, 1, 1, 0, -1, -1};
    // BFS 过程：以图像 2 为基础，向外扩展 (对应步骤 a 和 c 的重复)
    while (!q.empty()) {
        Point p = q.front();
        q.pop();
        // 考察 p' 的 8 邻域 
        for (int k = 0; k < 8; ++k) {
            int ni = p.y + dy[k];
            int nj = p.x + dx[k];
            if (ni >= 0 && ni < strong_edges.rows && nj >= 0 && nj < strong_edges.cols) { // 边界检查
                // 检查图像 1 中该邻域是否存在非零像素 q'且该像素尚未被包括到图像 2 中
                if (weak_and_strong.at<uchar>(ni, nj) == 255 && strong_edges.at<uchar>(ni, nj) == 0) {
                    strong_edges.at<uchar>(ni, nj) = 255;  // 将其包括到图像 2中，作为点 r 
                    q.push(Point(nj, ni)); // 从 r 开始重复第 a 步 ,入队继续跟踪轮廓线
                }
            }
        }
    }
    return strong_edges;
}

//自定义Canny实现，传入灰度图像，双阈值，返回边缘图像
Mat myCanny(const Mat& gray, float lowThreshold, float highThreshold){
    Mat blurred;
    // 1. 高斯滤波器，消除噪声
    GaussianBlur(gray, blurred, Size(5, 5), 1.4); //使用高斯核大小 5x5, sigma=1.4
    // 2. 计算横向与纵向两方向的微分
    Mat grad_x, grad_y;
    Sobel(blurred, grad_x, CV_32F, 1, 0, 3); // Sobel算子计算x方向梯度
    Sobel(blurred, grad_y, CV_32F, 0, 1, 3); // Sobel算子计算y方向梯度
    Mat magnitude(gray.rows, gray.cols, CV_32F);
    Mat angle(gray.rows, gray.cols, CV_32F);
    for (int i = 0; i < grad_x.rows; i++) {  
        for (int j = 0; j < grad_x.cols; j++) {  
        float gx = grad_x.at<float>(i, j);  // x方向梯度分量
        float gy = grad_y.at<float>(i, j);  // y方向梯度分量
        magnitude.at<float>(i, j) = sqrt(gx * gx + gy * gy);    // 公式：magnitude = sqrt(gx² + gy²)
        float ang_rad = atan2(gy, gx);  // 弧度制
        float ang_deg = ang_rad * (180.0 / CV_PI);  
        if (ang_deg < 0) 
            ang_deg += 360.0;  
        angle.at<float>(i, j) = ang_deg;    // 角度转换为度数制
        }
    }

    // 3. 非极大值抑制
    Mat nonMaxSuppressed = Mat::zeros(gray.size(), CV_32F);
    for (int i = 1; i < magnitude.rows - 1; i++){
        for(int j = 1; j < gray.cols - 1; ++j){
            float grad_mag = magnitude.at<float>(i, j);
            float grad_angle = angle.at<float>(i, j);
            float neighbor1 = 0.0f, neighbor2 = 0.0f;
            // 将角度量化到 4 个主要方向
            if ((grad_angle >= 0 && grad_angle < 22.5) || 
                (grad_angle >= 157.5 && grad_angle < 202.5) || (grad_angle >= 337.5 && grad_angle <= 360)){  //水平边缘
                    neighbor1 = magnitude.at<float>(i, j - 1);
                    neighbor2 = magnitude.at<float>(i, j + 1);
            }else if ((grad_angle >= 22.5 && grad_angle < 67.5) ||
                 (grad_angle >= 202.5 && grad_angle < 247.5)){
                    neighbor1 = magnitude.at<float>(i - 1, j + 1); // 右上
                    neighbor2 = magnitude.at<float>(i + 1, j - 1); // 左下  ]
            }else if ((grad_angle >= 67.5 && grad_angle < 112.5) ||
                 (grad_angle >= 247.5 && grad_angle < 292.5)){ //垂直边缘
                    neighbor1 = magnitude.at<float>(i - 1, j);
                    neighbor2 = magnitude.at<float>(i + 1, j);
            }else if ((grad_angle >= 112.5 && grad_angle < 157.5) ||
                 (grad_angle >= 292.5 && grad_angle < 337.5)){
                    neighbor1 = magnitude.at<float>(i - 1, j - 1); // 左上
                    neighbor2 = magnitude.at<float>(i + 1, j + 1); // 右下
            }
            // 非极大值抑制
            if (grad_mag >= neighbor1 && grad_mag >= neighbor2)
                nonMaxSuppressed.at<float>(i, j) = grad_mag;
            else
                nonMaxSuppressed.at<float>(i, j) = 0;
        }
    }

    // 4. 双阈值处理
    Mat weak_and_strong = Mat::zeros(gray.size(), CV_8U); //图像1
    Mat strong_edges = Mat::zeros(gray.size(), CV_8U);    //图像2
    for (int i = 0; i < nonMaxSuppressed.rows; i++){
        for(int j = 0; j < nonMaxSuppressed.cols; ++j){
            float pixel_value = nonMaxSuppressed.at<float>(i, j); 
            if (pixel_value >= highThreshold) { // T2
                // 像素同时出现在图像 1 和图像 2
                strong_edges.at<uchar>(i, j) = 255;
                weak_and_strong.at<uchar>(i, j) = 255;
            } else if (pixel_value >= lowThreshold)  // T1
                weak_and_strong.at<uchar>(i, j) = 255;// 像素只出现在图像 1 (弱边缘)
        }
    }

    // 5. 边缘连接
    Mat final_edges = iterative_edge_linking(weak_and_strong, strong_edges);
    return final_edges;
}

// 全局阈值分割
Mat totalThreshold(const Mat& gray, int& iterations, double& finalThreshold) {
    // 1.选择一个 T 的初始估计值即图像平均灰度
    double T_old = mean(gray)[0]; 
    double T_new = 0;
    iterations = 0; //返回的迭代次数
    const double TO = 0.1; //事先定义的参数T0
    long long sum_G1, count_G1, sum_G2, count_G2; //用于计算平均灰度
    while (true) {
        iterations++; 
        // 2. 用T分割图像，生成两组像素G1和G2
        sum_G1 = 0; count_G1 = 0; // G1: 灰度值 > T 的像素
        sum_G2 = 0; count_G2 = 0; // G2: 灰度值 <= T 的像素
        for (int i = 0; i < gray.rows; ++i) {
            const uchar* p = gray.ptr<uchar>(i);
            for (int j = 0; j < gray.cols; ++j) {
                uchar pixel_value = p[j];
                if (pixel_value > T_old) {
                    sum_G1 += pixel_value; 
                    count_G1++;            
                } else { // 灰度值 <= T_old
                    sum_G2 += pixel_value; 
                    count_G2++;            
                }
            }
        }
        // 3. 计算平均灰度值mu1和 mu2 
        double mu1 = (count_G1 > 0) ? (double)sum_G1 / count_G1 : 0; 
        double mu2 = (count_G2 > 0) ? (double)sum_G2 / count_G2 : 0; 
        // 4. 计算新的阈值 T = 1/2 * (mu1 + mu2)
        T_new = (mu1 + mu2) / 2.0;
        // 5. 重复步骤 2 到 4，直到逐次迭代得到的T值之差小于T0
        if (abs(T_old - T_new) < TO) {
            break;
        }
        T_old = T_new;
        if (iterations > 1000) { 
             cerr << "警告: 迭代次数过多，提前终止。" << endl;
             break; 
        }
    }
    finalThreshold = T_new; // 记录最终分割阈值
    Mat binary_img;
    threshold(gray, binary_img, finalThreshold, 255, THRESH_BINARY); // 应用最终阈值进行二值化
    return binary_img;
}

// 归一化直方图
vector<float> NormalizedHistogram(const Mat& gray, int N){
    const int histSize = 256;
    vector<int> hist(histSize, 0);
    for (int i = 0; i < gray.rows; ++i) {
        const uchar* p = gray.ptr<uchar>(i);
        for (int j = 0; j < gray.cols; ++j) {
            uchar pixel_value = p[j];
            hist[pixel_value]++; 
        }
    }
    vector<float> P_k_vector(histSize);
    for (int k = 0; k < histSize; ++k) 
        P_k_vector[k] = (float)hist[k] / N;
    return P_k_vector;
}


//Ostu算法分割
Mat OtsuThreshold(const Mat& gray, int& iterations, double& finalThreshold, double& separabilityMeasure) {
    // 1. 计算归一化直方图 P(k)
    int N = gray.rows * gray.cols;
    int histSize = 256;
    vector<float> P_k = NormalizedHistogram(gray, N);  
    // 准备数组存储累积值
    float P1[256] = {0}; // P_1(k)
    float m_k[256] = {0}; // m(k)
    // 预计算 P_1(k) 和 m(k)
    float P_sum = 0;
    float m_sum = 0;
    for (int k = 0; k < 256; ++k) {
        // 2. 计算累计和 P_1(k)
        P_sum += P_k[k];
        P1[k] = P_sum;
        // 3. 计算累积均值 m(k)
        m_sum += k * P_k[k];
        m_k[k] = m_sum;
    }
    // 4. 计算全局灰度均值 mG
    float mG = m_k[histSize - 1];
    // 寻找最佳阈值 k*
    float max_sigma_b_squared = -1.0f;
    vector<int> best_k_values; // 存储所有最大值对应的 k
    iterations = 0; // 记录遍历次数

    for (int k = 0; k < 256; ++k) {
        iterations++; 
        float P_C1 = P1[k];          // 前景概率 P_1(k)
        float P_C2 = 1.0f - P_C1;    // 背景概率 P_2(k)
        if (P_C1 < 1e-6 || P_C2 < 1e-6) continue; // 避免除以零和边界条件
        // 5. 计算类间方差 sigma^2_B(k)
        float numerator = pow(mG * P_C1 - m_k[k], 2);
        float denominator = P_C1 * P_C2;
        float sigma_b_squared = numerator / denominator; // (mG * P_C1 - m_k)^2 / (P_C1 * P_C2)
        // 6. 得到 Otsu 阈值 k_star
        if (sigma_b_squared > max_sigma_b_squared) {
            max_sigma_b_squared = sigma_b_squared;
            best_k_values.clear();
            best_k_values.push_back(k);
        } else if (sigma_b_squared == max_sigma_b_squared) {
             best_k_values.push_back(k); // 极大值不唯一
        }
    }
    // 极大值不唯一时，取平均值
    if (!best_k_values.empty()) {
        int sum_k = 0;
        for(int k : best_k_values) sum_k += k;
        finalThreshold = (double)sum_k / best_k_values.size(); 
    } else {
        finalThreshold = 0; // 默认值
    }

    // 7. 计算全局方差 sigma_G^2，然后计算可分离性测度 (Eta)
    float sigma_G_squared = 0;
    for (int k = 0; k < 256; ++k) 
        sigma_G_squared += pow(k - mG, 2) * P_k[k];
    if (sigma_G_squared > 1e-6) {
        separabilityMeasure = max_sigma_b_squared / sigma_G_squared; //可分离性测度 η = sigma^2_B(k*) / sigma^2_G
    } else {
        separabilityMeasure = 0;
    }
    Mat binary_img;
    threshold(gray, binary_img, finalThreshold, 255, THRESH_BINARY); // 二值化处理

    return binary_img;
}


int main(){
    Mat gray_image = imread(IMAGE_PATH1, IMREAD_GRAYSCALE);
    if (gray_image.empty()) {
        std::cerr << "错误: 无法读取图像. 检查路径是否正确: " << IMAGE_PATH1 << std::endl;
        return -1;
    }
    showAndWait("Gray Image", gray_image);
    Mat edges = myCanny(gray_image, 40, 80);
    showAndWait("My Canny Edges", edges);


    Mat gray_image2 = imread(IMAGE_PATH2, IMREAD_GRAYSCALE);
    if (gray_image2.empty()) {
        std::cerr << "错误: 无法读取图像. 检查路径是否正确: " << IMAGE_PATH2 << std::endl;
        return -1;
    }
    showAndWait("Gray Image", gray_image2);
    int iterations = 0;
    double finalThreshold = 0.0;
    Mat total = totalThreshold(gray_image2, iterations, finalThreshold);
    showAndWait("Total Thresholding", total);
    cout << "Total Thresholding迭代次数: " << iterations << ", 最终阈值: " << finalThreshold << endl;

    double separabilityMeasure = 0.0;
    Mat otsu = OtsuThreshold(gray_image2, iterations, finalThreshold, separabilityMeasure);
    showAndWait("Otsu Thresholding", otsu);
    cout << "Otsu Thresholding迭代次数: " << iterations << ", 最终阈值: " << finalThreshold 
         << ", 可分离性测度: " << separabilityMeasure << endl;

    return 0;
}



