#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace cv;
using namespace std;

const string IMAGE_PATH = std::string(IMAGE_DIR) +"/4.jpg"; //图像的相对路径

// 显示图像函数的封装
void showAndWait(const string& windowName, const Mat& image){
    if (!image.empty()) {
        imshow(windowName, image);
        waitKey(0);
        destroyAllWindows();
    }
}

//添加高斯噪声
Mat addGuassNoise(const Mat& gray_image,double mu,double sigma){
    Mat noise;
    noise.create(gray_image.size(),CV_32FC1);
    randn(noise, mu, sigma); //在noise矩阵中生成符合高斯分布的随机数
    Mat noise_image;
    gray_image.convertTo(noise_image, CV_32FC1);
    for (int i = 0; i < noise_image.rows; ++i) {
        // 获取每一行的指针
        float* p_dst = noise_image.ptr<float>(i);  
        const float* p_noise = noise.ptr<float>(i);  
        for (int j = 0; j < gray_image.cols; ++j) 
            p_dst[j] += p_noise[j]; //添加噪声    
    }
    noise_image.convertTo(noise_image, CV_8UC1, 1.0, 0.0);
    return noise_image;
}

//添加椒盐噪声
Mat addSaltAndPepperNoise(const Mat& gray_image,double density){
    Mat noise_image = gray_image.clone();
    RNG rng(getTickCount());
    int amount = (int)(gray_image.total()*density); //总像素数
    for (int k = 0; k < amount; ++k) {
        int i = rng.uniform(0, gray_image.rows);  // 行坐标
        int j = rng.uniform(0, gray_image.cols);  // 列坐标
        // 获取当前行的指针
        uchar* p_noise_row = noise_image.ptr<uchar>(i);
        const uchar* p_gray_row = gray_image.ptr<uchar>(i);
        if (rng.uniform(0, 2) == 0) 
            p_noise_row[j] = 255;
        else
            p_noise_row[j] = 0;
    }
    return noise_image;
}

//添加椒噪声
Mat addPepperNoise(const Mat& gray_image,double density){
    Mat noise_image = gray_image.clone();
    RNG rng(getTickCount());
    int amount = (int)(gray_image.total()*density); //总像素数
    for (int k = 0; k < amount; ++k) {
        int i = rng.uniform(0, gray_image.rows);  // 行坐标
        int j = rng.uniform(0, gray_image.cols);  // 列坐标
        // 获取当前行的指针
        uchar* p_noise_row = noise_image.ptr<uchar>(i);
        const uchar* p_gray_row = gray_image.ptr<uchar>(i);
        p_noise_row[j] = 0; //椒噪声，黑色，0
    }
    return noise_image;
}

//添加盐噪声
Mat addSaltNoise(const Mat& gray_image,double density){
    Mat noise_image = gray_image.clone();
    RNG rng(getTickCount());
    int amount = (int)(gray_image.total()*density); //总像素数
    for (int k = 0; k < amount; ++k) {
        int i = rng.uniform(0, gray_image.rows);  // 行坐标
        int j = rng.uniform(0, gray_image.cols);  // 列坐标
        // 获取当前行的指针
        uchar* p_noise_row = noise_image.ptr<uchar>(i);
        const uchar* p_gray_row = gray_image.ptr<uchar>(i);
        p_noise_row[j] = 255; //盐噪声，白色，255
    }
    return noise_image;
}

// 为彩色图像添加高斯噪声
Mat addColorGaussianNoise(const Mat& color_image, double mu, double sigma) {
    CV_Assert(color_image.type() == CV_8UC3);
    vector<Mat> bgr_channels;
    split(color_image, bgr_channels); // 分离通道

    // 对每个通道应用灰度噪声函数
    for (int i = 0; i < 3; ++i) 
        bgr_channels[i] = addGuassNoise(bgr_channels[i], mu, sigma);

    Mat noisy_color_image;
    merge(bgr_channels, noisy_color_image); // 合并通道
    return noisy_color_image;
}


//算术均值滤波器
void arithmeticMeanFilter(const Mat& gray_image,Mat& filtered_image,int k){
    filtered_image.create(gray_image.size(),gray_image.type());
    int half = k / 2;
    int window_pixels = k * k; //矩形子图窗口中的像素个数

    Mat padded_image;//填充图像
    copyMakeBorder(gray_image,padded_image,half,half,half,half,BORDER_REFLECT);
    for (int i = 0; i < gray_image.rows; ++i) {
        uchar* p_dst = filtered_image.ptr<uchar>(i);  // 获取目标图像的行指针
        for (int j = 0; j < gray_image.cols; ++j) {
            //(i,j) 为待滤波图像的中心，(I,J)为(i,j)在填充图像padded_image中的位置
            int I = i + half;
            int J = j + half;
            double sum = 0.0; //窗口内的像素和
            //遍历矩形子图窗口
            for (int u = -half; u <= half; ++u) {
                const uchar* p_src_row = padded_image.ptr<uchar>(I + u); // 获取填充图像的局部行指针
                for (int v = -half; v <= half; ++v)
                    sum += p_src_row[J + v]; 
            }
            double arithmetic_mean = sum / window_pixels;   //算术均值 = (1/mn) * Sum
            p_dst[j] = saturate_cast<uchar>(arithmetic_mean);
        }
    }
}

//几何均值滤波器
void geometricMeanFilter(const Mat& gray_image, Mat& filtered_image, int k) {
    filtered_image.create(gray_image.size(), gray_image.type());
    int half = k / 2;
    int window_pixels = k * k; // 矩形子图窗口中的像素个数

    Mat padded_image; // 填充图像
    copyMakeBorder(gray_image, padded_image, half, half, half, half, BORDER_REFLECT);
    for (int i = 0; i < gray_image.rows; ++i) {
        uchar* p_dst = filtered_image.ptr<uchar>(i);  // 获取目标图像的行指针
        for (int j = 0; j < gray_image.cols; ++j) {
            // (i,j) 为待滤波图像的中心，(I,J)为(i,j)在填充图像padded_image中的位置
            int I = i + half;
            int J = j + half;
            double product_log = 0.0; // 像素积，使用对数变为加法，防止乘法产生溢出
            bool has_zero = false;
            //遍历矩形子图窗口
            for (int u = -half; u <= half; ++u) {
                const uchar* p_src_row = padded_image.ptr<uchar>(I + u); // 获取填充图像的局部行指针
                for (int v = -half; v <= half; ++v) {
                    uchar val = p_src_row[J + v]; 
                    if (val == 0) {
                        has_zero = true; //当有值为0时，则乘积为0，直接退出循环
                        break;
                    }
                    product_log += log((double)val); //乘法运算可以变成对数的加法运算
                }
                if (has_zero) break;
            }
            double geometric_mean; //几何均值
            if (has_zero)
                geometric_mean = 0.0;
            else 
                geometric_mean = exp(product_log / window_pixels);//几何均值 = exp((1/mn) * Sum(Log(value)))
            p_dst[j] = saturate_cast<uchar>(geometric_mean);
        }
    }
}

//谐波平均滤波器
void harmonicMeanFilter(const Mat& gray_image, Mat& filtered_image, int k) {
    filtered_image.create(gray_image.size(), gray_image.type());
    int half = k / 2;
    int window_pixels = k * k; // 矩形子图窗口中的像素个数

    Mat padded_image; // 填充图像
    copyMakeBorder(gray_image, padded_image, half, half, half, half, BORDER_REFLECT);
    for (int i = 0; i < gray_image.rows; ++i) {
        uchar* p_dst = filtered_image.ptr<uchar>(i);  // 获取目标图像的行指针
        for (int j = 0; j < gray_image.cols; ++j) {
            // (i,j) 为待滤波图像的中心，(I,J)为(i,j)在填充图像padded_image中的位置
            int I = i + half;
            int J = j + half;
            double sum_reciprocal = 0.0; // 表示像素的倒数之和
            bool has_zero = false; //判断像素是不是0，像素是0时，倒数是无穷大，此时倒数和为无穷大，谐波均值为0
            //遍历矩形子图窗口
            for (int u = -half; u <= half; ++u) {
                const uchar* p_src_row = padded_image.ptr<uchar>(I + u); // 获取填充图像的局部行指针
                for (int v = -half; v <= half; ++v) {
                    uchar val = p_src_row[J + v];
                    if (val == 0) {
                        has_zero = true; // 倒数无限大，结果为 0
                        break;
                    }
                    sum_reciprocal += 1.0 / val; //倒数求和
                }
                if (has_zero) break;
            }
            double harmonic_mean; //谐波均值
            if (has_zero || sum_reciprocal == 0.0) //当has_zero时，谐波均值为0；当sum_reciprocal为0时，谐波均值无穷达，无意义
                harmonic_mean = 0.0;
            else 
                harmonic_mean = (double)window_pixels / sum_reciprocal; // 谐波均值 = m*n / (倒数之和)
            p_dst[j] = saturate_cast<uchar>(harmonic_mean);
        }
    }
}

//反谐波平均滤波器，Q为阶数，Q为正，消除胡椒噪声，Q为负，消除盐噪声
void contraHarmonicMeanFilter(const Mat& gray_image, Mat& filtered_image, int k, double Q) {
    filtered_image.create(gray_image.size(), gray_image.type());
    int half = k / 2;
    
    Mat padded_image; // 填充图像
    copyMakeBorder(gray_image, padded_image, half, half, half, half, BORDER_REFLECT);
    for (int i = 0; i < gray_image.rows; ++i) {
        uchar* p_dst = filtered_image.ptr<uchar>(i);  // 获取目标图像的行指针
        for (int j = 0; j < gray_image.cols; ++j) {
            // (i,j) 为待滤波图像的中心，(I,J)为(i,j)在填充图像padded_image中的位置
            int I = i + half;
            int J = j + half;   
            double sum_numerator = 0.0;     // 表示分子，r^(Q+1) 之和
            double sum_denominator = 0.0;   // 表示分母，r^Q 之和
            //遍历矩形子图窗口
            for (int u = -half; u <= half; ++u) {
                const uchar* p_src_row = padded_image.ptr<uchar>(I + u); // 获取填充图像的局部行指针
                for (int v = -half; v <= half; ++v) {
                    double val = (double)p_src_row[J + v];
                    sum_numerator += pow(val, Q + 1);
                    sum_denominator += pow(val, Q);
                }
            }
            double contra_harmonic_mean; //反谐波均值
            if (sum_denominator != 0.0) //分母不为0，才有意义
                contra_harmonic_mean = sum_numerator / sum_denominator; // 反谐波均值 = sum(r^(Q+1)) / sum(r^Q)
            else 
                contra_harmonic_mean = 0.0; 
            p_dst[j] = saturate_cast<uchar>(contra_harmonic_mean);
        }
    }
}


//中值滤波器(统计排序滤波器)
void meanFilter(const Mat& gray_image, Mat& filtered_image, int k){
    filtered_image.create(gray_image.size(), gray_image.type());
    int half = k / 2;

    Mat padded_image; // 填充图像
    copyMakeBorder(gray_image, padded_image, half, half, half, half, BORDER_REFLECT);
    for (int i = 0; i < gray_image.rows; ++i) {
        uchar* p_dst = filtered_image.ptr<uchar>(i);  // 获取目标图像的行指针
        for (int j = 0; j < gray_image.cols; ++j) {
            // (i,j) 为待滤波图像的中心，(I,J)为(i,j)在填充图像padded_image中的位置
            int I = i + half;
            int J = j + half;
            vector<uchar> window_values; // 存储子图窗口内的像素值
            //遍历矩形子图窗口
            for (int u = -half; u <= half; ++u) {
                const uchar* p_src_row = padded_image.ptr<uchar>(I + u); // 获取填充图像的局部行指针
                for (int v = -half; v <= half; ++v)
                    window_values.push_back(p_src_row[J + v]); // 将窗口内的像素值存入vector
            }
            std::sort(window_values.begin(), window_values.end()); //对窗口内的像素值排序
            uchar median_value = window_values[window_values.size() / 2]; //取中值
            p_dst[j] = median_value; //图像位置变为窗口中值
        }
    }
}

//自适应均值滤波器
void adaptiveMeanFilter(const Mat& gray_image, Mat& filtered_image, int k) {
    filtered_image.create(gray_image.size(), gray_image.type());
    int half = k / 2;
    int window_pixels = k * k; 

    Mat padded_image; // 填充图像
    copyMakeBorder(gray_image, padded_image, half, half, half, half, BORDER_REFLECT);
    
    //计算噪声图像的方差
    double sum_gray = 0.0;
    double mean_gray = 0.0;
    double sigma_n2 = 0.0;
    for (int i = 0; i < gray_image.rows; ++i){
        const uchar* p_src = gray_image.ptr<uchar>(i);
        for (int j = 0; j < gray_image.cols; ++j)
            sum_gray += p_src[j];
    }
    mean_gray = sum_gray / gray_image.total(); //得到均值
    sum_gray = 0.0;
    for (int i = 0; i < gray_image.rows; ++i) {
        const uchar* p_src = gray_image.ptr<uchar>(i);
        for (int j = 0; j < gray_image.cols; ++j) {
            double diff = p_src[j] - mean_gray;  // 像素值与均值的差
            sum_gray += diff * diff;  // 累加平方差
        }
    }
    sigma_n2 = sum_gray / gray_image.total(); // 得到噪声方差

    for (int i = 0; i < gray_image.rows; ++i) {
        uchar* p_dst = filtered_image.ptr<uchar>(i); // 获取目标图像的行指针
        for (int j = 0; j < gray_image.cols; ++j) {
            int I = i + half;
            int J = j + half;
            
            double z_mean = 0.0; // 表示局部平均灰度z_mean
            vector<double> window_values; //向量存储窗口中的像素值
            //遍历子窗口
            for (int u = -half; u <= half; ++u) {
                const uchar* p_src_row = padded_image.ptr<uchar>(I + u);
                for (int v = -half; v <= half; ++v) {
                    double val = (double)p_src_row[J + v];
                    window_values.push_back(val);
                    z_mean += val;
                }
            }
            z_mean /= window_pixels; // 局部平均灰度

            double sigma_S2 = 0.0; //局部方差
            for (double val : window_values) {
                double diff = val - z_mean;
                sigma_S2 += diff*diff;
            }
            sigma_S2 /= window_pixels; // 局部方差

            // 使用自适应表达式
            double g_xy = (double)gray_image.ptr<uchar>(i)[j]; // 原始像素值 g(x,y)
            double result;
            if(sigma_n2 <= sigma_S2){
                double ratio = sigma_n2 / sigma_S2;
                result = g_xy - ratio * (g_xy - z_mean); //自适应表达式
            }else
                result = z_mean; // 违反条件时，比率设置为1，即g_xy-g_xy+z_mean,阻止产生无意义的结果
            p_dst[j] = saturate_cast<uchar>(result);
        }
    }
}

//自适应中值滤波器
void adaptiveMedianFilter(const Mat& gray_image, Mat& filtered_image, int Smax) {
    filtered_image.create(gray_image.size(), gray_image.type());
    int half = Smax / 2; 
    Mat padded_image;
    copyMakeBorder(gray_image, padded_image, half, half, half, half, BORDER_REFLECT);// 填充图像，使用 Smax 作为最大边界

    for (int i = 0; i < gray_image.rows; ++i) {
        uchar* p_dst = filtered_image.ptr<uchar>(i); // 获取目标图像的行指针
        for (int j = 0; j < gray_image.cols; ++j) {
            int I = i + half; // 中心行在填充图像中的位置
            int J = j + half; // 中心列在填充图像中的位置
            int k = 3; // 从 3x3 开始增加尺寸        
            // 层次A: 找到有效中值
            uchar z_med = 0;
            bool is_found = false;
            while (k <= Smax) {
                int new_half = k / 2;
                vector<uchar> window_values; // 当前窗口中的像素
                // 遍历当前子窗口
                for (int u = -new_half; u <= new_half; ++u) {
                    const uchar* p_src_row = padded_image.ptr<uchar>(I + u);
                    for (int v = -new_half; v <= new_half; ++v)
                        window_values.push_back(p_src_row[J + v]);
                }
                std::sort(window_values.begin(), window_values.end()); //当前窗口的像素排序
                uchar z_min = window_values.front(); 
                uchar z_max = window_values.back();
                z_med = window_values[window_values.size() / 2]; //窗口中值
                // 层次A判断是否满足z_min<z_med<z_max来跳转到层次B
                if (z_med> z_min && z_med < z_max) {
                    // 跳转到层次B
                    uchar z_xy = gray_image.ptr<uchar>(i)[j]; // 原始中心像素值z_xy
                    if (z_xy > z_min && z_xy < z_max) //层次B判断是否满足z_min<z_xy<z_max
                        p_dst[j] = z_xy; //输出z_xy
                    else 
                        p_dst[j] = z_med; //否则输出z_med
                    is_found = true;
                    break; // 处理下一个像素
                } else
                    k += 2; //不能跳转层次B，则增加窗口尺寸
            }
            if(!is_found)
                p_dst[j] = z_med; // 如果窗口增大到 Smax 仍未满足条件，则直接输出 Smax 窗口的中值
        }
    }
}

//彩色图像算术均值滤波
void colorArithmeticMeanFilter(const Mat& color_image, Mat& filtered_image, int k) {
    CV_Assert(color_image.type() == CV_8UC3);
    
    vector<Mat> bgr_channels;
    split(color_image, bgr_channels); // 分离通道
    
    Mat filtered_channels[3];
    for (int i = 0; i < 3; ++i)
        arithmeticMeanFilter(bgr_channels[i], filtered_channels[i], k); //对每个通道调用灰度算术均值滤波函数

    vector<Mat> result_channels = {filtered_channels[0], filtered_channels[1], filtered_channels[2]};
    merge(result_channels, filtered_image);  // 合并通道
}

//彩色图像几何均值滤波
void colorGeometricMeanFilter(const Mat& color_image, Mat& filtered_image, int k) {
    CV_Assert(color_image.type() == CV_8UC3);
    
    vector<Mat> bgr_channels;
    split(color_image, bgr_channels); // 分离通道
    
    Mat filtered_channels[3];
    for (int i = 0; i < 3; ++i)
        geometricMeanFilter(bgr_channels[i], filtered_channels[i], k);// 对每个通道调用灰度几何均值滤波函数
    
    vector<Mat> result_channels = {filtered_channels[0], filtered_channels[1], filtered_channels[2]};
    merge(result_channels, filtered_image); // 合并通道
}


int main(){
    Mat color_image = imread(IMAGE_PATH);
    if (color_image.empty()) {
        cerr << "错误: 无法读取图像. 检查路径是否正确: " << IMAGE_PATH << endl;
        return -1;
    }

    Mat gray_image;
    cvtColor(color_image, gray_image, COLOR_BGR2GRAY);

    // 原始图像
    showAndWait("Original Color Image", color_image);
    showAndWait("Original Grayscale Image", gray_image);

    // 生成含噪图像 
    Mat noise_gaussian = addGuassNoise(gray_image, 0, 20.0); //高斯噪声
    Mat noise_sp = addSaltAndPepperNoise(gray_image, 0.02); //椒盐噪声
    Mat noise_pepper = addPepperNoise(gray_image, 0.02);    //胡椒噪声
    Mat noise_salt = addSaltNoise(gray_image, 0.02);        //盐噪声
    // 显示含噪图像
    showAndWait("Noise: Gaussian", noise_gaussian);
    showAndWait("Noise: Salt & Pepper", noise_sp);
    showAndWait("Noise: Pepper", noise_pepper);
    showAndWait("Noise: Salt", noise_salt);

    // 1. 均值滤波
    Mat filter_result;
    arithmeticMeanFilter(noise_gaussian,filter_result,5); //算术均值
    showAndWait("Arithmetic Mean (Gaussian)",filter_result); 

    arithmeticMeanFilter(noise_sp,filter_result,5); //算术均值
    showAndWait("Arithmetic Mean (Salt&Pepper)",filter_result);

    geometricMeanFilter(noise_gaussian, filter_result, 5); //几何均值
    showAndWait("Geometric Mean (Gaussian)",filter_result);

    geometricMeanFilter(noise_sp, filter_result, 5); //几何均值
    showAndWait("Geometric Mean (Salt&Pepper)",filter_result);

    harmonicMeanFilter(noise_pepper, filter_result, 5);  //谐波均值
    showAndWait("Harmonic Mean (Pepper)", filter_result);

    harmonicMeanFilter(noise_salt, filter_result, 5);  //谐波均值
    showAndWait("Harmonic Mean (Salt)", filter_result);

    harmonicMeanFilter(noise_gaussian, filter_result, 5);  //谐波均值
    showAndWait("Harmonic Mean (Gaussian)", filter_result);

    contraHarmonicMeanFilter(noise_pepper, filter_result, 5, 2.0); //反谐波均值
    showAndWait("Contra-Harmonic Mean (Pepper, Q=2.0)", filter_result);

    contraHarmonicMeanFilter(noise_salt, filter_result, 5, -1.0); //反谐波均值
    showAndWait("Contra-Harmonic Mean (Salt, Q=-1.0)", filter_result);

    //  2.中值滤波
    meanFilter(noise_gaussian,filter_result,5);
    showAndWait("Median Filter 5x5 (Guassian)", filter_result);

    meanFilter(noise_sp, filter_result, 5); 
    showAndWait("Median Filter 5x5 (Salt&Pepper)", filter_result);

    meanFilter(noise_salt, filter_result, 5); 
    showAndWait("Median Filter 5x5 (Salt)", filter_result);

    meanFilter(noise_pepper, filter_result, 5); 
    showAndWait("Median Filter 5x5 (Pepper)", filter_result);

    meanFilter(noise_gaussian,filter_result,9);
    showAndWait("Median Filter 9x9 (Guassian)", filter_result);

    meanFilter(noise_sp, filter_result, 9); 
    showAndWait("Median Filter 9x9 (Salt&Pepper)", filter_result);

    meanFilter(noise_salt, filter_result, 9); 
    showAndWait("Median Filter 9x9 (Salt)", filter_result);

    meanFilter(noise_pepper, filter_result, 9); 
    showAndWait("Median Filter 9x9 (Pepper)", filter_result);

    //  3.自适应均值滤波
    adaptiveMeanFilter(noise_gaussian, filter_result, 7);
    showAndWait("Adaptive Mean Filter (Gaussian)", filter_result);

    arithmeticMeanFilter(noise_gaussian, filter_result, 7); 
    showAndWait("Compare: Arithmetic Mean 7x7", filter_result); //对比标准均值滤波

    // 4.自适应中值滤波
    adaptiveMedianFilter(noise_sp, filter_result, 7);
    showAndWait("Adaptive Median Filter (Salt&Pepper)", filter_result);

    meanFilter(noise_sp, filter_result, 7); 
    showAndWait("Compare: Median Filter 7x7 (Salt&Pepper)", filter_result);

    // 5.彩色图像算术均值滤波
    Mat noise_color_guassian = addColorGaussianNoise(color_image,0,20);
    showAndWait("Noise: Gaussian",noise_color_guassian);

    colorArithmeticMeanFilter(noise_color_guassian, filter_result, 5); 
    showAndWait("Color Arithmetic Mean (5x5)", filter_result);

    colorGeometricMeanFilter(noise_color_guassian, filter_result, 5);
    showAndWait("Color Geometric Mean (5x5)", filter_result);

    return 0;
}




