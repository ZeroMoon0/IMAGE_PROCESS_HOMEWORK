#include <iostream>
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

// 得到频谱图，传入DFT后的复数矩阵
Mat getMagnitudeSpectrum(const Mat& complexI){
    Mat planes[2];
    split(complexI, planes); // planes[0]=实部, planes[1]=虚部
    magnitude(planes[0], planes[1], planes[0]);
    Mat magI = planes[0];  //MagI = sqrt(Re² + Im²)
    magI += Scalar::all(1); 
    log(magI, magI); //log(1 + Mag)可以压缩动态范围
    normalize(magI, magI, 0, 255, NORM_MINMAX); //归一化到(0，255)
    Mat mag_8U;
    magI.convertTo(mag_8U, CV_8U);
    return mag_8U;
}

//DFT 和 IDFT
void DFTAndIDFT(const Mat& gray_image){
    const int M = gray_image.rows; 
    const int N = gray_image.cols;
    // 1.填充原图尺寸为P=2M,Q=2N
    const int P = M * 2;
    const int Q = N * 2;
    Mat fp;
    Mat temp;
    copyMakeBorder(gray_image, temp, 0, P - M, 0, Q - N, BORDER_CONSTANT, Scalar::all(0));
    temp.convertTo(fp, CV_32F); 
    // 2. 将fp(x,y)乘以(-1)^(x+y)，使傅里叶变换位于P×Q大小的频率矩形的中心
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < Q; j++) {
            if ((i + j) % 2 != 0) 
                fp.at<float>(i, j) = -fp.at<float>(i, j);
        }
    }
    // 3 计算 DFT: F (P x Q, CV_32FC2)
    Mat real_image[2]; 
    real_image[0] = fp; 
    real_image[1] = Mat::zeros(fp.size(), CV_32F); 
    Mat F;
    merge(real_image, 2, F);
    dft(F, F);
    // 4 分离和显示频谱
    split(F, real_image); 
    Mat complex_dft_copy;
    merge(real_image, 2, complex_dft_copy); // 使用复数来绘制频谱
    Mat mag_spectrum = getMagnitudeSpectrum(complex_dft_copy);
    showAndWait("DFT Magnitude Spectrum", mag_spectrum);
    // 5 IDFT 验证
    Mat G;
    merge(real_image, 2, G); // G 是 P x Q
    dft(G, G, DFT_INVERSE | DFT_SCALE); 
    split(G, real_image);
    // 6. 反中心化和裁剪到原始 M x N 尺寸
    Mat idft_image(gray_image.rows, gray_image.cols, CV_32F); 
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float val = real_image[0].at<float>(i, j); 
            if ((i + j) % 2 != 0)
                idft_image.at<float>(i, j) = -val; //反中心化 f'(x,y) * (-1)^(x+y) = f(x,y)
            else
                idft_image.at<float>(i, j) = val;
        }
    }
    // 7. 归一化并显示
    normalize(idft_image, idft_image, 0, 255, NORM_MINMAX);
    idft_image.convertTo(idft_image, CV_8U);
    showAndWait("IDFT Grayscale Image (Verification)", idft_image);
}

// 过滤器类型的枚举
enum FilterType {
    ILPF = 1,  // 理想低通
    IHPF = 2,  // 理想高通
    BLPF = 3, // 布特沃斯低通
    BHPF = 4  // 布特沃斯高通
};

// 传递函数，其中D0为截止频率，n为阶数
Mat TransferFunc(int P, int Q,FilterType type,double D0,int n){
    Mat H(P, Q, CV_32F);// 传递函数大小P×Q,中心在(P/2,Q/2)
    int centerU = Q / 2; // 对应 u 轴
    int centerV = P / 2; // 对应 v 轴
    for (int v = 0; v < P; ++v) {
        for (int u = 0; u < Q; ++u) {
            double du = (double)u - centerU; 
            double dv = (double)v - centerV; 
            double D_uv = sqrt(du * du + dv * dv);//计算距离 D(u, v)

            double H_val = 0.0; //滤波器值对应点的值
            switch (type) {
                case ILPF: // 理想低通
                    H_val = (D_uv <= D0) ? 1.0 : 0.0; //  H = 1 if D <= D0, else 0
                    break;
                case IHPF: // 理想高通
                    H_val = (D_uv > D0) ? 1.0 : 0.0; // H = 1 if D > D0, else 0
                    break;
                case BLPF: // 布特沃斯低通
                    H_val = 1.0 / (1.0 + pow(D_uv / D0, 2 * n)); //  H = 1 / (1 + (D / D0)^2n)
                    break;
                case BHPF: // 布特沃斯高通
                    if (D_uv < 1e-6) { 
                        H_val = 0.0;
                    } else {
                        H_val = 1.0 / (1.0 + pow(D0 / D_uv, 2 * n)); // H = 1 / (1 + (D0 / D)^2n)
                    }
                    break;
                default:
                    H_val = 1.0; // 全通 
                    break;
            }
            H.at<float>(v, u) = (float)H_val;
        }
    }
    return H;
}

void frequency_filter(const Mat& gray_image,Mat& filter_image,const FilterType type,const double D0,const int n){
    const Mat& f = gray_image;
    const int M = f.rows; //原图的尺寸
    const int N = f.cols;
    // 1.填充原图尺寸为P=2M,Q=2N
    const int P = f.rows * 2;   // 填充后的尺寸
    const int Q = f.cols * 2;
    // 2. 使用0填充，形成大小为P×Q的填充后图像
    Mat fp;
    Mat temp; //临时存储uchar
    copyMakeBorder(f, temp, 0, P - M, 0, Q - N, BORDER_REFLECT);
    temp.convertTo(fp, CV_32F); //fp存储类型是CV_32F
    // 3. 将fp(x,y)乘以(-1)^(x+y)，使傅里叶变换位于P×Q大小的频率矩形的中心
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < Q; j++) {
            if ((i + j) % 2 != 0) 
                fp.at<float>(i, j) = -fp.at<float>(i, j); // 必须是 float 类型
        }
    }
    // 4. 计算步骤3得到的图像的DFT,即F(u,v)
    Mat real_imag[2];  //用来存储DFT后的结果
    real_imag[0] = fp; // 实部
    real_imag[1] = Mat::zeros(fp.size(), CV_32F); // 虚部

    Mat F;
    merge(real_imag, 2, F);
    dft(F, F);
    split(F, real_imag); // 重新分离F的实部[0]和虚部[1]

    // 5. 构建滤波器传递函数
    Mat H = TransferFunc(P, Q, type, D0, n);

    // 6. 采用对应像素相乘得到G,(G = F×H)
    for (int i = 0; i < P; i++) {
        for (int j = 0; j < Q; j++) {
            real_imag[0].at<float>(i, j) *= H.at<float>(i, j); // 实部
            real_imag[1].at<float>(i, j) *= H.at<float>(i, j); // 虚部
        }
    }

    // 7. 计算G的IDFT
    Mat G;
    merge(real_imag, 2, G);
    dft(G, G, DFT_INVERSE | DFT_SCALE); // 使用DFT_SCALE来正确归一化IDFT结果
    split(G, real_imag);

    // 8. 提取G的左上角部分的实部，并乘以(-1)^(x+y)进行反中心化
    Mat g_out(M, N, CV_32F);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float val = real_imag[0].at<float>(i, j); // 提取实部，并乘以(-1)^(i+j)进行反中心化
            if ((i + j) % 2 != 0)
                g_out.at<float>(i, j) = -val; // 奇数，取负
            else
                g_out.at<float>(i, j) = val; // 偶数，取正
        }
    }
    normalize(g_out, g_out, 0, 1, NORM_MINMAX);
    g_out.convertTo(filter_image, CV_8U, 255); //转换为8U
}


int main(){
    Mat gray_image = imread(IMAGE_PATH, IMREAD_GRAYSCALE);
    if (gray_image.empty()) {
        cerr << "错误: 无法读取图像. 检查路径是否正确: " << IMAGE_PATH << endl;
        return -1;
    }

    showAndWait("Original Grayscale Image", gray_image);

    DFTAndIDFT(gray_image);

    // 2. 理想高通和理想低通滤波
    Mat filter_image;
    frequency_filter(gray_image,filter_image,ILPF,50,0);
    showAndWait("ILPF Filter Image (D0=50)",filter_image);

    frequency_filter(gray_image,filter_image,IHPF,50,0);
    showAndWait("IHPF Filter Image (D0=50)",filter_image);

    // 3.布特沃斯低通和高通滤波
    frequency_filter(gray_image,filter_image,BLPF,100,5);
    showAndWait("BLPF Filter Image (D0=100, n=5)",filter_image);

    frequency_filter(gray_image,filter_image,BHPF,100,5);
    showAndWait("BHPF Filter Image (D0=100, n=5)",filter_image);

    return 0;

}



