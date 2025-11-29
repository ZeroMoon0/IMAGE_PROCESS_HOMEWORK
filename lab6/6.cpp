#include <iostream>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <queue>
#include <map>


using namespace cv;

const std::string IMAGE_PATH = std::string(IMAGE_DIR) +"/6.jpg"; //图像的相对路径

// 显示图像函数的封装
void showAndWait(const std::string& windowName, const Mat& image){
    if (!image.empty()) {
        imshow(windowName, image);
        waitKey(0);
        destroyAllWindows();
    }
}

class HuffmanNode{
public:
    int frequency; //频率
    int gray_value; //灰度值
    std::unique_ptr<HuffmanNode> left;
    std::unique_ptr<HuffmanNode> right;  // 左右子节点
    HuffmanNode(int gray, int freq): gray_value(gray), frequency(freq), left(nullptr), right(nullptr) {}
    ~HuffmanNode() = default;
};

//基于仿函数的比较器
struct CompareNode{
    bool operator()(const HuffmanNode* a,const HuffmanNode* b){
        return a->frequency > b->frequency;
    }
};

//构建哈夫曼树
std::unique_ptr<HuffmanNode> buildHuffmanTree(const std::map<int, int>& frequency){
    std::priority_queue<HuffmanNode*,std::vector<HuffmanNode*>,CompareNode> pq; //优先队列依赖于列表
    //初始化最小堆
    for(auto it=frequency.begin();it != frequency.end();++it){
        pq.push(new HuffmanNode(it->first,it->second));
    }
    //构建哈夫曼树
    while(pq.size() > 1){
        //得到两个最小的节点
        HuffmanNode* _left = pq.top(); 
        pq.pop();
        HuffmanNode* _right = pq.top(); 
        pq.pop();
        int merge_freq = _left->frequency + _right->frequency;
        std::unique_ptr<HuffmanNode> _parent = std::make_unique<HuffmanNode>(-1,merge_freq);
        _parent->left = std::unique_ptr<HuffmanNode>(_left);
        _parent->right = std::unique_ptr<HuffmanNode>(_right);
        pq.push(_parent.release());//将两个最小的节点拼接而成的新节点放入最小堆
    }
    if (pq.empty()) return nullptr;
    HuffmanNode* _root = pq.top(); 
    pq.pop();
    return std::unique_ptr<HuffmanNode>(_root);     //只剩一个节点，此时一定为哈夫曼树的根节点
}


// 递归生成哈夫曼编码表,c参数为哈夫曼树的根，编码，编码映射
void generateHuffmanCode(const HuffmanNode* root, std::string code, std::map<int, std::string>& codes){
    if (!root) return;
    if (root->gray_value != -1) {
        codes[root->gray_value] = code;
        return;
    }
    // 遍历左右子树
    generateHuffmanCode(root->left.get(), code + "0", codes); //左子树添0
    generateHuffmanCode(root->right.get(), code + "1", codes); //右子树添1
}

// 统计整个图像的频率
std::map<int,int> collectFrequency(const Mat& pray_image){
    std::map<int, int> frequency;
    const uchar* p = pray_image.ptr<uchar>(0);
    size_t total_pixels = pray_image.total();
    for (size_t i = 0; i < total_pixels; ++i) 
        frequency[static_cast<int>(p[i])]++;
    return frequency;
}


//对图像数据进行哈夫曼编码
void HuffamnEncode(const Mat& pray_image){
    int rows = pray_image.rows;
    int cols = pray_image.cols;

    long long original_size_bytes = (long long)rows * cols;
    long long original_size_bits = original_size_bytes * 8;
    std::cout << "--- 图像信息 ---" << std::endl;
    std::cout << "尺寸: " << rows << "x" << cols << std::endl;
    std::cout << "原始大小: " << original_size_bytes << " 字节 (" << original_size_bits << " 位)" << std::endl;

    // 统计整个图像的全局频率
    std::map<int, int> global_frequency = collectFrequency(pray_image);
    std::unique_ptr<HuffmanNode> root = buildHuffmanTree(global_frequency);
    std::map<int, std::string> global_codes;
    //生成哈夫曼编码
    generateHuffmanCode(root.get(), "", global_codes);
    //估算编码表的存储开销，假设编码表存储有 像素值，编码长度
    long long code_table_overhead_bits = 0;
    for (const auto& pair : global_codes) {
        code_table_overhead_bits += 8; // 存储像素值 (0-255)
        code_table_overhead_bits += 8; // 存储编码长度
    }

    //逐行编码
    long long encoded_bits = 0;

    for (int i = 0; i < rows; ++i) {
        const uchar* row_ptr = pray_image.ptr<uchar>(i);
        for (int j = 0; j < cols; ++j) {
            int pixel_value = static_cast<int>(row_ptr[j]);
            encoded_bits += global_codes[pixel_value].length();   //估算哈夫曼编码后的编码部分开销
        }
    }
    //压缩后的总开销
    long long total_compressed_bits = encoded_bits + code_table_overhead_bits;
    long long total_compressed_bytes = (long long)std::ceil(total_compressed_bits / 8.0);
    
    double compression_ratio = 0.0;
    if (total_compressed_bits > 0) 
        compression_ratio = (double)original_size_bits / total_compressed_bits; // 计算压缩比，压缩比 = 原始大小 / 压缩后总大小 (包含编码表)
    std::cout << "\n--- 哈夫曼编码压缩结果 ---" << std::endl;
    std::cout << "原始总位数: " << original_size_bits << std::endl;
    std::cout << "编码表开销(估算位数): " << code_table_overhead_bits << std::endl;
    std::cout << "编码位串总长度: " << encoded_bits << std::endl;
    std::cout << "压缩后总位数: " << total_compressed_bits << std::endl;
    std::cout << "压缩后总字节数: " << total_compressed_bytes << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "哈夫曼编码压缩比: " << compression_ratio << std::endl;
    std::cout << std::endl;
}

//LZW编码
const int CODE_BIT_WIDTH = 12; // LZW 码字的位数,码字的位宽决定了压缩后的数据量
const int MAX_CODE = (1 << CODE_BIT_WIDTH);

//LZW 编码函数
std::vector<int> lzw_encode_row_global(const cv::Mat& row, std::map<std::string, int>& dictionary, int& next_code){
    if (row.empty()) return {};
    std::vector<int> encoded_output;
    std::string P = "";
    const uchar* row_ptr = row.ptr<uchar>(0);
    for (int j = 0; j < row.cols; ++j) {
        char C = static_cast<char>(row_ptr[j]); // 获取当前像素值
        std::string new_sequence = P + C;
        // 检查字典是否包含 P + C (new_sequence)
        if (dictionary.count(new_sequence)) 
            P = new_sequence;  // P + C 存在，扩展当前序列 P
        else {
            encoded_output.push_back(dictionary[P]); // P + C 不存在，输出 P 的码字
            if(next_code < MAX_CODE)  
                dictionary[new_sequence] = next_code++; //将 P + C 加入字典 (如果字典未满)
            P = std::string(1, C);// 当前序列 P 重置为 C (新的单字符序列)
        }
    }
    if (!P.empty()) 
        encoded_output.push_back(dictionary[P]);
    return encoded_output;
}

void LZWEncode(const Mat& img) {
    int rows = img.rows;
    int cols = img.cols;
    
    long long original_size_bytes = (long long)rows * cols;
    long long original_size_bits = original_size_bytes * 8;

    std::map<std::string, int> global_dictionary;   //用字典来存储LZW所需要的字典
    for (int i = 0; i < 256; ++i)
        global_dictionary[std::string(1, (char)i)] = i; //初始化字典，将0-255注册到字典中
    int next_code = 256; 

    long long total_encoded_bits = 0;
    std::cout << "--- 图像信息 ---" << std::endl;
    std::cout << "尺寸: " << rows << "x" << cols << std::endl;
    std::cout << "原始大小: " << original_size_bytes << " 字节 (" << original_size_bits << " 位)" << std::endl;
    std::cout << "LZW 码字位宽: " << CODE_BIT_WIDTH << " 位" << std::endl;
    std::cout << "--- 全局字典初始化完成, 逐行编码 ---" << std::endl;

    // 按行进行 LZW 编码
    for (int i = 0; i < rows; ++i) {
        cv::Mat row = img.row(i); // 获取第 i 行
        std::vector<int> encoded_codes = lzw_encode_row_global(row, global_dictionary, next_code); //逐行进行LZW编码
        long long row_encoded_bits = encoded_codes.size()* CODE_BIT_WIDTH;
        total_encoded_bits += row_encoded_bits;
    }
    
    // 计算压缩比,LZW无需存储字典
    long long total_compressed_bytes = (long long)std::ceil(total_encoded_bits / 8.0);
    double compression_ratio = 0.0;
    if (total_encoded_bits > 0) 
        compression_ratio = (double)original_size_bits / total_encoded_bits;
    
    std::cout << "\n--- LZW编码压缩结果 ---" << std::endl;
    std::cout << "原始总位数: " << original_size_bits << std::endl;
    std::cout << "总编码位长: " << total_encoded_bits << std::endl;
    std::cout << "最终字典大小: " << global_dictionary.size() << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "LZW编码压缩比 " << compression_ratio << std::endl;
    std::cout << std::endl;
}


const int BLOCK_SIZE = 16; //定义块尺寸，要求为16*16

//对图像块进行DCT变换、量化压缩和IDCT反变换
void process_block(const Mat& src_image,Mat& dst_image){
    Mat dct_image; 
    dct(src_image, dct_image);  //进行dct变换

    std::vector<float> abs_coeffs;  //收集所有系数的绝对值
    abs_coeffs.reserve(BLOCK_SIZE * BLOCK_SIZE);
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        const float* row_ptr = dct_image.ptr<float>(i);
        for (int j = 0; j < BLOCK_SIZE; ++j)
            abs_coeffs.push_back(std::abs(row_ptr[j]));
    }
    std::sort(abs_coeffs.begin(), abs_coeffs.end(), [](float a, float b) {return a > b;});  //降序排列
    float threshold = abs_coeffs[BLOCK_SIZE*BLOCK_SIZE/2];  // 阈值

    for (int i = 0; i < BLOCK_SIZE; ++i) {
        float* row_ptr = dct_image.ptr<float>(i);
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            if (std::abs(row_ptr[j]) <= threshold && (i!=0 || j!=0)) {
                 row_ptr[j] = 0.0f;
            }
        }
    }

    idct(dct_image, dst_image);   //进行dct反变换
}

void dct_compression(const Mat& src_image, Mat& compressed_image){
    int rows = src_image.rows;
    int cols = src_image.cols;
    compressed_image.create(rows, cols, src_image.type());

    //遍历子图像
    for (int i = 0; i < rows; i += BLOCK_SIZE){
        for (int j = 0; j < cols; j += BLOCK_SIZE) {
            if (i + BLOCK_SIZE > rows || j + BLOCK_SIZE > cols)
                continue; //如果块不满足16*16大小，则跳过
            // 分割子图
            Rect block_rect(j, i, BLOCK_SIZE, BLOCK_SIZE);
            Mat block = src_image(block_rect);
            block.convertTo(block,CV_32FC1);
            subtract(block, Scalar(128.0f), block);

            //正变换,量化编码，反变换
            Mat reconstructed_block;
            process_block(block,reconstructed_block);

            //加回 128.0f 并转换为 CV_8UC1
            add(reconstructed_block, Scalar(128.0f), reconstructed_block);
            reconstructed_block.convertTo(compressed_image(block_rect), CV_8UC1);
        }
    }
}

int main(){
    Mat gray_image = imread(IMAGE_PATH, IMREAD_GRAYSCALE);
    if (gray_image.empty()) {
        std::cerr << "错误: 无法读取图像. 检查路径是否正确: " << IMAGE_PATH << std::endl;
        return -1;
    }
    HuffamnEncode(gray_image);
    LZWEncode(gray_image);

    //DCT变换
    Mat compressed_image;
    dct_compression(gray_image, compressed_image);

    showAndWait("Original Gray Image :",gray_image);
    showAndWait("DCT Compressed Gray Image :",compressed_image);

    HuffamnEncode(compressed_image);

    return 0;
}






