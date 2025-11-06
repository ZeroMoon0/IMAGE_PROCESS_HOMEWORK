cmake_minimum_required(VERSION 3.10)

# 我们明确指定使用 CXX (C++) 语言
project(lab1_project CXX) 

# 3. 查找 OpenCV 包
# REQUIRED 确保如果找不到，配置过程会失败，并给出提示
find_package(OpenCV REQUIRED) 

# 4. 创建可执行文件
# lab1_exe 是最终生成的可执行文件的名称
add_executable(lab1_exe 1.cpp) 

# 5. 链接 OpenCV 库
# ${OpenCV_LIBS} 包含了所有 OpenCV 依赖库的链接器标志
target_link_libraries(lab1_exe ${OpenCV_LIBS})