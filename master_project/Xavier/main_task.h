#ifndef MAIN_TASK_H
#define MAIN_TASK_H
#include <deque>
#include<iostream>
#include <unistd.h>

/******************opencv*******************/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>
#include <sstream>
#include <time.h>

#include <thread>
#include<mutex>
#include<condition_variable>


/******************Others*******************/
#include <assert.h>
#include <cmath>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <unordered_map>
#include <algorithm>
#include <float.h>
#include <string.h>
#include <chrono>
#include <iterator>
/****************** tensor RT*******************/
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"
#include "NvOnnxParser.h"
#include "BatchStream.h"
using namespace nvinfer1;
using namespace nvcaffeparser1;


using namespace std;
using namespace cv;

class main_task
{
public:
    main_task();
    ~main_task();
    /******************任务函数 *******************/

    int task_1(ifstream *input,deque<Mat> &output);
    int task_2(deque<Mat> &input,deque<float *> &output);
    int task_3(deque<float *> &input,deque<float *> &output);
    int task_4(deque<float *> &input,deque<Mat> &output_img,deque<Mat> &output_label);
    int task_5(deque<Mat> &input_img,deque<Mat> &input_label,vector<Mat> &output);

    int task_show(deque<Mat> &input_img,deque<Mat> &input_label,vector<Mat> &intput_bin);


    /****************** 应用函数*******************/

    ifstream& open_file(ifstream &in,const string &file);
    cv::Mat map2threeunchar(cv::Mat real_out,cv::Mat real_out_);
    cv::Mat read2mat(float * prob,cv::Mat out);
    float* normal(cv::Mat &img);
    float doInference(IExecutionContext& context, float* input, float* output, int batchSize);
    int detect_object(cv::Mat &img,cv::Mat &labels,std::vector<cv::Mat> &result);
    bool onnxToTRTModel(const std::string& modelFile, // name of the onnx model
                        unsigned int maxBatchSize,    // batch size - NB must be at least as large as the batch we want to run with
                        nvinfer1::DataType dataType,//
                        IInt8Calibrator* calibrator,//矫正器
                        IHostMemory*& trtModelStream); // output buffer for the TensorRT model
    void cv_show(cv::Mat &img,const string &win_name,
                 int win_x,int win_y,
                 float f_x,float f_y,
                 char wait_flag,char bgr_flag);

    int init();
    int normal_test();

    /******************条件变量*******************/

    condition_variable task1_condition;
    condition_variable task2_condition;
    condition_variable task3_condition;
    condition_variable task4_condition;
    condition_variable task5_condition;
    condition_variable task6_condition;
    /******************锁标志*******************/
    mutex task1_lock;
    mutex task2_lock;
    mutex task3_lock;
    mutex task4_lock;
    mutex task5_lock;
    mutex task6_lock;

    mutex task4_use_lock;
    mutex task6_show_lock;


    mutex task1_lock_FIFO;
    mutex task2_lock_FIFO;
    mutex task3_lock_FIFO;
    mutex task4_lock_FIFO;
    mutex task5_lock_FIFO;

    /******************缓存*******************/

    deque<Mat > task_1_out_FIFO;
    deque<float *> task_2_out_FIFO;
    deque<float *> task_3_out_FIFO;
    deque<Mat> task_4_out_FIFO_img,task_4_out_FIFO_label;
    vector<Mat> task_5_out_FIFO;

    /****************** 结束标志*******************/

    bool task1_end;
    bool task5_end;
    bool task3_end=true;
    bool task2_end;
    bool task4_end;

    unsigned int times;
    clock_t start=clock();
    clock_t end=clock();
    double endtime;
    cv::Mat cap_frame;//原图像
    cv::Mat show_frame;//显示图像

    /******************图片变量*******************/

    int p=1;
    cv::Mat out;//原始输出
    cv::Mat out_color;//彩色输出
    cv::Mat real_out;//大图彩色输出
    int outputSize;
    /******************推理变量*******************/

    IExecutionContext*   context;
    ICudaEngine* engine ;
    IRuntime* infer;

};





#endif // main_task_H
