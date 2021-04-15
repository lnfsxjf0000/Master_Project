#include "main_task.h"
#define baidu_model
//#define city_scapes_model

#define loop_times 9000
#define INT_8  //8位
//#define FLOAT_16
//#define FLOAT_32
#define show_img
const bool cout_debug=true;




#define win_w 0
#define win_h 1080
#define stride 50
#define resize_w 650
#define resize_h 350


Logger gLogger;
static int gUseDLACore = -1;

// stuff we know about the network and the caffe input/output blobs
const char* INPUT_BLOB_NAME = "0";
const char* OUTPUT_BLOB_NAME = "652";
const char* gNetworkName{nullptr};

/******************数据集*******************/

#ifdef baidu_model
const std::vector<std::string> directories{"/home/nvidia/project/baidu_project/tx2/release/", "/"};
#endif
#ifdef city_scapes_model
const std::vector<std::string> directories{"data/samples/mnist/", "data/mnist/"};
#endif
/****************** mask*******************/
#ifdef baidu_model
const int object_class=21; //19city  21百度 包含路面 20 不包含路面
//const int h=75;//113;//128　　　百度原始　75 288
//const int w=288;//256
const int h=45;//113;//128
const int w=160;//256
//static const float kMean[3] = { 89.84892,81.72008, 73.357635 };
static const float kStdDev[3] = { 45.319, 46.152, 44.914 };// baidu
const int map_[21][3] = {
                            {0, 0, 0} ,
                            {54, 106, 228},
                            {255, 0, 128},
                            {255, 0, 0},
                            {102, 202, 156},
                            {  249, 74, 9},
                            {255, 255, 0},
                            {0, 0, 230},
                            {65, 65, 65},
                            {1, 255, 14},

                            {128, 78, 160},
                            {150, 100, 100},
                            { 128, 128, 0},
                            {180, 165, 180},
                            {107, 142, 35},
                            {201, 255, 229},
                            {0, 191, 255},
                            {51, 255, 51},
                            {250, 128, 114},
                            {127, 255, 0},
                            {0,0,0}
                           {128,50,128}//地面

};

#endif

#ifdef city_scapes_model
const int object_class=19; //19city
const int h=128;//128通过原来的onnx 改变输入测试图像大小 修改图像输入尺寸
const int w=256;//256 city
static const float kMean[3] = { 72.393, 82.909, 73.158 };
static const float kStdDev[3] = { 45.319, 46.152, 44.914 };// city

static const unsigned char map_[19][3] = { {128,64,128} ,
                            {244, 35, 232},
                            {70, 70, 70},
                            {102, 102, 156},
                            {190, 153, 153},
                            { 153, 153, 153},
                            {250, 170, 30},
                            {220, 220, 0},
                            {107, 142, 35},
                            {152, 251, 152},
                            {220, 20, 60},
                            {255, 0, 0},
                            {0, 0, 142},
                            {0, 0, 70},
                            {0, 60, 100},
                            {0, 80, 100},
                            {0, 0, 230},
                            {119, 11, 32}};
#endif


#include<sys/timeb.h>
string getSystemTime(void)
{
    timeb t;
    ftime(&t);
    char tmp[16];
    strftime(tmp,sizeof(tmp),"%Hh-%Mm-%Ss",localtime(&t.time));//Year Month Day Hour Minute Second
    char tm[20];
    sprintf(tm, "%s-%d ms",tmp,t.millitm);
    return tm;
}

main_task::main_task()
{
    times=0;

}

main_task::~main_task()
{
    context->destroy();
    engine->destroy();
    infer->destroy();
}

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs;
    dirs.push_back(std::string("data/int8/") + gNetworkName + std::string("/"));
    dirs.push_back(std::string("/"));
    return locateFile(input, dirs);
}



int main_task::task_1(ifstream *input, deque<Mat> &output)
{


    cout<<"task1 begin "<<getSystemTime()<< endl;
    ifstream file;
    string textname="./list/baidu_train_list.txt";
    string line_res;
    open_file(file,textname);

    VideoCapture cap("./avi/train.avi");
    static auto nowtime=getSystemTime();
    static auto lasttime=nowtime;
    const int step=20;//计算帧率的周期


    while(task3_end)
    {
        unique_lock<mutex> lock(task1_lock);//线程1锁
        task1_condition.wait(lock);//线程1等待

        getline(file,line_res);
        line_res=line_res.substr(0,line_res.find("\t"));
        line_res="./"+line_res;
        cout<<line_res<<"                                        times "<<times<<endl;
        if(times%step==0&&times>10)
        {
//            计算帧率
            lasttime=nowtime;
            nowtime=getSystemTime();
            string last_ms=lasttime.substr(12,4);
            string now_ms=nowtime.substr(12,4);
            string now_s=nowtime.substr(8,2);
            string last_s=lasttime.substr(8,2);
            int now_s_int,last_s_int,now_ms_int,last_ms_int;
            sscanf(now_ms.c_str(),"%d",&now_ms_int);//转换数据
            sscanf(now_s.c_str(),"%d",&now_s_int);//转换数据
            sscanf(last_s.c_str(),"%d",&last_s_int);//转换数据
            sscanf(last_ms.c_str(),"%d",&last_ms_int);//转换数据
            if((now_ms_int>last_ms_int)&&(now_s_int>=last_s_int))
            {
                int cost_time=(now_s_int-last_s_int)*1000+(now_ms_int-last_ms_int);
                float cost_each_time=float(cost_time*1.0/step);
                float fps=1000/cost_each_time;
                cout<<endl<<endl;
                cout<<"*************avgtime************* :"<<cost_each_time<<" ms "<<" *************fps************* : "<<fps<<endl;
                cout<<endl<<endl;
            }
            else
            {
                cout<<"nowtiime:"<<nowtime<<"s: "<<now_s_int<<"ms: "<<now_ms_int<<endl;
                cout<<"lasttime:"<<lasttime<<"s: "<<last_s_int<<"ms: "<<last_ms_int<<endl;
            }

        }

        cv::String cv_img(line_res);

        /******************读图*******************/

        if(cout_debug)
        {
            cout << "1.读图开始: " <<getSystemTime()<< endl;
        }

        start=clock();
        cap>>cap_frame;//视频读取  6ms/


        end=clock();
        endtime=(double)(end-start)/CLOCKS_PER_SEC;
        if(cout_debug)
        {
            cout << "1.读图结束: " <<getSystemTime()<<"  读图时间: "<<endtime*1000<< " ms"<< endl;
        }
        if(cap_frame.empty())
        {
           cout<<"task1 break "<<getSystemTime()<< endl;
           break;
        }


        if(times==loop_times)
        {
            break;
        }

        if(!cap_frame.empty())
        task1_lock_FIFO.lock();
        output.push_back(cap_frame.clone());//线程1处理
        task1_lock_FIFO.unlock();
        task1_end=true;

    }
    task3_end=false;
    cout<<"task1 over "<<getSystemTime()<< endl;
    return 0 ;
}

//输入转换
float* main_task::normal(cv::Mat &img) {
    static const float kMean[3] = { 101.44892,95.55008, 83.257635 };//百度 
    float * data;
    data = (float*)calloc(img.rows*img.cols * 3, sizeof(float));

    for (int c = 0; c < 3; ++c)
    {
        for (int i = 0; i < img.rows; ++i)
        { //获取第i行首像素指针
            cv::Vec3b *p1 = img.ptr<cv::Vec3b>(i);
            for (int j = 0; j < img.cols; ++j)
            {
                data[c * img.cols * img.rows + i * img.cols + j] = p1[j][2-c]-kMean[2-c];//rgb
            }
        }
    }

    return data;
}

int main_task::task_2(deque<Mat> &input, deque<float *> &output)
{


    float* data;
    cout<<"task2 begin "<<getSystemTime()<< endl;
    while(task3_end)
    {
        unique_lock<mutex> lock(task2_lock);//线程2锁
        task2_condition.wait(lock);//线程2等待
        if(input.size()<4)
        {
            task2_end=true;
            continue;
        }
        /****************** 输入数据*******************/
        Mat frame=input[2];
        if(!(frame.size()==Size(1692,855)||frame.size()==Size(1692,854)))//854
        {
            cout<<" wrong size: "<<frame.size()<<endl;
            task2_end=true;
            input.pop_front();
            continue;

        }
        cv::Rect roi(200, 300, 1280, 360);//百度缩小二分之一
        frame=frame(roi);

        start=clock();
        if(cout_debug)
        {
            cout << "2.输入转换开始: " <<getSystemTime()<< endl;
        }
        data= normal(frame);
        end=clock();
        endtime=(double)(end-start)/CLOCKS_PER_SEC;

        if(cout_debug)
        {
            cout << "2.输入转化结束: " <<getSystemTime()<<"  输入转换时间: "<<endtime*1000<< " ms"<< endl;
        }
        task2_lock_FIFO.lock();
        output.push_back(data); //线程2处理执行完毕
        task2_lock_FIFO.unlock();
        if(input.size()>=4)
        {   task1_lock_FIFO.lock();
            task4_use_lock.lock();//在task4使用时候不弹出 造成错误
            input.pop_front();
            task4_use_lock.unlock();
            task1_lock_FIFO.unlock();

        }
        if(times==loop_times)
        {
            break;
        }
        task2_end=true;

    }
    cout<<"task2 over "<<getSystemTime()<< endl;

    return 0;
}


int main_task::task_3(deque<float *> &input, deque<float *> &output)
{


    init();//必须放到主线程里初始化
    float *data;
    float outdata[2][160*45*21];//2维度指针缓存
    float *prob=outdata[0];

    cout<<"task3 begin  "<<getSystemTime()<< endl;
    cout<<"           算法开始执行    "<<getSystemTime()<< endl<< endl<< endl<< endl<< endl;
    bool odd_flag=false;//奇数偶数标志 切换
    task3_end=true;
    while(task3_end)
    {


        task1_end=false;
        task2_end=false;
        task4_end=false;
        task5_end=false;
        task1_condition.notify_one();//唤起其他线程
        task2_condition.notify_one();//唤起其他线程
        task4_condition.notify_one();//唤起其他线程
        task5_condition.notify_one();//唤起其他线程

        if(input.size()==0)
        {
            continue;
        }
        /******************前向传播 *******************/

//        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        odd_flag=!odd_flag;//取反 缓存
        if(odd_flag)
        {
            prob=outdata[0];
        }
        else
        {
            prob=outdata[1];
        }
        start=clock();
        data=input[0];
        if(cout_debug)
        {
            cout << "3.前向传播开始: " <<getSystemTime()<< endl;
        }
        doInference(*context,data, prob, 1);// release 7ms
        free(data);//释放内存

        end=clock();
        endtime=(double)(end-start)/CLOCKS_PER_SEC;

        if(cout_debug)
        {
            cout << "3.前向传播结束: " <<getSystemTime()<< "  前向传播时间: "<<endtime*1000<< " ms"<<endl;
        }
        task3_lock_FIFO.lock();
        output.push_back(prob); //线程3处理执行完毕
        task3_lock_FIFO.unlock();
        times++;


        if(input.size()>=2)
        {
            task2_lock_FIFO.lock();
            input.pop_front();
            task2_lock_FIFO.unlock();
        }

        if(times==loop_times)
        {
            break;
        }
        /****************** 延时对齐等待其他任务完毕*******************/

        if(times>10)
        {
            int while_times=0;
            while(!(task1_end&task5_end&task2_end&task4_end))
            {
                while_times++;
                if(while_times>20)
                {
                    break;
                }

                std::this_thread::sleep_for(std::chrono::microseconds(100));
                if(cout_debug)
                cout<<"等待任务："<<task1_end<<task2_end<<task4_end<<task5_end<<" while_times: "<<while_times<<" "<<getSystemTime()<<endl;
            }
        }
    }
    task3_end=false;
    cout<<"           算法结束执行    "<<getSystemTime()<< endl<< endl<< endl<< endl<< endl;
    cout<<"task3 end "<<getSystemTime()<< endl;
    return 0;
}

int main_task::task_4(deque<float *> &input, deque<Mat> &output_img, deque<Mat> &output_label)
{
    cout<<"task4 begin "<<getSystemTime()<< endl;
    while(task3_end)
    {
        unique_lock<mutex> lock(task4_lock);//线程4锁
        task4_condition.wait(lock);//线程4等待
        if(input.size()==0)
        {
            task4_end=true;
            continue;
        }
        /****************** 输出数据*******************/
        start=clock();
        if(cout_debug)
        {
            cout << "4.输出数据转换开始: " <<getSystemTime()<< endl;
        }
        task4_use_lock.lock();//在使用变量时候,task2任务不弹出
        show_frame=task_1_out_FIFO[0].clone();
        task4_use_lock.unlock();

        out=read2mat(input[0],out);//5MS  release 1ms
        out_color= map2threeunchar(out, out_color);//3MS release 1ms

        if(out_color.empty())
        {
           cout<<"task4  out null "<<getSystemTime()<< endl;
           if(input.size()>=2)
           {
               input.pop_front();
           }
           task4_end=true;
           continue;
        }

        cv::Size resize_wh(600,300);
        cv::resize(out_color,real_out,cv::Size(2300,600),0,0,cv::INTER_NEAREST);
        if(task_1_out_FIFO[0].empty())
        {
            cout<<"task4  task_1_out_FIFO  null "<<getSystemTime()<< endl;
            if(input.size()>=2)
            {
                input.pop_front();
            }
            task4_end=true;
            continue;

        }


        if(show_frame.empty())
        {
            cout<<"task4  show_frame  null "<<getSystemTime()<< endl;
            if(input.size()>=2)
            {
                input.pop_front();
            }
            task4_end=true;
            continue;

        }
        cv::resize(real_out,real_out,resize_wh,0,0,cv::INTER_NEAREST);//4ms
        cv::resize(show_frame,show_frame,resize_wh,0,0,cv::INTER_NEAREST);//4ms

        end=clock();
        endtime=(double)(end-start)/CLOCKS_PER_SEC;
        if(cout_debug)
        {
            cout << "4.输出数据转化结束: " <<getSystemTime()<<"  输出数据转换时间: "<<endtime*1000<< " ms"<< endl;
        }


        if(!show_frame.empty()&&!real_out.empty())
        {
            task4_lock_FIFO.lock();
            output_img.push_back(show_frame); //线程4处理执行完毕
            output_label.push_back(real_out); //线程4处理执行完毕
            task4_lock_FIFO.unlock();
        }

        if(input.size()>=2)
        {
            task3_lock_FIFO.lock();
            input.pop_front();
            task3_lock_FIFO.unlock();
        }
        if(times==loop_times)
        {
            break;
        }
        task4_end=true;

    }
    cout<<"task4 end "<<getSystemTime()<< endl;
    return 0;
}

int main_task::task_5(deque<Mat> &input_img, deque<Mat> &input_label, vector<Mat> &output)
{
    cout<<"task5 begin "<<getSystemTime()<< endl;
    while(task3_end)
    {
        unique_lock<mutex> lock(task5_lock);//线程5锁
        task5_condition.wait(lock);//线程5等待
        if(times>=loop_times)
        {
            break;
        }
        output.clear();
        if(input_label.size()==0||input_img.size()==0)
        {
            task5_end=true;
            continue;
        }
        /****************** 输出数据*******************/
        start=clock();
        if(cout_debug)
        {
            cout << "5.*******OPENCV开始******* : " <<getSystemTime()<< endl;
        }

        if(!input_img[0].empty()&&!input_label[0].empty())//非空判断
        detect_object(input_img[0],input_label[0],output);//Opencv后续处理
        end=clock();
        endtime=(double)(end-start)/CLOCKS_PER_SEC;
        if(cout_debug)
        {
            cout << "5.######opencv结束######: " <<getSystemTime()<<"  opencv后续处理时间: "<<endtime*1000<< " ms"<<endl<< endl;
        }
        while(input_label.size()>=2)
        {
            task6_show_lock.lock();
            task4_lock_FIFO.lock();
            input_label.pop_front();
            input_img.pop_front();
            task4_lock_FIFO.unlock();
            task6_show_lock.unlock();
        }
        task5_end=true;
    }
    cout<<"task5 end "<<getSystemTime()<< endl;
    return 0;
}

int main_task::task_show(deque<Mat> &input_img, deque<Mat> &input_label, vector<Mat> &intput_bin)
{
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(6,&mask);
    if(sched_setaffinity(0,sizeof(mask),&mask)<0){perror("sched_setaffinity error");}
    cout<<"task6 begin "<<getSystemTime()<< endl;
    while(task3_end)
    {
        unique_lock<mutex> lock(task6_lock);//线程5锁
        task6_condition.wait(lock);//线程5等待
        if(times>=loop_times)
        {
            break;
        }

        if(input_label.size()==0||input_img.size()==0)
        {

            continue;
        }
        if(cout_debug)
        {
            cout << "6.*******显示图像开始******* : " <<getSystemTime()<< endl;
        }
        task6_show_lock.lock();

        cv_show(input_img[0],"img",win_w+resize_w*0,0,1,1,0,1);
        cv_show(input_label[0],"resize_labels",win_w+resize_w*1,0,1,1,0,1);
        #ifdef show_img
        waitKey(1);//
        #endif
        task6_show_lock.unlock();

        end=clock();
        endtime=(double)(end-start)/CLOCKS_PER_SEC;
        if(cout_debug)
        {
            cout << "6.######显示图像结束######: " <<getSystemTime()<<"  显示图像时间: "<<endtime*1000<< " ms"<< endl;
        }
    }
    cout<<"task6 end "<<getSystemTime()<< endl;
    return 0;

}


//5ms
int main_task::detect_object(cv::Mat &img,cv::Mat &labels,std::vector<cv::Mat> &result)
{

    const bool debug=false;


    cv::Mat resize_img,resize_labels;

    if(img.empty()||labels.empty())
        return 5;

    resize_labels=labels.clone();
    resize_img=img.clone();
    vector<cv::Mat> label_channels;
    cv::split(resize_labels,label_channels);


    vector<cv::Mat> mask_vec;//21类目标
    vector<cv::Mat> line;
    vector<cv::Mat> stopline;
    vector<cv::Mat> parking;
    vector<cv::Mat> zebra;
    vector<cv::Mat> turn_guides;
    vector<cv::Mat> road;

    for(int i=0;i<object_class;i++)
    {
        cv::Mat res(resize_labels.rows,resize_labels.cols,CV_8U);
        res=(label_channels[2]==map_[i][0]&label_channels[1]==map_[i][1]&label_channels[0]==map_[i][2]);

        mask_vec.push_back(res);
        if(i==1||i==2)
        {
            line.push_back(res);//车道线
        }
        else if(i==3)
        {
            stopline.push_back(res);//停止线
        }
        else if(i==5)
        {
            parking.push_back(res);//车位线
        }
        else if(i==6)
        {
            zebra.push_back(res);//斑马线
        }
        else if (i==7||(i>8&&i<15)) {
            turn_guides.push_back(res);//转向
        }
        else if (i==20) {
            road.push_back(res);
        }
    }
    cv::Mat line_mask=line[0]|line[1];
    cv::Mat guids_mask=turn_guides[0]|turn_guides[1]|turn_guides[2]|turn_guides[3]|turn_guides[4]|turn_guides[5]|turn_guides[6];
    cv::Mat zebra_mask=zebra[0];
    cv::Mat parking_mask=parking[0];
    cv::Mat road_mask=road[0];
    cv::Mat stopline_mask=stopline[0];

    result.clear();//释放内存
    result.push_back(line_mask);
    result.push_back(stopline_mask);
    result.push_back(parking_mask);
    result.push_back(zebra_mask);
    result.push_back(road_mask);
    result.push_back(guids_mask);
    if(debug)
    cout<<"--6"<<endl;
//    cv::Mat white_morpholo_region;
    /******************车道线后处理 ARM速度慢 *******************/


    /******************停止线后处理 *******************/
   cv::Mat stopline_close_kernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(111,11), cv::Point(-1, -1));//5
//    cv::Mat stopline_open_kernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5), cv::Point(-1, -1));//5
//    cv::morphologyEx(stopline_mask,stopline_mask, cv::MORPH_OPEN , stopline_open_kernal , cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, 0);
   cv::morphologyEx(stopline_mask,stopline_mask, cv::MORPH_CLOSE , stopline_close_kernal, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, 0);
   /******************斑马线后处理 *******************/
   cv::Mat zebra_close_kernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7), cv::Point(-1, -1));//5
   cv::morphologyEx(zebra_mask,zebra_mask, cv::MORPH_CLOSE , zebra_close_kernal, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, 0);
   cv::morphologyEx(zebra_mask,zebra_mask, cv::MORPH_OPEN , zebra_close_kernal, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, 0);
   /*****************向导后处理 *******************/


   /******************道路后处理 *******************/
   cv::Mat road_open_kernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11,11), cv::Point(-1, -1));//5
   cv::Mat road_close_kernal = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21,21), cv::Point(-1, -1));//5
   cv::morphologyEx(road_mask,road_mask, cv::MORPH_CLOSE , road_close_kernal, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, 0);
   cv::morphologyEx(road_mask,road_mask, cv::MORPH_OPEN , road_open_kernal,cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, 0);


//    cout<<"--7"<<endl;
    cv_show(resize_img,"img",win_w+resize_w*0,0,1,1,0,1);
    cv_show(resize_labels,"resize_labels",win_w+resize_w*1,0,1,1,0,1);
//    cv_show(line_mask,"line",win_w+resize_w*0,resize_h*1,1,1,0,1);
//    cv_show(stopline[0],"stop_line",win_w+resize_w*1,resize_h*1,1,1,0,1);
//    cv_show(parking_mask,"parking_mask",win_w+resize_w*2,resize_h*0,1,1,0,1);
//    cv_show(zebra[0],"zebra",win_w+resize_w*0,resize_h*2,1,1,0,1);
//    cv_show(road[0],"road",win_w+resize_w*1,resize_h*2,1,1,0,1);
//    cv_show(guids_mask,"guids_mask",win_w+resize_w*2,resize_h*1,1,1,0,1);

    waitKey(1);//
return 0;
}

ifstream &main_task::open_file(ifstream &in, const string &file)
{

        in.close();
        in.clear();
        in.open(file.c_str());
        return in;

}






//输出转化
cv::Mat main_task::read2mat(float * prob,cv::Mat out)
{
    for (int i = 0; i < h; ++i)
    {
        cv::Vec<float, object_class> *p1 = out.ptr<cv::Vec<float, object_class>>(i);
        for (int j = 0; j < w; ++j)
        {
            for (int c = 0; c < object_class; ++c)
            {
                p1[j][c] = prob[c * w * h + i * w + j];
            }
        }
    }
    return out;
}

//图像查表恢复
cv::Mat main_task::map2threeunchar(cv::Mat real_out,cv::Mat real_out_)
{
    for (int i = 0; i < h; ++i)
    {
        cv::Vec<float, object_class> *p1 = real_out.ptr<cv::Vec<float, object_class>>(i);
        cv::Vec3b *p2 = real_out_.ptr<cv::Vec3b>(i);
        for (int j = 0; j < w; ++j)
        {
            int index = 0;
            float swap;

            for (int c = 0; c < object_class; ++c)
            {
                if (p1[j][0] < p1[j][c])
                {
                    swap = p1[j][0];
                    p1[j][0] = p1[j][c];
                    p1[j][c] = swap;
                    index = c;
                }
            }
            p2[j][0] = map_[index][2];
            p2[j][1] = map_[index][1];
            p2[j][2] = map_[index][0];
        }
    }
    return real_out_;
}




class Int8EntropyCalibrator : public IInt8EntropyCalibrator
{
public:
    Int8EntropyCalibrator(BatchStream& stream, int firstBatch, bool readCache = true)
        : mStream(stream)
        , mReadCache(readCache)
    {
        DimsNCHW dims = mStream.getDims();
        mInputCount = mStream.getBatchSize() * dims.c() * dims.h() * dims.w();
        CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
        mStream.reset(firstBatch);
    }

    virtual ~Int8EntropyCalibrator()
    {
        CHECK(cudaFree(mDeviceInput));
    }

    int getBatchSize() const override { return mStream.getBatchSize(); }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        if (!mStream.next())
            return false;

        CHECK(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
        assert(!strcmp(names[0], INPUT_BLOB_NAME));
        bindings[0] = mDeviceInput;
        return true;
    }

    const void* readCalibrationCache(size_t& length) override
    {
        mCalibrationCache.clear();
        std::ifstream input(calibrationTableName(), std::ios::binary);
        input >> std::noskipws;
        if (mReadCache && input.good())
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));

        length = mCalibrationCache.size();
        return length ? &mCalibrationCache[0] : nullptr;
    }

    void writeCalibrationCache(const void* cache, size_t length) override
    {
        std::ofstream output(calibrationTableName(), std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    static std::string calibrationTableName()
    {
        assert(gNetworkName);
        return std::string("CalibrationTable");
    }
    BatchStream mStream;
    bool mReadCache{true};

    size_t mInputCount;
    void* mDeviceInput{nullptr};
    std::vector<char> mCalibrationCache;
};

bool main_task::onnxToTRTModel(const std::string& modelFile, // name of the onnx model
                    unsigned int maxBatchSize,    // batch size - NB must be at least as large as the batch we want to run with
                    nvinfer1::DataType dataType,//
                    IInt8Calibrator* calibrator,//矫正器
                    IHostMemory*& trtModelStream) // output buffer for the TensorRT model
{
    int verbosity = (int) nvinfer1::ILogger::Severity::kWARNING;
    IBuilder* builder = createInferBuilder(gLogger);
    if (builder == nullptr)
    {
        return false;
    }
    nvinfer1::INetworkDefinition* network = builder->createNetwork();
    auto parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser->parseFromFile(locateFile(modelFile, directories).c_str(), verbosity))
    {
        string msg("failed to parse onnx file");
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }
    if ((dataType == nvinfer1::DataType::kINT8 && !builder->platformHasFastInt8()) || (dataType == nvinfer1::DataType::kHALF && !builder->platformHasFastFp16()))
        return false;

    //  build the engine int8
    builder->setMaxWorkspaceSize(1 << 33);//30    8G
    builder->setAverageFindIterations(1);
    builder->setMinFindIterations(1);
    builder->setDebugSync(true);
    builder->setInt8Mode(dataType == nvinfer1::DataType::kINT8);
    builder->setFp16Mode(dataType == nvinfer1::DataType::kHALF);
    builder->setInt8Calibrator(calibrator);
    if (gUseDLACore >= 0)
    {
        samplesCommon::enableDLA(builder, gUseDLACore);//change
        if (maxBatchSize > builder->getMaxDLABatchSize())
        {
            std::cerr << "Requested batch size " << maxBatchSize << " is greater than the max DLA batch size of "
                      << builder->getMaxDLABatchSize() << ". Reducing batch size accordingly." << std::endl;
            maxBatchSize = builder->getMaxDLABatchSize();
        }
    }

    builder->setMaxBatchSize(maxBatchSize);
    ICudaEngine* engine = builder->buildCudaEngine(*network);//
    assert(engine);
    parser->destroy();
    trtModelStream = engine->serialize();

    engine->destroy();
    network->destroy();
    builder->destroy();
    return true;
}

float main_task::doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{

    const bool debug=true;

    const ICudaEngine& engine = context.getEngine();
    assert(engine.getNbBindings() == 2);
    void* buffers[2];
    float ms{0.0f};

    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    int   outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // create GPU buffers and a stream
    Dims3 inputDims = static_cast<Dims3&&>(context.getEngine().getBindingDimensions(context.getEngine().getBindingIndex(INPUT_BLOB_NAME)));
    Dims3 outputDims = static_cast<Dims3&&>(context.getEngine().getBindingDimensions(context.getEngine().getBindingIndex(OUTPUT_BLOB_NAME)));

    size_t inputSize = batchSize * inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * sizeof(float);
    size_t outputSize = (batchSize * outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * sizeof(float));

    CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
    CHECK(cudaMalloc(&buffers[outputIndex], outputSize));
    cudaStream_t stream;
    cudaEvent_t start, end;
    CHECK(cudaMemcpy(buffers[inputIndex], input, inputSize, cudaMemcpyHostToDevice));

    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));
    cudaEventRecord(start, stream);

    float *input_end=input+inputDims.d[0] * inputDims.d[1] * inputDims.d[2]-1;//数据起始位置对的
    float *input_end_1=input+inputDims.d[0] * inputDims.d[1] * inputDims.d[2];//结束位置对的
    context.enqueue(batchSize, buffers, stream, nullptr);//推理

    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    CHECK(cudaMemcpy(output, buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    CHECK(cudaStreamDestroy(stream));
    return ms;
}




int main_task::init()
{

    cout<<"scoreModel begin"<<endl;
    cudaSetDevice(0); //set device id
    gNetworkName = std::string("baidu").c_str();

    int batchSize=4;
    int firstBatch=20;
    int nbScoreBatches=30;
    bool quiet = false;

    IHostMemory* trtModelStream{nullptr};
    bool valid = false;

#ifdef INT_8
    BatchStream calibrationStream(4, 1);
    Int8EntropyCalibrator calibrator(calibrationStream, 1);
    valid=onnxToTRTModel("baidu_relu_360.onnx",4,nvinfer1::DataType::kINT8,&calibrator, trtModelStream);//8
#endif
#ifdef FLOAT_16
    BatchStream calibrationStream(4, 1);
    Int8EntropyCalibrator calibrator(calibrationStream, 1);
    valid=onnxToTRTModel("baidu_relu_360.onnx",4,nvinfer1::DataType::kHALF,&calibrator, trtModelStream);//8
#endif
#ifdef FLOAT_32
    IInt8Calibrator* calibrator=nullptr;//32位
    valid=onnxToTRTModel("baidu_relu_360.onnx",batchSize,nvinfer1::DataType::kFLOAT,calibrator, trtModelStream);//32
#endif

    assert(trtModelStream != nullptr);
    // Create engine and deserialize model.
    infer = createInferRuntime(gLogger);
    assert(infer != nullptr);
    if (gUseDLACore >= 0)
    {
        infer->setDLACore(gUseDLACore);

    }
    engine = infer->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    assert(engine != nullptr);
    trtModelStream->destroy();
    context = engine->createExecutionContext();
    assert(context != nullptr);

    Dims3 outputDims = static_cast<Dims3&&>(context->getEngine().getBindingDimensions(context->getEngine().getBindingIndex(OUTPUT_BLOB_NAME)));
    outputSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2];

    out.create(outputDims.d[1]*p , outputDims.d[2]*p,CV_32FC(object_class));//原始输出
    out_color.create(outputDims.d[1] *p, outputDims.d[2]*p,CV_8UC3);//小图彩色输出
    real_out.create(outputDims.d[1]*8 , outputDims.d[2]*8,CV_8UC3);//大图彩色输出
    std::cout<<"outputDims.d[1]:"<<outputDims.d[1]<<"outputDims.d[2]:"<<outputDims.d[2]<<endl;

    cout<<"scoreModel over"<<endl;
}




/******************自定义显示函数 *******************/
void main_task::cv_show(cv::Mat &img,const string &win_name,
             int win_x,int win_y,
             float f_x,float f_y,
             char wait_flag,char bgr_flag)
{
    cv::Mat src,resize_img;

    src=img.clone();

    if(bgr_flag==0)
    {
        cv::cvtColor(src,src,cv::COLOR_RGB2BGR);
    }

    int width=src.cols;
    int heigh=src.rows;
    if(src.empty())
    {
        return ;
    }

    f_x=1;f_y=1;//修改后
    cv::resize(src,resize_img,cv::Size (int(width*f_x),int(heigh*f_y)),f_x,f_y,cv::INTER_CUBIC);
    cv::namedWindow(win_name);
    cv::imshow(win_name,resize_img);
    cv::moveWindow(win_name,win_x,win_y);

    if(wait_flag)
    {
        cv::waitKey(0);
    }

}

