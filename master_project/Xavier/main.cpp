#include <QCoreApplication>
#include "main_task.h"


/********************************************
//该文件功能:
//备注:5个线程识别车道线停止线
//修改日期:2019.
**********************************************/


deque<int> inputdata;
deque<int> outputdata;




int main(int argc, char *argv[])
{

    main_task test;
    ifstream *nothing;

//    test.normal_test();//串行测试
    std::thread th1(&main_task::task_1,&test,nothing,ref(test.task_1_out_FIFO));
    std::thread th2(&main_task::task_2,&test,ref(test.task_1_out_FIFO),ref(test.task_2_out_FIFO));
    std::thread th3(&main_task::task_3,&test,ref(test.task_2_out_FIFO),ref(test.task_3_out_FIFO));
    std::thread th4(&main_task::task_4,&test,ref(test.task_3_out_FIFO),ref(test.task_4_out_FIFO_img),ref(test.task_4_out_FIFO_label));
    std::thread th5(&main_task::task_5,&test,ref(test.task_4_out_FIFO_img),ref(test.task_4_out_FIFO_label),ref(test.task_5_out_FIFO));
    std::thread th6(&main_task::task_show,&test,ref(test.task_4_out_FIFO_img),ref(test.task_4_out_FIFO_label),ref(test.task_5_out_FIFO));

    th1.join();
    th2.join();
    th3.join();
    th4.join();
    th5.join();
    th6.join();

    cout<<"all over "<<endl;
    pause();

    cv::destroyAllWindows();
    shutdownProtobufLibrary();
}
