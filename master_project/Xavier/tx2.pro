QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

OBJECTS_DIR =./OBJ

SOURCES += main.cpp \
    mask_task.cpp

HEADERS += \
    main_task.h \
    BatchStream.h


#opencv
INCLUDEPATH +=/usr/local/include \
                /usr/local/opencv \
                /usr/local/opencv2

LIBS += /usr/lib/libopencv_highgui.so \
        /usr/lib/libopencv_core.so    \
        /usr/lib/libopencv_imgproc.so\
        /usr/lib/libopencv_videoio.so\
        /usr/lib/libopencv_imgcodecs.so####   fuck


#tensorRT
INCLUDEPATH +=/usr/include/aarch64-linux-gnu \
            /usr/src/tensorrt/samples/common
LIBS += /usr/lib/aarch64-linux-gnu/libnv*.so

#CUDA
INCLUDEPATH += /usr/local/cuda/include
LIBS += /usr/local/cuda/lib64/*.so
