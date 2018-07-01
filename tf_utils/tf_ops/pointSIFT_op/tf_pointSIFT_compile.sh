#/bin/bash
/usr/local/cuda-8.0/bin/nvcc pointSIFT.cu -o pointSIFT_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.4
g++ -std=c++11 main.cpp pointSIFT_g.cu.o -o tf_pointSIFT_so.so -shared -fPIC -I /home/jmydurant/anaconda3/envs/pointsift/lib/python3.5/site-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /home/jmydurant/anaconda3/envs/pointsift/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/home/jmydurant/anaconda3/envs/pointsift/lib/python3.5/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
