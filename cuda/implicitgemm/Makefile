CC=nvcc
TARGET := implgemm_OPT

CXXFLAGS += -arch=sm_60
INCLUDES  += -I./include
LDFLAGS = -lcudnn 

# 定义源文件列表
SRCS := ./src/$(TARGET).cu ./main.cu

#将对应的c文件名转为o文件后放在下面的CUR_OBJS变量中
CUR_OBJS=${SRCS:.cu=.o}

EXECUTABLE=implgemm

all:$(EXECUTABLE)

$(EXECUTABLE): $(CUR_OBJS)
	$(CC) $(CUR_OBJS) $(LDFLAGS) -o $(EXECUTABLE)
      
	
%.o:%.cu
	$(CC) -c -w $< $(CXXFLAGS) $(INCLUDES) -o $@
	
	
clean:
	rm -f $(EXECUTABLE)
	rm -f ./src/*.o
	rm -f ./*.o
