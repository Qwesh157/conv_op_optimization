CC=nvcc
TARGET := implgemm

CXXFLAGS += -arch=sm_80
INCLUDES  += -I./include
LDFLAGS = 

SRCS := ./src/$(TARGET).cu ./main.cu

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
