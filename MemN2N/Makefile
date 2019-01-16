.SUFFIXES : .c .o

PATH_HOME = ${CURDIR}
#PATH_USER_LIB_SRC = /home/sspark/Projects/01_memory_network/lib
PATH_USER_LIB_SRC = $(PATH_HOME)/../lib
PATH_USER_LIB_OBJ = .
#PATH_USER_INC = /home/sspark/Projects/01_memory_network_tmp/lib
PATH_SRC = .
PATH_OBJ = .

#USER_LIB=-L$(PATH_USER_LIB)
USER_INC=-I$(PATH_USER_LIB_SRC) -I$(PATH_SRC)


CC = gcc
CFLAGS = -w
#LFLAGS = -lm $(USER_LIB) -lpthread -lstdc++
LFLAGS = -lm -pthread -lstdc++
LFLAGS_CUDA = -lcuda -lcudart -L/usr/local/cuda/lib64
#LFLAGS_CUDA = -L/usr/local/cuda/lib64 -lcuda -lcudart
INC = $(USER_INC)
#SRC = MemN2N.c sample.c
SRC = $(PATH_SRC)/MemN2N.c $(PATH_SRC)/sample.c
OBJ = $(SRC:.c=.o)

#OBJ = $(patsubst $(PATH_SRC), $(PATH_OBJ), $(SRC:.c=.o))
#OBJ = $(addprefix $(PATH_OBJ)/,$(SRC:.c=.o))
#OBJ = $(PATH_OBJ)/MemN2N.o $(PATH_OBJ)/sample.o


#USER_LIB_SRC = $(PATH_USER_LIB_SRC)/common.c $(PATH_USER_LIB_SRC)/layer.c
#USER_LIB_OBJ = $(USER_LIB_SRC:.c=.o)

USER_LIB_SRC = common.c layer.c
USER_LIB_OBJ = $(USER_LIB_SRC:.c=.o)


#USER_LIB_SRC_CU = $(PATH_USER_LIB_SRC)/layer_cuda.cu
#USER_LIB_OBJ_CUDA = $(USER_LIB_SRC_CU:.cu=.o)

USER_LIB_SRC_CU = layer_cuda.cu
USER_LIB_OBJ_CUDA = $(USER_LIB_SRC_CU:.cu=.o)

TARGET = MemN2N

all : $(TARGET)

$(TARGET) : $(OBJ) $(USER_LIB_OBJ) $(USER_LIB_OBJ_CUDA)
	$(CC) $(CFLAGS) $(INC) $(OBJ) $(USER_LIB_OBJ) $(USER_LIB_OBJ_CUDA) $(LFLAGS) $(LFLAGS_CUDA) -o $(PATH_OBJ)/$(TARGET)

#$(OBJ) : 
#	$(CC) $(CFLAGS) $(INC) -c $*.c -o $@

MemN2N.o : MemN2N.c MemN2N.h define.h $(PATH_USER_LIB_SRC)/common.h
	$(CC) $(CFLAGS) $(INC) -c MemN2N.c -o $(PATH_OBJ)/MemN2N.o

sample.o : sample.c sample.h $(PATH_USER_LIB_SRC)/common.h
	$(CC) $(CFLAGS) $(INC) -c sample.c -o $(PATH_OBJ)/sample.o


common.o : $(PATH_USER_LIB_SRC)/common.c $(PATH_USER_LIB_SRC)/common.h $(PATH_SRC)/define.h
	$(CC) $(CFLAGS) $(INC) -c $(PATH_USER_LIB_SRC)/common.c -o $(PATH_USER_LIB_OBJ)/common.o 

layer.o : $(PATH_USER_LIB_SRC)/layer.c $(PATH_USER_LIB_SRC)/layer.h $(PATH_USER_LIB_SRC)/common.h $(PATH_SRC)/define.h
	$(CC) $(CFLAGS) $(INC) -c $(PATH_USER_LIB_SRC)/layer.c -o $(PATH_USER_LIB_OBJ)/layer.o


layer_cuda.o : $(PATH_USER_LIB_SRC)/layer_cuda.cu $(PATH_USER_LIB_SRC)/layer_cuda.h $(PATH_USER_LIB_SRC)/common.h $(PATH_SRC)/define.h 
	#nvcc -c -arch=sm_61 $(INC) $(PATH_USER_LIB_SRC)/layer_cuda.cu -o $(PATH_USER_LIB_OBJ)/layer_cuda.o 
	nvcc -c $(INC) $(PATH_USER_LIB_SRC)/layer_cuda.cu -o $(PATH_USER_LIB_OBJ)/layer_cuda.o 
	#nvcc -c $(INC) $(PATH_USER_LIB_SRC)/layer_cuda.cu -o $(PATH_USER_LIB_OBJ)/layer_cuda.o 



clean :
	rm $(OBJ) $(USER_LIB_OBJ) $(USER_LIB_OBJ_CUDA) $(TARGET)
