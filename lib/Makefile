CC = gcc
LFLAGS = -lm
SRC = common.c layer.c
OBJ = $(SRC:.c=.o)

.SUFFIXES : .c .o

all : $(OBJ)

#common.o : common.c common.h
#	$(CC) -c common.c
#
#sample.o : sample.c sample.h common.h
#	$(CC) -c sample.c 

clean :
	rm $(OBJ)
