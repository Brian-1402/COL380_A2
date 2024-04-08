# Remove the mkdir statment for final submission because piazza post directly calls ./subtask{i} instead of ./bin/subtask{i}

CCFLAGS = -g -Wall -std=c++11
CC = g++

all: subtask1 subtask2 subtask3 subtask4

subtask1: subtask1.cpp
	mkdir -p ./bin
	$(CC) $(CCFLAGS) -o ./bin/subtask1 subtask1.cpp

subtask2: subtask2.cu
	mkdir -p ./bin
	nvcc -o ./bin/subtask2 subtask2.cu

subtask3: subtask3.cu
	mkdir -p ./bin
	nvcc -o ./bin/subtask3 subtask3.cu

subtask4: subtask4.cu
	mkdir -p ./bin
	nvcc -o ./bin/subtask4 subtask4.cu

clean:
	rm -rf ./bin