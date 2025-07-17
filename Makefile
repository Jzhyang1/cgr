main: 
	g++ -std=gnu++23 -g train.cxx -o train.a

gpu:
	nvcc -std=c++23 train.cxx -o train.a

test: 
	g++ -std=gnu++23 -g test.cxx -o test.a
	./test.a