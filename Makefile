main: 
	g++ -std=gnu++23 -g train.cxx -o train.a

gpu:
	nvcc -ccbin /usr/bin/gcc-14 -Wno-deprecated-gpu-targets -Xcompiler="-fpermissive" tensors_cuda.cu -c tensors_cuda.o

eval: 
	g++ -std=gnu++23 -g eval.cxx -o eval.a
	./eval.a

test: 
	g++ -std=gnu++23 -g test.cxx -o test.a
	./test.a

example: 
	g++ -std=gnu++23 -g example.cxx -o example.a
	./example.a