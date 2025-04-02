default:
	nvcc -ccbin=g++-11 -O0 src/main.cu -o build/main.bin
	./build/main.bin