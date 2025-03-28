project('tblite-gpu', 'cuda')

tblite_gpu_lib = library('tblite-gpu', 'src/kernel.cu')

test_exe = executable('test-kernel', 'tests/test_kernel.cu',
    link_with: tblite_gpu_lib)

test('CUDA Kernel Test', test_exe)
