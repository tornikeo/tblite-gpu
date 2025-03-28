project('tblite-gpu', 'cuda')

tblitegpu_lib = library('tblite-gpu', 'src/kernel.cu')

tblitegpu_inc = tblitegpu_lib.private_dir_include()
tblitegpu_dep = declare_dependency(
  link_with: tblitegpu_lib,
  include_directories: tblitegpu_inc,
)

test_exe = executable('test-kernel', 'tests/test_kernel.cu',
    link_with: tblitegpu_lib)

test('CUDA Kernel Test', test_exe)

