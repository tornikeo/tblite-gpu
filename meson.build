project(
  'tblite-gpu',
  'cuda',
  version: '0.2.3',
  license: 'Apache-2.0 OR MIT',
  meson_version: '>=0.53',
  default_options: [
    'buildtype=debuoptimized',
    'default_library=both',
  ],
)

build_args_common = [
  '-ccbin=cc',
  '-c',
  # The 64-bit floating-point version of atomicAdd() is only supported by devices of compute capability
  # 6.x and higher. We put our current compute cap for an RTX 4060
  '-gencode=arch=compute_70,code=sm_70',  # Volta architecture (e.g., V100)
  '-gencode=arch=compute_90,code=sm_90',  # H100
  '-gencode=arch=compute_89,code=sm_89',  # RTX 4060
  '-gencode=arch=compute_80,code=sm_80',  # Ampere architecture (e.g., A100)
  # '--gpu-architecture=native,'
  # '-gencode=arch=compute_75,code=sm_75',  # Turing architecture (e.g., RTX 2080)
  # '-gencode=arch=compute_61,code=sm_61',  # Pascal architecture (e.g., GTX 1080) (unsupported due to __grid_constant__ not being available
  # '-gencode=arch=compute_60,code=sm_60',  # Pascal architecture (e.g., Tesla P100)
  '-m64',
  '--ptxas-options=-v',
  # '--use_fast_math',
  # '--ftz=true',
  # '--prec-div=false',
  # '--prec-sqrt=false',
  # '--default-stream=per-thread',
]

install = not (meson.is_subproject() and get_option('default_library') == 'static')

tblitegpu_lib = library(
  'tblite-gpu', 'src/main.cu', 
  install: install,
  cuda_args : build_args_common,
)

tblitegpu_inc = tblitegpu_lib.private_dir_include()
tblitegpu_dep = declare_dependency(
  link_with: tblitegpu_lib,
  include_directories: tblitegpu_inc,
)

nvcc_id = 'nvcc'
if install
  module_id = meson.project_name() / nvcc_id
  pkg = import('pkgconfig')
  pkg.generate(
    tblitegpu_lib,
    description: 'A gpu accelerated tblite',
    subdirs: ['', module_id],
  )
endif

test_exe = executable(
  'test-kernel', 
  sources: 'tests/test_kernel.cu', 
  link_with: tblitegpu_lib
)

test('CUDA Kernel Test', test_exe)