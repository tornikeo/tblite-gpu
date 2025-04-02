project(
  'tblite-gpu',
  'cuda',
  version: '0.2.3',
  license: 'Apache-2.0 OR MIT',
  meson_version: '>=0.53',
  default_options: [
    'buildtype=debugoptimized',
    'default_library=both',
  ],
)

install = not (meson.is_subproject() and get_option('default_library') == 'static')

tblitegpu_lib = library(
  'tblite-gpu', 'src/main.cu', 
  install: install
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