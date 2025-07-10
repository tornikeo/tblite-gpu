# This file is part of tblite-gpu.
# SPDX-Identifier: LGPL-3.0-or-later
#
# tblite-gpu is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tblite-gpu is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with tblite-gpu.  If not, see <https://www.gnu.org/licenses/>.

project(
  'tblite-gpu',
  'cuda',
  version: '0.0.1',
  license: 'LGPL-3.0-or-later',
  meson_version: '>=0.53',
  default_options: [
    'buildtype=debugoptimized',
    'default_library=both',
  ],
)

build_args_common = [
  '-ccbin=cc',
  '-Xcompiler=-march=native',
  '-gencode=arch=compute_75,code=sm_75',
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