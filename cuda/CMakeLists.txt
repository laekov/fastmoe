cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(moe)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

if(NOT PYTHON_INCLUDE)
	set(PYTHON_INCLUDE "/home/jiezhong/miniconda3/include/python3.8")
endif()

if(NOT CUDA_HOME)
	set(CUDA_HOME "/usr/local/cuda")
endif()

if(NOT CUDA_SAMPLE_INCLUDE)
	set(CUDA_SAMPLE_INCLUDE "/usr/local/cuda/samples/common/inc")
endif()

include_directories(
	"${PYTHON_INCLUDE}"
	"${CUDA_HOME}/include"
	"${CUDA_SAMPLE_INCLUDE}"
	)
add_executable(moe moe.cpp cuda_stream_manager.cpp)
target_link_libraries(moe 
                    "${TORCH_LIBRARIES}")
set_property(TARGET moe PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET moe
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:moe>)
endif (MSVC)
