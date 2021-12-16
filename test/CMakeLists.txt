cmake_minimum_required(VERSION 3.13)
project(onnx_test)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
set (CMAKE_CXX_STANDARD 17)
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

add_compile_definitions(USE_CUDA)

add_library(cuda_energy_profiler cuda_energy_profiler.cc)
target_link_libraries(cuda_energy_profiler nvidia-ml pthread)

# set(ONNXRUNTIME_ROOTDIR "/home/t-chengluo/donglinb/Projects/onnx_all/onnxruntime/build/Linux/install")
set(ONNXRUNTIME_ROOTDIR "/home/t-chengluo/donglinb/Projects/onnxruntime/build/Linux/install")
include_directories(
    "${ONNXRUNTIME_ROOTDIR}/include"
    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session"
)
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

# include_directories(/home/t-chengluo/donglinb/libraries/boost_1_77_0)
# link_directories(/home/t-chengluo/donglinb/libraries/boost_1_77_0/stage/lib)

list(APPEND EXTERNAL_LIBS 
    onnxruntime 
    onnxruntime_providers_shared 
    libonnxruntime_providers_cuda.so 
    # cuda_energy_profiler
    # boost_regex 
)

add_executable(analyze_ort analyze_ort.cpp)
target_link_libraries(analyze_ort ${EXTERNAL_LIBS})

add_executable(run_ort run_ort.cpp)
target_link_libraries(run_ort ${EXTERNAL_LIBS} cuda_energy_profiler)