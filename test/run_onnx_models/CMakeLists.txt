cmake_minimum_required(VERSION 3.13)
project(onnx_test)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

add_compile_definitions(USE_CUDA)
add_subdirectory(cuda_energy_profiler)

set(ONNXRUNTIME_ROOTDIR "/home/t-chengluo/donglinb/Projects/onnxruntime/build/Linux/install")
include_directories(
    "${ONNXRUNTIME_ROOTDIR}/include"
    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session"
)
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

list(APPEND EXTERNAL_LIBS 
    onnxruntime 
    onnxruntime_providers_shared 
    libonnxruntime_providers_cuda.so 
)

add_executable(run_no_profiler src/run_no_profiler.cpp)
target_link_libraries(run_no_profiler ${EXTERNAL_LIBS})

add_executable(run_with_profiler src/run_with_profiler.cpp)
target_link_libraries(run_with_profiler ${EXTERNAL_LIBS} cuda_energy_profiler)

add_executable(run_ort_clean src/run_ort_clean.cpp)
target_link_libraries(run_ort_clean ${EXTERNAL_LIBS} cuda_energy_profiler)
