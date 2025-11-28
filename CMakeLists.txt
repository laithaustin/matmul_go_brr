cmake_minimum_required(VERSION 3.10)
project(sgemm LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Add source directory to include path
include_directories(${PROJECT_SOURCE_DIR}/src)

# CUDA architecture - adjust for your GPU
# Common options:
#   RTX 30xx series: -arch=compute_86
#   RTX 40xx series: -arch=compute_89
#   Tesla V100: -arch=compute_70
#   Tesla A100: -arch=compute_80
set(CMAKE_CUDA_ARCHITECTURES 86)

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Optional: Enable debug info
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")

# Optional: L1 cache configuration
# -Xptxas -dlcm=ca (cache all, more L1 cache)
# -Xptxas -dlcm=cg (cache global, more shared memory)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -dlcm=ca")

# C++ compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")

# Collect source files
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRCS)

# Create executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
add_executable(sgemm sgemm.cu ${SRCS})

# Link libraries
target_link_libraries(sgemm ${CUDA_LIBRARIES} cublas)

# Set CUDA properties
set_target_properties(sgemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
