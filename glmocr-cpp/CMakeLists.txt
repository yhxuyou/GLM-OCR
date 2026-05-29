cmake_minimum_required(VERSION 3.15)
project(glmocr-cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build tests" OFF)
option(USE_CUDA "Enable CUDA execution provider for ONNX Runtime" OFF)

find_package(Threads REQUIRED)
find_package(CURL REQUIRED)

# ONNX Runtime
if(NOT DEFINED ONNXRUNTIME_ROOT)
    if(USE_CUDA)
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-gpu")
    else()
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")
    endif()
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(STATUS "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}")
    message(STATUS "Please download ONNX Runtime and extract to ${ONNXRUNTIME_ROOT}")
    message(STATUS "  CPU: https://github.com/microsoft/onnxruntime/releases")
    message(STATUS "  GPU: https://github.com/microsoft/onnxruntime/releases (note: download GPU version)")
    message(FATAL_ERROR "ONNX Runtime is required")
endif()

include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
)
FetchContent_MakeAvailable(stb)

add_library(glmocr STATIC
    src/config.cpp
    src/page_loader.cpp
    src/ocr_client.cpp
    src/result_formatter.cpp
    src/pipeline.cpp
    src/layout_detector.cpp
    src/doc_detector.cpp
    src/orientation_detector.cpp
    src/undistorter.cpp
    src/server.cpp
    src/utils/image_utils.cpp
    src/utils/string_utils.cpp
    src/utils/base64.cpp
)

target_include_directories(glmocr PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${stb_SOURCE_DIR}
    ${ONNXRUNTIME_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(glmocr PUBLIC
    Threads::Threads
    CURL::libcurl
    nlohmann_json::nlohmann_json
)

# Link ONNX Runtime
if(WIN32)
    target_link_libraries(glmocr PUBLIC "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
elseif(APPLE)
    target_link_libraries(glmocr PUBLIC "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib")
else()
    target_link_libraries(glmocr PUBLIC "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
endif()

target_link_directories(glmocr PUBLIC "${ONNXRUNTIME_ROOT}/lib")

if(BUILD_EXAMPLES)
    add_executable(glmocr_cli examples/cli.cpp)
    target_link_libraries(glmocr_cli PRIVATE glmocr)
    
    add_executable(simple_example examples/simple.cpp)
    target_link_libraries(simple_example PRIVATE glmocr)

    add_executable(glmocr_server examples/server_main.cpp)
    target_link_libraries(glmocr_server PRIVATE glmocr)
endif()

if(BUILD_TESTS)
    enable_testing()
endif()

install(TARGETS glmocr
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
