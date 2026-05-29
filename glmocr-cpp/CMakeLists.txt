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

set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

# --- ONNX Runtime ---
if(NOT DEFINED ONNXRUNTIME_ROOT)
    if(USE_CUDA)
        set(ONNXRUNTIME_ROOT "${THIRD_PARTY_DIR}/onnxruntime-gpu")
    else()
        set(ONNXRUNTIME_ROOT "${THIRD_PARTY_DIR}/onnxruntime")
    endif()
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}\n"
        "Please run prepare_deps.sh on an internet-connected machine first,\n"
        "or manually download ONNX Runtime to ${ONNXRUNTIME_ROOT}"
    )
endif()

# --- nlohmann/json (header-only) ---
set(NLOHMANN_JSON_DIR "${THIRD_PARTY_DIR}/nlohmann_json")
if(NOT EXISTS "${NLOHMANN_JSON_DIR}/single_include/nlohmann/json.hpp")
    message(FATAL_ERROR
        "nlohmann/json not found at ${NLOHMANN_JSON_DIR}\n"
        "Please run prepare_deps.sh on an internet-connected machine first."
    )
endif()
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE "${NLOHMANN_JSON_DIR}/single_include")
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)

# --- stb (header-only) ---
set(STB_DIR "${THIRD_PARTY_DIR}/stb")
if(NOT EXISTS "${STB_DIR}/stb_image.h")
    message(FATAL_ERROR
        "stb not found at ${STB_DIR}\n"
        "Please run prepare_deps.sh on an internet-connected machine first."
    )
endif()

# --- yaml-cpp (compiled) ---
set(YAML_CPP_DIR "${THIRD_PARTY_DIR}/yaml-cpp")
if(NOT EXISTS "${YAML_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "yaml-cpp not found at ${YAML_CPP_DIR}\n"
        "Please run prepare_deps.sh on an internet-connected machine first."
    )
endif()
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Disable yaml-cpp tests" FORCE)
add_subdirectory("${YAML_CPP_DIR}" yaml-cpp)

# --- cpp-httplib (header-only) ---
set(HTTPLIB_FILE "${THIRD_PARTY_DIR}/httplib.h")
if(NOT EXISTS "${HTTPLIB_FILE}")
    message(FATAL_ERROR
        "cpp-httplib not found at ${HTTPLIB_FILE}\n"
        "Please run prepare_deps.sh on an internet-connected machine first."
    )
endif()

# --- Main library ---
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
    ${STB_DIR}
    ${ONNXRUNTIME_ROOT}/include
    ${THIRD_PARTY_DIR}
)

target_link_libraries(glmocr PUBLIC
    Threads::Threads
    CURL::libcurl
    nlohmann_json::nlohmann_json
    yaml-cpp
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

    add_executable(test_visualize examples/test_visualize.cpp)
    target_link_libraries(test_visualize PRIVATE glmocr)
endif()

if(BUILD_TESTS)
    enable_testing()
endif()

install(TARGETS glmocr
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
