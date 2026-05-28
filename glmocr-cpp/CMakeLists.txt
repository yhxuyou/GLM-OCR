cmake_minimum_required(VERSION 3.15)
project(glmocr-cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build tests" OFF)

# Find required packages
find_package(Threads REQUIRED)

# libcurl for HTTP requests
find_package(CURL REQUIRED)

# nlohmann/json (header-only)
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
)
FetchContent_MakeAvailable(nlohmann_json)

# stb_image for image loading (header-only)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
)
FetchContent_MakeAvailable(stb)

# Create glmocr library
add_library(glmocr STATIC
    src/config.cpp
    src/page_loader.cpp
    src/ocr_client.cpp
    src/result_formatter.cpp
    src/pipeline.cpp
    src/utils/image_utils.cpp
    src/utils/string_utils.cpp
    src/utils/base64.cpp
)

target_include_directories(glmocr PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${stb_SOURCE_DIR}
)

target_link_libraries(glmocr PUBLIC
    Threads::Threads
    CURL::libcurl
    nlohmann_json::nlohmann_json
)

# Build examples if enabled
if(BUILD_EXAMPLES)
    add_executable(glmocr_cli examples/cli.cpp)
    target_link_libraries(glmocr_cli PRIVATE glmocr)
    
    add_executable(simple_example examples/simple.cpp)
    target_link_libraries(simple_example PRIVATE glmocr)
endif()

# Build tests if enabled
if(BUILD_TESTS)
    enable_testing()
    # Add test files here
endif()

# Installation
install(TARGETS glmocr
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)
