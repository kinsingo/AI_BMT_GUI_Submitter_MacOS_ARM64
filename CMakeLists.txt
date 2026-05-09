cmake_minimum_required(VERSION 3.14)

project(AI_BMT_GUI_Submitter_App VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - parallel processing enabled")
else()
    message(WARNING "OpenMP not found - install with: brew install libomp")
    # Try to find libomp manually on macOS
    if(APPLE)
        find_library(OpenMP_LIBRARY
            NAMES omp
            PATHS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib
        )
        if(OpenMP_LIBRARY)
            set(OpenMP_CXX_FOUND TRUE)
            include_directories(/opt/homebrew/opt/libomp/include)
            set(OpenMP_CXX_FLAGS "-Xpreprocessor;-fopenmp")
            set(OpenMP_CXX_LIB_NAMES "omp")
        endif()
    endif()
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV4 REQUIRED opencv4)
# Using official ONNX Runtime with GPU support instead of find_package
# find_package(OnnxRuntime REQUIRED)

set(PROJECT_SOURCES main.cpp)

# Create the executable
add_executable(AI_BMT_GUI_Submitter ${PROJECT_SOURCES})

# Add the directory where the headers are located
target_include_directories(AI_BMT_GUI_Submitter PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OPENCV4_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-osx-arm64-1.23.1/include
)

# Add compile definitions and link directories
target_compile_definitions(AI_BMT_GUI_Submitter PUBLIC ${OPENCV4_CFLAGS_OTHER})
target_link_directories(AI_BMT_GUI_Submitter PUBLIC 
    ${OPENCV4_LIBRARY_DIRS}
    /opt/homebrew/opt/opencv/lib
)

# Add OpenMP flags if available
if(OpenMP_CXX_FOUND)
    target_compile_options(AI_BMT_GUI_Submitter PRIVATE -Xpreprocessor -fopenmp)
endif()

# Link the libraries to the executable
target_link_libraries(AI_BMT_GUI_Submitter PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/build/lib/libAI_BMT_GUI_Library.dylib
    ${OPENCV4_LIBRARIES}
    ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-osx-arm64-1.23.1/lib/libonnxruntime.dylib
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    if(APPLE AND OpenMP_LIBRARY)
        target_link_libraries(AI_BMT_GUI_Submitter PUBLIC ${OpenMP_LIBRARY})
    else()
        target_link_libraries(AI_BMT_GUI_Submitter PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

# Set RPATH to include the lib directory during the build and install phases (macOS)
set_target_properties(AI_BMT_GUI_Submitter PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib;/opt/homebrew/opt/opencv/lib;${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-osx-arm64-1.23.1/lib"
    INSTALL_RPATH "@executable_path/../lib;@loader_path/../lib;/opt/homebrew/opt/opencv/lib;@executable_path/../onnxruntime-osx-arm64-1.23.1/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
    MACOSX_RPATH TRUE
)

# Ensure RPATH is always used (not stripped out)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
