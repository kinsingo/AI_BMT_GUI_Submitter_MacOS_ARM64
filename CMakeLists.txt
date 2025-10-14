cmake_minimum_required(VERSION 3.14)

project(AI_BMT_GUI_Submitter_App VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_SOURCES main.cpp)

# Create the executable
add_executable(AI_BMT_GUI_Submitter ${PROJECT_SOURCES})

# Add the directory where the headers are located
target_include_directories(AI_BMT_GUI_Submitter PUBLIC 
${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link the libraries to the executable (macOS dylib)
target_link_libraries(AI_BMT_GUI_Submitter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/build/lib/libAI_BMT_GUI_Library.dylib)

# Set RPATH to include the lib directory during the build and install phases (macOS)
set_target_properties(AI_BMT_GUI_Submitter PROPERTIES
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "@executable_path/../lib;@loader_path/../lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
    MACOSX_RPATH TRUE
)

# Ensure RPATH is always used (not stripped out)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
