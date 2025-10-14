OpenCV and ONNX Runtime are required to build the example code "ObjectDetection_Implementation.cpp".
After installing OpenCV and ONNX Runtime, make sure to add the following configuration to your CMakeLists.txt file:

# OpenCV 
find_package(OpenCV REQUIRED)
target_include_directories(AI_BMT_GUI_Submitter PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(AI_BMT_GUI_Submitter PUBLIC ${OpenCV_LIBS})

# ONNX Runtime (Be sure to update ONNXRUNTIME_DIR to match your installation path)
set(ONNXRUNTIME_DIR "/path/to/onnxruntime-linux-x64")  # Modify this path
target_include_directories(AI_BMT_GUI_Submitter PUBLIC ${ONNXRUNTIME_DIR}/include)
target_link_libraries(AI_BMT_GUI_Submitter PUBLIC ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
