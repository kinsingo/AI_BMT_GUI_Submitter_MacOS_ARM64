#include "ai_bmt_gui_caller.h"
#include "ai_bmt_interface.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

using namespace std;
using namespace cv;
using namespace Ort;

class ObjectDetection_Interface_Implementation : public AI_BMT_Interface
{
private:
    Env env;
    RunOptions runOptions;
    shared_ptr<Session> session;
    array<const char *, 1> inputNames;
    array<const char *, 1> outputNames;
    MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    bool isUseMacOSGPU;
    vector<int64_t> outputShape; // 자동 감지된 출력 shape

public:
    // Constructor with GPU usage option
    explicit ObjectDetection_Interface_Implementation(bool useMacOSGPU = false)
        : isUseMacOSGPU(useMacOSGPU) {}

    virtual InterfaceType getInterfaceType() override
    {
        return InterfaceType::ObjectDetection;
    }

    // Power measurement selection (default: do not measure)
    virtual PowerDeviceType getPowerDeviceType() override
    {
        return PowerDeviceType::AppleSoC;
    }

    virtual void initialize(string modelPath) override
    {
        // session initializer
        SessionOptions sessionOptions;
        // Apply GPU acceleration if requested
        if (isUseMacOSGPU)
        {
            try
            {
                sessionOptions.AppendExecutionProvider("CoreML");
                cout << "Using CoreML execution provider for GPU acceleration" << endl;
            }
            catch (...)
            {
                cout << "CoreML execution provider not available, falling back to CPU" << endl;
                isUseMacOSGPU = false; // Update flag to reflect actual usage
            }
        }
        session = make_shared<Session>(env, modelPath.c_str(), sessionOptions);

        // Get input and output names
        AllocatorWithDefaultOptions allocator;
        AllocatedStringPtr inputName = session->GetInputNameAllocated(0, allocator);
        AllocatedStringPtr outputName = session->GetOutputNameAllocated(0, allocator);
        inputNames = {inputName.get()};
        outputNames = {outputName.get()};
        inputName.release();
        outputName.release();

        // 출력 shape 자동 감지
        TypeInfo outputTypeInfo = session->GetOutputTypeInfo(0);
        auto tensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        outputShape = tensorInfo.GetShape();

        cout << "Detected output shape: [";
        for (size_t i = 0; i < outputShape.size(); ++i)
        {
            cout << outputShape[i];
            if (i < outputShape.size() - 1)
                cout << ", ";
        }
        cout << "]" << endl;
    }

    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.cpu_type = "Apple M4";                                                          // e.g., Intel i7-9750HF
        data.accelerator_type = isUseMacOSGPU ? "Apple M4 GPU (CoreML)" : "";                // e.g., DeepX M1(NPU)
        return data;
    }

    virtual VariantType preprocessVisionData(const string &imagePath) override
    {
        // Load padded image
        Mat image = imread(imagePath);
        if (image.empty())
        {
            cerr << "Image not found at: " << imagePath << endl;
            throw runtime_error("Image not found!");
        }

        // Convert to float and normalize
        Mat floatImg;
        image.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);
        cvtColor(floatImg, floatImg, COLOR_BGR2RGB);

        // HWC → CHW
        vector<Mat> chw;
        split(floatImg, chw);
        vector<float> inputTensorValues;
        for (int c = 0; c < 3; ++c)
        {
            inputTensorValues.insert(inputTensorValues.end(),
                                     (float *)chw[c].datastart, (float *)chw[c].dataend);
        }
        return inputTensorValues;
    }

    virtual vector<BMTVisionResult> inferVision(const vector<VariantType> &data) override
    {
        // onnx option setting
        const int querySize = data.size();
        vector<BMTVisionResult> results;
        array<int64_t, 4> inputShape = {1, 3, 640, 640};

        // outputShape는 initialize()에서 자동 감지됨
        // 출력 데이터 크기 계산
        size_t outputDataSize = 1;
        for (size_t i = 1; i < outputShape.size(); ++i)
            outputDataSize *= outputShape[i];

        for (int i = 0; i < querySize; i++)
        {
            vector<float> imageVec = get<vector<float>>(data[i]);
            vector<float> outputData(outputDataSize);
            auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, imageVec.data(), imageVec.size(), inputShape.data(), inputShape.size());
            auto outputTensor = Value::CreateTensor<float>(memory_info, outputData.data(), outputData.size(), outputShape.data(), outputShape.size());

            // Run inference
            session->Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);

            // Update results
            BMTVisionResult result;
            result.objectDetectionResult = outputData;
            results.push_back(result);
        }
        return results;
    }
};

int main(int argc, char *argv[])
{
    try
    {
        bool useMacOSGPU = true;
        shared_ptr<AI_BMT_Interface> interface = make_shared<ObjectDetection_Interface_Implementation>(useMacOSGPU);
        cout << "Starting Object Detection BMT with " << (useMacOSGPU ? "GPU (CoreML)" : "CPU") << " acceleration" << endl;
        return AI_BMT_GUI_CALLER::call_BMT_GUI_For_Single_Task(argc, argv, interface);
    }
    catch (const exception &ex)
    {
        cout << ex.what() << endl;
    }
}
