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

class ImageClassification_Interface_Implementation : public AI_BMT_Interface
{
private:
    Env env;
    RunOptions runOptions;
    shared_ptr<Session> session;
    array<const char *, 1> inputNames;
    array<const char *, 1> outputNames;
    MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    bool isUseMacOSGPU;
    bool isCustomDataset;
    const array<int64_t, 4> inputShape = {1, 3, 224, 224};
    const array<int64_t, 2> outputShape = {1, 1000};

public:
    // Constructor with GPU usage option
    explicit ImageClassification_Interface_Implementation(bool useMacOSGPU = false, bool isCustomDataset = false)
        : isUseMacOSGPU(useMacOSGPU), isCustomDataset(isCustomDataset) {}

    virtual InterfaceType getInterfaceType() override
    {
        if (isCustomDataset)
            return InterfaceType::ImageClassification_CustomDataset;
        else
            return InterfaceType::ImageClassification;
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
    }

    virtual Optional_Data getOptionalData() override
    {
        Optional_Data data;
        data.cpu_type = "Apple M4";                                                          // e.g., Intel i7-9750HF
        data.accelerator_type = isUseMacOSGPU ? "Apple M4 GPU (CoreML)" : "";                // e.g., DeepX M1(NPU)
        data.submitter = "";                                                                 // e.g., DeepX
        data.cpu_core_count = "10";                                                          // e.g., 16
        data.cpu_ram_capacity = "24GB";                                                      // e.g., 32GB
        data.cooling = "Passive";                                                            // e.g., Air, Liquid, Passive
        data.cooling_option = "Passive";                                                     // e.g., Active, Passive (Active = with fan/pump, Passive = without fan)
        data.cpu_accelerator_interconnect_interface = isUseMacOSGPU ? "Unified Memory" : ""; // e.g., PCIe Gen5 x16
        data.benchmark_model = "";                                                           // e.g., ResNet-50
        data.operating_system = "macOS 15.5";                                                // e.g., Ubuntu 20.04.5 LTS
        return data;
    }

    virtual VariantType preprocessVisionData(const string &imagePath) override
    {
        Mat image = imread(imagePath);

        // convert BGR to RGB before reshaping
        cvtColor(image, image, cv::COLOR_BGR2RGB);

        if (isCustomDataset)
        {
            const int target_short = 232;
            const int crop = 224;

            int h = image.rows;
            int w = image.cols;

            // 1) 짧은 변을 232로 맞추는 비율 (종횡비 유지)
            double scale = static_cast<double>(target_short) / std::min(h, w);
            int new_w = static_cast<int>(std::round(w * scale));
            int new_h = static_cast<int>(std::round(h * scale));

            // Downscale면 INTER_AREA, Upscale면 INTER_LINEAR 권장
            int interp = (scale < 1.0) ? cv::INTER_AREA : cv::INTER_LINEAR;

            cv::Mat resized;
            cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, interp);

            // 2) 중심 224x224 크롭
            int x = (resized.cols - crop) / 2;
            int y = (resized.rows - crop) / 2;
            cv::Rect roi(x, y, crop, crop);
            image = resized(roi).clone();
        }

        // Convert to float and normalize to [0, 1]
        image.convertTo(image, CV_32F, 1.0 / 255.0);

        // Mean and Std deviation values for ImageNet
        const vector<float> means = {0.485, 0.456, 0.406};
        const vector<float> stds = {0.229, 0.224, 0.225};

        // Split channels for proper normalization
        vector<Mat> channels;
        split(image, channels);

        vector<float> output;
        output.reserve(3 * 224 * 224);

        // Normalize each channel and arrange in CHW format (Channel-Height-Width)
        for (int c = 0; c < 3; ++c)
        {
            channels[c] = (channels[c] - means[c]) / stds[c];

            float *data = reinterpret_cast<float *>(channels[c].data);
            for (int i = 0; i < 224 * 224; ++i)
            {
                output.push_back(data[i]);
            }
        }

        return output;
    }

    virtual vector<BMTVisionResult> inferVision(const vector<VariantType> &data) override
    {
        vector<BMTVisionResult> results;
        vector<float> outputData(1000);
        for (int i = 0; i < data.size(); ++i)
        {
            const vector<float> &imageVec = get<vector<float>>(data[i]);
            auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float *>(imageVec.data()), imageVec.size(), inputShape.data(), inputShape.size());
            auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, outputData.data(), outputData.size(), outputShape.data(), outputShape.size());
            session->Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
            BMTVisionResult result;
            result.classProbabilities = std::move(outputData);
            results.push_back(std::move(result));
        }
        return results;
    }
};

int main(int argc, char *argv[])
{
    try
    {
        bool useMacOSGPU = true;
        bool isCustomDataset = false;
        shared_ptr<AI_BMT_Interface> interface = make_shared<ImageClassification_Interface_Implementation>(useMacOSGPU, isCustomDataset);
        cout << "Starting Object Detection BMT with " << (useMacOSGPU ? "GPU (CoreML)" : "CPU") << " acceleration" << endl;
        return AI_BMT_GUI_CALLER::call_BMT_GUI_For_Single_Task(argc, argv, interface);
    }
    catch (const exception &ex)
    {
        cout << ex.what() << endl;
    }
}
