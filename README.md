> **Last Updated:** 2025-10-14 (1st Release of macOS)

## Environment

1.  ISA(Instruction Set Architecture) : ARM64(aarch64)
2.  OS : macOS (Apple Silicon)

## Submitter User Guide Steps

Step1) Build System Set-up  
Step2) Interface Implementation  
Step3) Build and Start BMT

## Step 1) Build System Set-up (Installation Guide for macOS)

**1. Install Packages**

- Install Homebrew (if not already installed):
  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```
- Open a terminal and run the following commands to install CMake and Ninja Build System.
  ```bash
  brew install cmake                    # CMake
  brew install ninja                    # Ninja Build Systems
  ```
- Note: Xcode Command Line Tools (includes clang/clang++) should be installed automatically. If not, run:
  ```bash
  xcode-select --install
  ```

**2. Verify the Installation**

- You can check the versions of the installed tools by running the following commands. If these commands return version information for each tool, the installation was successful.
  ```bash
  cmake --version
  ninja --version
  clang --version
  ```

## Step2) Interface Implementation

- Implement AI_BMT_Interface to operate with the intended AI Processing Unit (e.g., CPU, GPU, NPU).

```cpp
#ifndef AI_BMT_INTERFACE_H
#define AI_BMT_INTERFACE_H
#include "label_type.h"
using namespace std;

class EXPORT_SYMBOL AI_BMT_Interface
{
public:
   virtual ~AI_BMT_Interface(){}

    // Optional: override to provide system metadata.
    // Returned values will be stored in the database (used for benchmarking context).
   virtual Optional_Data getOptionalData();

   // return the implemented interface task type.
   virtual InterfaceType getInterfaceType() = 0;

   // This initialize(..) function is guaranteed to be called before preprocess(..) and infer(..) are executed.
   // The submitter can load the model using the provided modelPath
   virtual void initialize(string modelPath) = 0;

   // Vision tasks: preprocessing & inference
   // - preprocessVisionData: convert raw image file into model input format
   // - inferVision: run inference on preprocessed data and return results
   virtual VariantType preprocessVisionData(const string& imagePath) {throw runtime_error("preprocessVisionData(..) should be implemented for vision task");}
   virtual vector<BMTVisionResult> inferVision(const vector<VariantType>& data) {throw runtime_error("inferVision(..) should be implemented for vision task");}

   // LLM tasks: preprocessing & inference
   // - preprocessLLMData: convert raw text input into model input format
   // - inferLLM: run inference on preprocessed data and return results
   virtual VariantType preprocessLLMData(const LLMPreprocessedInput& llmData) {throw runtime_error("LLMPreprocessedInput(..) should be implemented for llm task");}
   virtual vector<BMTLLMResult> inferLLM(const vector<VariantType>& data) {throw runtime_error("inferLLM(..) should be implemented for llm task");}
};

#endif // AI_BMT_INTERFACE_H
```

## Step3) Build and Start BMT

**1. Generate the Ninja build system using cmake**

- Run the following command to remove existing cache
  ```bash
  rm -rf CMakeCache.txt CMakeFiles cmake_install.cmake AI_BMT_GUI_Submitter .cmake
  rm -f build.ninja .ninja_deps .ninja_log
  ```
- Run the following command to execute CMake in the current directory (usually the build directory). This command will generate the Ninja build system based on the CMakeLists.txt file located in the parent directory. Once successfully executed, the project will be ready to be built using Ninja.
  ```bash
  cmake -G "Ninja" ..
  ```

**2. Setting Library Path for Executable in Current Directory**

- Run the following command to make the executable(AI_BMT_GUI_Submitter) can reference the libraries located in the lib folder of the current directory.
  ```bash
  export DYLD_LIBRARY_PATH=$(pwd)/lib:$DYLD_LIBRARY_PATH
  ```

**3. Build the project**

- Run the following command to build the project using the build system configured by CMake in the current directory. This will compile the project and create the executable AI_BMT_GUI_Submitter in the build folder.
  ```bash
  cmake --build .
  ```

**4. Start Performance Analysis**

- Run the following command to start created excutable. When the GUI Popup, Click [Start BMT] button to start AI Performance Analysis.
  ```bash
  ./AI_BMT_GUI_Submitter
  ```

**Run all commands at once (For Initial Build)**

```bash
# Install dependencies (run only once)
# brew install cmake ninja

# Build commands
rm -rf CMakeCache.txt CMakeFiles cmake_install.cmake AI_BMT_GUI_Submitter .cmake
rm -f build.ninja .ninja_deps .ninja_log
cmake -G "Ninja" ..
export DYLD_LIBRARY_PATH=$(pwd)/lib:$DYLD_LIBRARY_PATH
cmake --build .
./AI_BMT_GUI_Submitter
```

**Run all commands at once (For Rebuild)**

- Using following commands in `build/` directory.

```bash
rm -rf CMakeCache.txt CMakeFiles cmake_install.cmake AI_BMT_GUI_Submitter .cmake
rm -f build.ninja .ninja_deps .ninja_log
cmake -G "Ninja" ..
export DYLD_LIBRARY_PATH=$(pwd)/lib:$DYLD_LIBRARY_PATH
cmake --build .
./AI_BMT_GUI_Submitter
```

**Execute AI-BMT App**

- Using following commands in `build/` directory.

```bash
export DYLD_LIBRARY_PATH=$(pwd)/lib:$DYLD_LIBRARY_PATH
./AI_BMT_GUI_Submitter
```
