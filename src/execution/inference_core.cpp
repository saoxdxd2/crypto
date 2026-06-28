#include "inference_core.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include <onnxruntime_cxx_api.h>

using namespace std;

// Global Pointers
static Ort::Env* env = nullptr;
static Ort::SessionOptions* session_options = nullptr;
static Ort::Session* lobert_session = nullptr;
static Ort::Session* fincast_session = nullptr;
static Ort::MemoryInfo* memory_info = nullptr;

extern "C" {

int init_models(const char* lobert_path, const char* fincast_path) {
    try {
        env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ThreeBrainEngine");
        session_options = new Ort::SessionOptions();
        memory_info = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
        
        session_options->SetIntraOpNumThreads(1);
        session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        #ifdef _WIN32
        std::string l_path(lobert_path);
        std::string f_path(fincast_path);
        std::wstring w_lobert(l_path.begin(), l_path.end());
        std::wstring w_fincast(f_path.begin(), f_path.end());
        lobert_session = new Ort::Session(*env, w_lobert.c_str(), *session_options);
        fincast_session = new Ort::Session(*env, w_fincast.c_str(), *session_options);
        #else
        lobert_session = new Ort::Session(*env, lobert_path, *session_options);
        fincast_session = new Ort::Session(*env, fincast_path, *session_options);
        #endif

        cout << "[C++ Native] ONNX Models Loaded Successfully." << endl;
        return 0;
    } catch (const Ort::Exception& e) {
        cerr << "[C++ Error] " << e.what() << endl;
        return -1;
    }
}

float run_inference(const float* market_data, int data_size) {
    if (!lobert_session || !memory_info) return 0.0f;

    vector<int64_t> input_shape = {1, 128, 5};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        *memory_info, const_cast<float*>(market_data), data_size, 
        input_shape.data(), input_shape.size()
    );

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    
    try {
        auto output_tensors = lobert_session->Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1
        );

        float* out_arr = output_tensors.front().GetTensorMutableData<float>();
        return out_arr[0];
    } catch (...) {
        return 0.0f;
    }
}

void shutdown_models() {
    delete lobert_session;
    delete fincast_session;
    delete session_options;
    delete memory_info;
    delete env;
    cout << "[C++ Native] Models shut down safely." << endl;
}

} // extern "C"
