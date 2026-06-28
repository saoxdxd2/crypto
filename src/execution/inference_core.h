#ifndef INFERENCE_CORE_H
#define INFERENCE_CORE_H

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Returns 0 on success, -1 on failure
EXPORT_API int init_models(const char* lobert_path, const char* fincast_path);

// Runs the ONNX LOBERT model on 128x5 market data
// Returns the NET EDGE signal.
EXPORT_API float run_inference(const float* market_data, int data_size);

// Cleanup resources
EXPORT_API void shutdown_models();

#ifdef __cplusplus
}
#endif

#endif // INFERENCE_CORE_H
