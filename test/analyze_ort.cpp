#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <cassert>

#include <onnxruntime_c_api.h>
#include "cmdLine.h"

#define PrintVar(x) (std::cout << #x << " = " << x << std::endl)

const OrtTensorTypeAndShapeInfo* GetModelTensorTypeInfo(const OrtApi* api, const OrtSession *session, const size_t index)
{
    OrtTypeInfo *type_info;
    const OrtTensorTypeAndShapeInfo *tensor_info;
    api->SessionGetInputTypeInfo(session, index, &type_info);
    api->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    return tensor_info;
}

void DisplayTensorTypeInfo(const OrtApi* api, const OrtTensorTypeAndShapeInfo *tensor_info)
{
    enum ONNXTensorElementDataType element_type;
    api->GetTensorElementType(tensor_info, &element_type);
    PrintVar(element_type);

    size_t n_dims;
    api->GetDimensionsCount(tensor_info, &n_dims);
    PrintVar(n_dims);

    int64_t *dim_values = new int64_t[n_dims];
    api->GetDimensions(tensor_info, dim_values, n_dims);
    std::cout << "shape : ";
    for(size_t i = 0; i < n_dims; i++)
    {
        std::cout << dim_values[i] << " ";
    }
    std::cout << std::endl;
}

void DisplayModelInfo(const OrtApi* api, const OrtSession *session)
{
    size_t n_inputs, n_outputs;
    api->SessionGetInputCount(session, &n_inputs);
    api->SessionGetOutputCount(session, &n_outputs);
    PrintVar(n_inputs);
    PrintVar(n_outputs);

    OrtAllocator *allocator;
    api->GetAllocatorWithDefaultOptions(&allocator);
    assert(allocator != NULL);

    std::cout << std::endl;

    for(size_t i = 0; i < n_inputs; i++)
    {
        char* name;
        api->SessionGetInputName(session, i, allocator, &name);
        std::cout << "Input #" << i << ": " << name << std::endl;

        OrtTypeInfo *type_info;
        const OrtTensorTypeAndShapeInfo *tensor_info;
        api->SessionGetInputTypeInfo(session, i, &type_info);
        api->CastTypeInfoToTensorInfo(type_info, &tensor_info);
        DisplayTensorTypeInfo(api, tensor_info);
        api->ReleaseTypeInfo(type_info);

        std::cout << std::endl;
    }

    for(size_t i = 0; i < n_outputs; i++)
    {
        char* name;
        api->SessionGetOutputName(session, i, allocator, &name);
        std::cout << "Output #" << i << ": " << name << std::endl;

        OrtTypeInfo *type_info;
        const OrtTensorTypeAndShapeInfo *tensor_info;
        api->SessionGetOutputTypeInfo(session, i, &type_info);
        api->CastTypeInfoToTensorInfo(type_info, &tensor_info);
        DisplayTensorTypeInfo(api, tensor_info);
        api->ReleaseTypeInfo(type_info);

        std::cout << std::endl;
    }
}

int main(int argc, char** argv)
{
    std::string strModel = "";
    std::string strOptimizedModel = "";
    std::string strProfileOutput = "";

    CmdLine cmd;
    cmd.add(make_option('i', strModel, "model"));
    cmd.add(make_option('o', strOptimizedModel, "optimized_model"));
    cmd.add(make_option('p', strProfileOutput, "profile_output"));
    cmd.process(argc, argv);

    if(strModel.empty())
    {
        std::cout << "Input model must be specified with -i" << std::endl;
        return EXIT_FAILURE;
    }

    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    assert(api != NULL);

    // load model
    OrtEnv *env;
    api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "analyze_ort", &env);
    assert(env != NULL);

    OrtSessionOptions *session_options;
    api->CreateSessionOptions(&session_options);
    assert(session_options != NULL);
    api->SetSessionExecutionMode(session_options, ORT_SEQUENTIAL);
    if(!strOptimizedModel.empty())
    {
        std::cout << "Optimized model output set to: " << strOptimizedModel << std::endl;
        api->SetOptimizedModelFilePath(session_options, strOptimizedModel.c_str());
    }
    if(!strProfileOutput.empty())
    {
        std::cout << "Profiling output set to: " << strProfileOutput << std::endl;
        api->EnableProfiling(session_options, strProfileOutput.c_str());
    }

    auto enable_cuda = [&](OrtSessionOptions* session_options)->int {
        // OrtCUDAProviderOptions is a C struct. C programming language doesn't have constructors/destructors.
        OrtCUDAProviderOptions o;
        // Here we use memset to initialize every field of the above data struct to zero.
        memset(&o, 0, sizeof(o));
        // But is zero a valid value for every variable? Not quite. It is not guaranteed. In the other words: does every enum
        // type contain zero? The following line can be omitted because EXHAUSTIVE is mapped to zero in onnxruntime_c_api.h.
        o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        o.gpu_mem_limit = SIZE_MAX;
        OrtStatus* onnx_status = api->SessionOptionsAppendExecutionProvider_CUDA(session_options, &o);
        if (onnx_status != NULL) {
            const char* msg = api->GetErrorMessage(onnx_status);
            std::cerr << msg << std::endl;
            api->ReleaseStatus(onnx_status);
            return -1;
        }
        return 0;
    };

    int ret = enable_cuda(session_options);
    if(ret)
    {
        std::cerr << "CUDA is not available." << std::endl;
    }
    else
    {
        std::cout << "CUDA is enabled." << std::endl;
    }

    OrtSession *session;
    api->CreateSession(env, strModel.c_str(), session_options, &session);

    DisplayModelInfo(api, session);

    // setup input and output
    OrtAllocator *allocator;
    api->GetAllocatorWithDefaultOptions(&allocator);
    assert(allocator != NULL);

    size_t n_inputs = 0, n_outputs = 0;
    api->SessionGetInputCount(session, &n_inputs);
    api->SessionGetOutputCount(session, &n_outputs);
    OrtValue** ort_input_values = new OrtValue*[n_inputs];
    OrtValue** ort_output_values = new OrtValue*[n_outputs];
    char** input_names = new char*[n_inputs];
    char** output_names = new char*[n_outputs];

    for(size_t i = 0; i < n_inputs; i++)
    {
        api->SessionGetInputName(session, i, allocator, &input_names[i]);

        OrtTypeInfo *type_info;
        const OrtTensorTypeAndShapeInfo *tensor_info;
        api->SessionGetInputTypeInfo(session, i, &type_info);
        api->CastTypeInfoToTensorInfo(type_info, &tensor_info);

        enum ONNXTensorElementDataType element_type;
        api->GetTensorElementType(tensor_info, &element_type);
        size_t n_dims;
        api->GetDimensionsCount(tensor_info, &n_dims);
        int64_t *dim_values = new int64_t[n_dims];
        api->GetDimensions(tensor_info, dim_values, n_dims);

        api->CreateTensorAsOrtValue(allocator, dim_values, n_dims, element_type, &ort_input_values[i]);
        api->ReleaseTypeInfo(type_info);
    }
    for(size_t i = 0; i < n_outputs; i++)
    {
        api->SessionGetOutputName(session, i, allocator, &output_names[i]);

        OrtTypeInfo *type_info;
        const OrtTensorTypeAndShapeInfo *tensor_info;
        api->SessionGetOutputTypeInfo(session, i, &type_info);
        api->CastTypeInfoToTensorInfo(type_info, &tensor_info);

        enum ONNXTensorElementDataType element_type;
        api->GetTensorElementType(tensor_info, &element_type);
        size_t n_dims;
        api->GetDimensionsCount(tensor_info, &n_dims);
        int64_t *dim_values = new int64_t[n_dims];
        api->GetDimensions(tensor_info, dim_values, n_dims);

        api->CreateTensorAsOrtValue(allocator, dim_values, n_dims, element_type, &ort_output_values[i]);
        api->ReleaseTypeInfo(type_info);
    }

    // run inference
    std::cout << "Run begin." << std::endl;
    api->Run(session, NULL, input_names, ort_input_values, n_inputs, output_names, n_outputs, ort_output_values);
    std::cout << "Run finished." << std::endl;


    api->ReleaseSessionOptions(session_options);
    api->ReleaseSession(session);
    api->ReleaseEnv(env);
    // api->ReleaseAllocator(allocator);
    
    return 0;
}
