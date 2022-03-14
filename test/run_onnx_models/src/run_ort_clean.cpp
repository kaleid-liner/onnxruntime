#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <cassert>
#include <limits>

#include <onnxruntime_c_api.h>
#include <cuda_energy_profiler.h>
#include "cmdLine.h"

using std::cout;
using std::endl;
using namespace onnxruntime::profiling;

// const OrtTensorTypeAndShapeInfo* GetModelTensorTypeInfo(const OrtApi* api, const OrtSession *session, const size_t index)
// {
//     OrtTypeInfo *type_info;
//     const OrtTensorTypeAndShapeInfo *tensor_info;
//     api->SessionGetInputTypeInfo(session, index, &type_info);
//     api->CastTypeInfoToTensorInfo(type_info, &tensor_info);
//     return tensor_info;
// }

int main(int argc, char** argv)
{
    std::string strModel = "";
    unsigned int nNumRepeat = 1;
    bool do_warming_up = false;
    int warm_up_repeat = 1;
    double time_limit = std::numeric_limits<double>::max();

    CmdLine cmd;
    cmd.add(make_option('i', strModel, "model"));
    cmd.add(make_option('r', nNumRepeat, "repeat"));
    cmd.add(make_switch('w', "warmup"));
    cmd.add(make_option('x', warm_up_repeat, "warmup_repeat"));
    cmd.add(make_option('t', time_limit, "time_limit"));
    cmd.process(argc, argv);

    if(strModel.empty())
    {
        std::cerr << "Input model must be specified with -i" << std::endl;
        return EXIT_FAILURE;
    }

    if(cmd.used('w'))
    {
        do_warming_up = true;
        std::cout << "warming up enabled." << std::endl;
        if(warm_up_repeat <= 0)
        {
            std::cerr << "warm_up_repeat <= 0 (current value: " << warm_up_repeat << "), using default value 1." << std::endl;
            warm_up_repeat = 1;
        }
    }

    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    assert(api != NULL);

    GPUInspector& gpu_ins = GPUInspector::Instance();
    gpu_ins.SetLoopRepeat(nNumRepeat);

    // create environment
    OrtEnv *env;
    api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "run_ort", &env);
    assert(env != NULL);

    // create session options
    OrtSessionOptions *session_options;
    api->CreateSessionOptions(&session_options);
    assert(session_options != NULL);
    api->SetSessionExecutionMode(session_options, ORT_SEQUENTIAL);

    // enable cuda
    auto enable_cuda = [&](OrtSessionOptions* session_options)->int {
        // OrtCUDAProviderOptions is a C struct. C programming language doesn't have constructors/destructors.
        OrtCUDAProviderOptions o;
        // Here we use memset to initialize every field of the above data struct to zero.
        memset(&o, 0, sizeof(o));
        // But is zero a valid value for every variable? Not quite. It is not guaranteed. In the other words: does every enum
        // type contain zero? The following line can be omitted because EXHAUSTIVE is mapped to zero in onnxruntime_c_api.h.
        o.device_id = 0;
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
        std::cout << "CUDA is not available." << std::endl;
    }
    else
    {
        std::cout << "CUDA is enabled." << std::endl;
    }

    // create session and load model
    OrtSession *session;
    api->CreateSession(env, strModel.c_str(), session_options, &session);

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

    // warming up
    if(do_warming_up)
    {
        std::cout << "Warming up begin." << std::endl;
        for(int i = 0; i < warm_up_repeat; i++)
        {
            api->Run(session, NULL, input_names, ort_input_values, n_inputs, output_names, n_outputs, ort_output_values);
        }
        std::cout << "Warming up finished." << std::endl;
    }

    // run inference
    unsigned int repeat = gpu_ins.GetLoopRepeat();
    onnxruntime::profiling::Timer run_timer;
    std::cout << "Run begin." << std::endl;
    gpu_ins.StartInspect();
    run_timer.start();
    for(unsigned int i = 0; i < repeat; i++)
    {
        api->Run(session, NULL, input_names, ort_input_values, n_inputs, output_names, n_outputs, ort_output_values);
        double time_elapsed = run_timer.getElapsedTimeInSec();
        if(time_elapsed >= time_limit)
        {
            std::cout << "early stop due to exceeding time limit, actual repeat times: " << i + 1 << std::endl;
            gpu_ins.SetLoopRepeat(i + 1);
            repeat = i + 1;
        }
    }
    run_timer.stop();
    gpu_ins.StopInspect();
    std::cout << "Run finished." << std::endl;

    api->ReleaseSessionOptions(session_options);
    api->ReleaseSession(session);
    api->ReleaseEnv(env);
    // api->ReleaseAllocator(allocator);

    // display energy and latency
    std::cout << "#latency:" << gpu_ins.GetDurationInSec() / repeat << ",energy:" << gpu_ins.CalculateEnergy(0) / repeat << std::endl;

    return 0;
}
