#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <cassert>

#include <onnxruntime_c_api.h>
#include "cuda_energy_profiler.h"
#include "cmdLine.h"

#define PrintVar(x) (std::cout << #x << " = " << x << std::endl)

using std::cout;
using std::endl;
using namespace onnxruntime::profiling;

void DisplayGPUInfo(const GPUInspector::GPUInfo_t& info)
{
    PrintVar(info.time_stamp);
    PrintVar(info.used_memory_percent);
    PrintVar(info.power_watt);
    PrintVar(info.temperature);
    PrintVar(info.memory_util);
    PrintVar(info.gpu_util);
}

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
    std::string strGPUReadingCSV = "gpu_readings.csv";
    std::string strLogTXT = "running_logs.txt";
    std::string strProfileOutput = "";
    unsigned int nNumRepeat = 1;
    bool do_warming_up = false;
    int warm_up_repeat = 1;

    CmdLine cmd;
    cmd.add(make_option('i', strModel, "model"));
    cmd.add(make_option('o', strOptimizedModel, "optimized_model"));
    cmd.add(make_option('g', strGPUReadingCSV, "gpu_readings"));
    cmd.add(make_option('l', strLogTXT, "logs"));
    cmd.add(make_option('r', nNumRepeat, "repeat"));
    cmd.add(make_option('p', strProfileOutput, "profile_output"));
    cmd.add(make_switch('w', "warmup"));
    cmd.add(make_option('x', warm_up_repeat, "warmup_repeat"));
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

    std::cout << "Get GPUInspector Instance." << std::endl;
    GPUInspector& gpu_ins = GPUInspector::Instance();
    gpu_ins.Init(0.01);
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

    // enable cuda
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

    // create session and load model
    OrtSession *session;
    api->CreateSession(env, strModel.c_str(), session_options, &session);
    std::cout << "Load model finished." << std::endl;

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
    std::vector<double> start_cost, stop_cost;
    for(int i = 0; i < 1; i++)
    {
        {
            onnxruntime::profiling::Timer timer;
            timer.start();
            gpu_ins.StartInspect();
            timer.stop();
            start_cost.push_back(timer.getElapsedTimeInMicroSec());
        }

        std::cout << "Run begin." << std::endl;
        for(unsigned int i = 0; i < repeat; i++)
        {
            api->Run(session, NULL, input_names, ort_input_values, n_inputs, output_names, n_outputs, ort_output_values);
        }
        std::cout << "Run finished." << std::endl;

        {
            onnxruntime::profiling::Timer timer;
            timer.start();
            gpu_ins.StopInspect();
            timer.stop();
            stop_cost.push_back(timer.getElapsedTimeInMicroSec());
        }
    }

    // cost stats
    assert(start_cost.size() == stop_cost.size());
    double sum_start = 0, sum_stop = 0;
    std::cout << "timing stats (microseconds):\nstart  stop" << std::endl;
    for(int i = 0; i < start_cost.size(); i++)
    {
        std::cout << start_cost[i] << "  " << stop_cost[i] << std::endl;
        sum_start += start_cost[i];
        sum_stop += stop_cost[i];
    }
    std::cout << "avg_start = " << sum_start / start_cost.size() << " , avg_stop = " << sum_stop / stop_cost.size() << std::endl;

    api->ReleaseSessionOptions(session_options);
    api->ReleaseSession(session);
    api->ReleaseEnv(env);
    // api->ReleaseAllocator(allocator);

    // display log info
    std::cout << "NumDevices = " << gpu_ins.NumDevices() << std::endl;
    PrintVar(repeat);

    // display energy and latency
    std::vector<double> energies;
    gpu_ins.CalculateEnergy(energies);
    std::cout << "energies(in vector) : ";
    for(double item : energies)
    {
        std::cout << item << " ";
    }
    std::cout << std::endl;
    std::cout << "gpu latency = " << gpu_ins.GetDurationInSec() << " second" << std::endl;

    // export latency and energy
    std::ofstream f_log(strLogTXT);
    f_log << "#model,repeat,latency,energy" << std::endl;
    f_log << std::fixed;
    f_log << strModel << "," << repeat << "," << gpu_ins.GetDurationInSec();
    for(double item : energies)
    {
        f_log << "," << item;
    }
    f_log << std::endl;
    f_log.close();

    // export GPU readings
    std::ofstream f_gpu(strGPUReadingCSV);
    f_gpu << "gpu_id,timestamp,used_memory,power,temperature,memory_util,gpu_util" << std::endl;
    f_gpu << std::fixed;
    for(unsigned int gpu_id = 0; gpu_id < gpu_ins.NumDevices(); gpu_id++)
    {
        std::vector<GPUInspector::GPUInfo_t> gpu_readings;
        gpu_ins.ExportReadings(gpu_id, gpu_readings);
        for(const auto& info : gpu_readings)
        {
            f_gpu << gpu_id << "," << info.time_stamp << "," << info.used_memory_percent << "," << info.power_watt << "," 
                << info.temperature << "," << info.memory_util << "," << info.gpu_util << std::endl;
        }
    }
    f_gpu.close();

    // std::ofstream f_runlog("running_logs.txt");
    // f_runlog << "thread_id,gpu_id,start_time,end_time" << std::endl;
    // f_runlog << std::fixed;
    // auto& run_logs = gpu_ins.running_logs_;
    // for(int i = 0; i < run_logs.size(); i++)
    // {
    //     for(auto& item : run_logs[i])
    //     {
    //         f_runlog << item.thread_id << "," << item.gpu_id << "," << item.start_time << "," << item.end_time << std::endl;
    //     }
    // }
    // f_runlog.close();

    return 0;
}
