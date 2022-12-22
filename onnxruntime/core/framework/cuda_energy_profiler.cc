#define GPU_ENERGY_PROFILE
#if defined(USE_CUDA) && defined(GPU_ENERGY_PROFILE)

#include "cuda_energy_profiler.h"

namespace onnxruntime {

namespace profiling {

//////////////////////////////////////////////////////////////////////////////
// Timer.cpp
// =========
// High Resolution Timer.
// This timer is able to measure the elapsed time with 1 micro-second accuracy
// in both Windows, Linux and Unix system 
//
//  AUTHOR: Song Ho Ahn (song.ahn@gmail.com)
// CREATED: 2003-01-13
// UPDATED: 2017-03-30
//
// Copyright (c) 2003 Song Ho Ahn
//////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
// constructor
///////////////////////////////////////////////////////////////////////////////
Timer::Timer()
{
#if defined(WIN32) || defined(_WIN32)
    QueryPerformanceFrequency(&frequency);
    startCount.QuadPart = 0;
    endCount.QuadPart = 0;
#else
    startCount.tv_sec = startCount.tv_usec = 0;
    endCount.tv_sec = endCount.tv_usec = 0;
#endif

    stopped = 0;
    startTimeInMicroSec = 0;
    endTimeInMicroSec = 0;
}

///////////////////////////////////////////////////////////////////////////////
// distructor
///////////////////////////////////////////////////////////////////////////////
Timer::~Timer()
{
}

///////////////////////////////////////////////////////////////////////////////
// start timer.
// startCount will be set at this point.
///////////////////////////////////////////////////////////////////////////////
void Timer::start()
{
    stopped = 0; // reset stop flag
#if defined(WIN32) || defined(_WIN32)
    QueryPerformanceCounter(&startCount);
#else
    gettimeofday(&startCount, NULL);
#endif
}

///////////////////////////////////////////////////////////////////////////////
// stop the timer.
// endCount will be set at this point.
///////////////////////////////////////////////////////////////////////////////
void Timer::stop()
{
    stopped = 1; // set timer stopped flag

#if defined(WIN32) || defined(_WIN32)
    QueryPerformanceCounter(&endCount);
#else
    gettimeofday(&endCount, NULL);
#endif
}

///////////////////////////////////////////////////////////////////////////////
// compute elapsed time in micro-second resolution.
// other getElapsedTime will call this first, then convert to correspond resolution.
///////////////////////////////////////////////////////////////////////////////
double Timer::getElapsedTimeInMicroSec()
{
#if defined(WIN32) || defined(_WIN32)
    if(!stopped)
        QueryPerformanceCounter(&endCount);

    startTimeInMicroSec = startCount.QuadPart * (1000000.0 / frequency.QuadPart);
    endTimeInMicroSec = endCount.QuadPart * (1000000.0 / frequency.QuadPart);
#else
    if(!stopped)
        gettimeofday(&endCount, NULL);

    startTimeInMicroSec = (startCount.tv_sec * 1000000.0) + startCount.tv_usec;
    endTimeInMicroSec = (endCount.tv_sec * 1000000.0) + endCount.tv_usec;
#endif

    return endTimeInMicroSec - startTimeInMicroSec;
}

///////////////////////////////////////////////////////////////////////////////
// divide elapsedTimeInMicroSec by 1000
///////////////////////////////////////////////////////////////////////////////
double Timer::getElapsedTimeInMilliSec()
{
    return this->getElapsedTimeInMicroSec() * 0.001;
}

///////////////////////////////////////////////////////////////////////////////
// divide elapsedTimeInMicroSec by 1000000
///////////////////////////////////////////////////////////////////////////////
double Timer::getElapsedTimeInSec()
{
    return this->getElapsedTimeInMicroSec() * 0.000001;
}

///////////////////////////////////////////////////////////////////////////////
// same as getElapsedTimeInSec()
///////////////////////////////////////////////////////////////////////////////
double Timer::getElapsedTime()
{
    return this->getElapsedTimeInSec();
}

}  // namespace profiling
}  // namespace onnxruntime


#include <stdexcept>
#include <iostream>
#include <numeric>
#include <nvml.h>

#ifdef USE_CTPL_THREAD_POOL
#include "ctpl_stl.h"
#endif

namespace onnxruntime {

namespace profiling {


struct GPUInfoContainer
{
    std::unordered_map<unsigned int, nvmlDevice_t> devices;
    std::unordered_map<unsigned int, std::vector<GPUInfo_t>> recordings;
    void ClearAll()
    {
        devices.clear();
        recordings.clear();
    }
    void ClearRecords()
    {
        recordings.clear();
    }
};


static GPUInfo_t GetGPUInfoImpl(const nvmlDevice_t device)
{
    GPUInfo_t gpu_info;
    gpu_info.time_stamp = 0.0;
    // get momory info
    nvmlMemory_t mem_info;
    nvmlDeviceGetMemoryInfo(device, &mem_info);
    gpu_info.used_memory_percent = static_cast<double>(mem_info.used) / static_cast<double>(mem_info.total) * 100.0;
    // get power usage
    unsigned int power_mw = 0;
    nvmlDeviceGetPowerUsage(device, &power_mw);
    gpu_info.power_watt = static_cast<double>(power_mw) * 1.0e-3;
    // get temperature
    unsigned int temp;
    nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
    gpu_info.temperature = static_cast<double>(temp);
    // get utils
    nvmlUtilization_t utils;
    nvmlDeviceGetUtilizationRates(device, &utils);
    gpu_info.memory_util = static_cast<double>(utils.memory);
    gpu_info.gpu_util = static_cast<double>(utils.gpu);
    // energy since the driver was last reloaded
    unsigned long long energy_mj = 0;
    nvmlDeviceGetTotalEnergyConsumption(device, &energy_mj);
    gpu_info.energy_since_boot = static_cast<double>(energy_mj) * 1.0e-3;
    return gpu_info;
}


GPUInspector::GPUInspector()
{
    timer_ = std::make_shared<Timer>();
    recording_container_ = std::make_shared<GPUInfoContainer>();
    nvmlInit();
    _init();
}

GPUInspector::~GPUInspector()
{
    nvmlShutdown();
}

GPUInspector& GPUInspector::Instance()
{
    static GPUInspector instance;
    return instance;
}

unsigned int GPUInspector::NumTotalDevices() 
{
    Instance();
    unsigned int deviceCount = 0;
    nvmlDeviceGetCount(&deviceCount);
    return deviceCount;
}

unsigned int GPUInspector::NumInspectedDevices()
{
    return Instance().recording_container_->devices.size();
}

void GPUInspector::InspectedDeviceIds(std::vector<unsigned int>& device_ids)
{
    const auto & devices = Instance().recording_container_->devices;
    device_ids.clear();
    for(const auto& it : devices)
    {
        device_ids.push_back(it.first);
    }
}

GPUInfo_t GPUInspector::GetGPUInfo(unsigned int gpu_id)
{
    Instance();
    unsigned int deviceCount = 0;
    nvmlDeviceGetCount(&deviceCount);
    if(gpu_id >= deviceCount)
    {
        throw std::runtime_error("Invalid GPU Id while getting GPU Info.");
    }
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(gpu_id, &device);
    return GetGPUInfoImpl(device);
}

void GPUInspector::StartInspect()
{
    Instance()._start_inspect();
}

void GPUInspector::StopInspect()
{
    Instance()._stop_inspect();
}

void GPUInspector::ExportReadings(unsigned int gpu_id, std::vector<GPUInfo_t>& readings)
{
    if(Instance().running_inspect_)
    {
        throw std::runtime_error("Can not export readings while GPUInspect is running.");
    }
    const auto & recordings = Instance().recording_container_->recordings;
    if(recordings.count(gpu_id))
    {
        readings = recordings.at(gpu_id);
    }
    else
    {
        throw std::runtime_error("Invalid GPU Id while exporting readings.");
    }
}

void GPUInspector::ExportAllReadings(std::unordered_map<unsigned int, std::vector<GPUInfo_t>>& all_readings)
{
    if(Instance().running_inspect_)
    {
        throw std::runtime_error("Can not export readings while GPUInspect is running.");
    }
    all_readings = Instance().recording_container_->recordings;
}

double GPUInspector::CalculateEnergy(const std::vector<GPUInfo_t>& readings)
{
    double result = 0.0;
    for(size_t i = 1; i < readings.size(); i++)
    {
        double dt = readings[i].time_stamp - readings[i - 1].time_stamp;
        result += 0.5 * (readings[i - 1].power_watt + readings[i].power_watt) * dt;
    }
    return result;
}

double GPUInspector::CalculateEnergy(unsigned int gpu_id)
{
    if(Instance().running_inspect_)
    {
        throw std::runtime_error("Can not calculate energy while GPUInspect is running.");
    }
    const auto & recordings = Instance().recording_container_->recordings;
    if(recordings.count(gpu_id) == 0)
    {
        throw std::runtime_error("Invalid GPU Id while calculating energy.");
    }
    return CalculateEnergy(recordings.at(gpu_id));
}

void GPUInspector::CalculateEnergy(std::unordered_map<unsigned int, double>& energies)
{
    if(Instance().running_inspect_)
    {
        throw std::runtime_error("Can not calculate energy while GPUInspect is running.");
    }
    const auto & recordings = Instance().recording_container_->recordings;
    energies.clear();
    for(const auto& it : recordings)
    {
        energies[it.first] = CalculateEnergy(it.second);
    }
}

double GPUInspector::GetDurationInSec()
{
    return Instance().timer_->getElapsedTimeInSec();
}

bool GPUInspector::Reset(std::vector<unsigned int> gpu_ids, double sampling_interval)
{
    return Instance()._init(gpu_ids, sampling_interval);
}

bool GPUInspector::_init(std::vector<unsigned int> gpu_ids, double sampling_interval)
{
    if(running_inspect_)
    {
        _stop_inspect();
    }

    recording_container_->ClearAll();

    // get device handle
    unsigned int deviceCount = 0;
    nvmlDeviceGetCount(&deviceCount);
    nvmlDevice_t deviceHandle;
    if(gpu_ids.empty())
    {
        gpu_ids.resize(deviceCount);
        std::iota(gpu_ids.begin(), gpu_ids.end(), 0);
    }
    for(unsigned int id : gpu_ids)
    {
        if(id >= deviceCount)
        {
            std::cerr << "Error: invalid GPU Id for initialization." << std::endl;
            return false;
        }
        nvmlDeviceGetHandleByIndex(id, &deviceHandle);
        recording_container_->devices[id] = deviceHandle;
    }

    // sampling interval
    sampling_interval_micro_second_ = sampling_interval * 1000000;

    // thread
#ifdef USE_CTPL_THREAD_POOL
    int n_threads = 1;
    pthread_pool_ = std::unique_ptr<ctpl::thread_pool>(new ctpl::thread_pool(n_threads));
    _thread_pool_wait_ready();
#endif

    std::cout << "GPUInspector initialized, numDevices = " << deviceCount << ", sampling interval = " << sampling_interval << std::endl;
    std::cout << "Inspected GPU Id:";
    for(unsigned int id : gpu_ids)
    {
        std::cout << " " << id;
    }
    std::cout << std::endl;

    return true;
}

void GPUInspector::_run()
{
    recording_container_->ClearRecords();
    timer_->start();
    running_inspect_ = true;

    Timer local_timer;
    while(running_inspect_)
    {
        local_timer.start();

        // get readings
        for(const auto& it : recording_container_->devices)
        {
            GPUInfo_t info = GetGPUInfoImpl(it.second);
            info.time_stamp = timer_->getElapsedTimeInSec();
            recording_container_->recordings[it.first].push_back(info);
        }
        
        local_timer.stop();
        int sleep_time = static_cast<int>(sampling_interval_micro_second_ - local_timer.getElapsedTimeInMicroSec());
        if(sleep_time > 0)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_time));
        }
    }

    timer_->stop();
    running_inspect_ = false;
}

void GPUInspector::_start_inspect()
{
#ifdef USE_CTPL_THREAD_POOL
    auto run_func = [this](int thread_id) { _run(); };
    pthread_pool_->push(run_func);
#else
    pthread_inspect_ = std::shared_ptr<std::thread>(
        new std::thread(&GPUInspector::_run, this));
#endif
    while(!running_inspect_)
    {
        std::this_thread::sleep_for(std::chrono::nanoseconds(5));
    }
}

void GPUInspector::_stop_inspect()
{
    if(!running_inspect_)
    {
        std::cerr << "GPUInspector not started while requested to stop." << std::endl;
        return;
    }
    running_inspect_ = false;
#ifdef USE_CTPL_THREAD_POOL
    _thread_pool_wait_ready();
#else
    pthread_inspect_->join();
    pthread_inspect_ = nullptr;
#endif
}

#ifdef USE_CTPL_THREAD_POOL

void GPUInspector::_thread_pool_wait_ready()
{
    if(!pthread_pool_) return;
    while(pthread_pool_->n_idle() != pthread_pool_->size())
    {
        std::this_thread::sleep_for(std::chrono::nanoseconds(5));
    }
}

#endif


}  // namespace profiling
}  // namespace onnxruntime

#endif  // #if defined(USE_CUDA) && defined(GPU_ENERGY_PROFILE)