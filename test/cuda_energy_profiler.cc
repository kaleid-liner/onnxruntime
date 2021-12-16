#include "cuda_energy_profiler.h"

#if defined(USE_CUDA)

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

namespace onnxruntime {

namespace profiling {

GPUInspector& GPUInspector::Instance()
{
    static GPUInspector instance;
    return instance;
}

bool GPUInspector::Init(double sampling_interval)
{
    if(pthread_inspect_)
    {
        running_inspect_ = false;
        pthread_inspect_->join();
        pthread_inspect_ = nullptr;
        recordings_.clear();
    }

    unsigned int deviceCount = 0;
    nvmlDeviceGetCount(&deviceCount);
    devices_.resize(deviceCount);
    for(unsigned int gpu_id = 0; gpu_id < deviceCount; gpu_id++)
    {
        nvmlDeviceGetHandleByIndex(gpu_id, &devices_[gpu_id]);
    }
    recordings_.resize(deviceCount);

    initialized_ = true;
    sampling_interval_micro_second_ = sampling_interval * 1000000;

    // // read GPUInfo once for warming up
    // for(unsigned int gpu_id = 0; gpu_id < deviceCount; gpu_id++)
    // {
    //     GetGPUInfo(devices_[gpu_id]);
    // }
    std::cout << "GPUInspector initialized, numDevices = " << deviceCount << ", sampling interval = " << sampling_interval << std::endl;
    
    return true;
}

GPUInfo_t GPUInspector::GetGPUInfo(const nvmlDevice_t& device)
{
    // CheckInit();
    GPUInfo_t gpu_info;
    gpu_info.time_stamp = 0.0;
    // get momory info
    nvmlMemory_t mem_info;
    nvmlDeviceGetMemoryInfo(device, &mem_info);
    gpu_info.used_memory_percent = static_cast<double>(mem_info.used) / static_cast<double>(mem_info.total) * 100.0;
    // get power usage
    unsigned int power_mw = 0;
    nvmlDeviceGetPowerUsage(device, &power_mw);
    gpu_info.power_watt = static_cast<double>(power_mw) / 1000.0;
    // get temperature
    unsigned int temp;
    nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
    gpu_info.temperature = static_cast<double>(temp);
    // get utils
    nvmlUtilization_t utils;
    nvmlDeviceGetUtilizationRates(device, &utils);
    gpu_info.memory_util = static_cast<double>(utils.memory);
    gpu_info.gpu_util = static_cast<double>(utils.gpu);
    return gpu_info;
}

GPUInfo_t GPUInspector::GetGPUInfo(unsigned int gpu_id)
{
    CheckInit();
    if(gpu_id >= devices_.size())
    {
        throw std::runtime_error("Invalid gpu_id while getting GPU Info.");
    }
    return GetGPUInfo(devices_[gpu_id]);
}

void GPUInspector::StartInspect()
{
    CheckInit();
    pthread_inspect_ = std::shared_ptr<std::thread>(
        new std::thread(&GPUInspector::Run, this));
    while(!running_inspect_)
    {
        std::this_thread::sleep_for(std::chrono::nanoseconds(5));
    }
}

void GPUInspector::StopInspect()
{
    if(!running_inspect_)
    {
        throw std::runtime_error("GPUInspector not started while requested to stop.");
    }
    running_inspect_ = false;
    pthread_inspect_->join();
    pthread_inspect_ = nullptr;
}

void GPUInspector::ExportReadings(unsigned int gpu_id, std::vector<GPUInfo_t>& readings) const
{
    if(running_inspect_)
    {
        throw std::runtime_error("Can not export readings while GPUInspect is running.");
    }
    if(gpu_id < recordings_.size())
    {
        readings = recordings_[gpu_id];
    }
    else
    {
        throw std::runtime_error("Invalid gpu_id while exporting readings.");
    }
}

void GPUInspector::ExportAllReadings(std::vector<std::vector<GPUInfo_t>>& all_readings) const
{
    if(running_inspect_)
    {
        throw std::runtime_error("Can not export readings while GPUInspect is running.");
    }
    all_readings = recordings_;
}

unsigned int GPUInspector::NumDevices() const
{
    CheckInit();
    return devices_.size();
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

double GPUInspector::CalculateEnergy(unsigned int gpu_id) const
{
    if(running_inspect_)
    {
        throw std::runtime_error("Can not calculate energy while GPUInspect is running.");
    }
    if(gpu_id >= devices_.size())
    {
        throw std::runtime_error("Invalid gpu_id while calculating energy.");
    }
    return CalculateEnergy(recordings_[gpu_id]);
}

void GPUInspector::CalculateEnergy(std::vector<double>& energies) const
{
    if(running_inspect_)
    {
        throw std::runtime_error("Can not calculate energy while GPUInspect is running.");
    }
    energies.clear();
    for(unsigned int gpu_id = 0; gpu_id < recordings_.size(); gpu_id++)
    {
        energies.push_back(CalculateEnergy(recordings_[gpu_id]));
    }
}

double GPUInspector::GetDurationInSec()
{
    return timer_.getElapsedTimeInSec();
}

GPUInspector::GPUInspector()
{
    nvmlInit();

    initialized_ = false;
    running_inspect_ = false;
    loop_repeat_ = 1;
    sampling_interval_micro_second_ = 0.05 * 1000000;
    pthread_inspect_ = nullptr;

    // auto init
    Init();
}

GPUInspector::~GPUInspector()
{
    nvmlShutdown();
}

inline void GPUInspector::CheckInit() const
{
    if(!initialized_)
    {
        throw std::runtime_error("GPUInspector not initialized.");
    }
}

void GPUInspector::Run()
{
    recordings_.clear();
    recordings_.resize(devices_.size());
    timer_.start();
    running_inspect_ = true;
    while(running_inspect_)
    {
        Timer local_timer;
        local_timer.start();

        // get readings
        for(unsigned int gpu_id = 0; gpu_id < devices_.size(); gpu_id++)
        {
            GPUInfo_t info = GetGPUInfo(devices_[gpu_id]);
            info.time_stamp = timer_.getElapsedTimeInSec();
            recordings_[gpu_id].push_back(info);
        }
        
        local_timer.stop();
        int sleep_time = static_cast<int>(sampling_interval_micro_second_ - local_timer.getElapsedTimeInMicroSec());
        if(sleep_time > 0)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_time));
        }
        // else
        // {
        //     std::cerr << "Exceeded sampling interval, time cost = " << local_timer.getElapsedTimeInSec() << std::endl;
        // }
    }
    timer_.stop();
}

}  // namespace profiling
}  // namespace onnxruntime

#endif  // #if defined(USE_CUDA)