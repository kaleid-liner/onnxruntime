#pragma once

#define GPU_ENERGY_PROFILE
#if defined(USE_CUDA) && defined(GPU_ENERGY_PROFILE)

//////////////////////////////////////////////////////////////////////////////
// Timer.h
// =======
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

#ifndef CUDA_ENERGY_PROFILER_TIMER_H
#define CUDA_ENERGY_PROFILER_TIMER_H
#if defined(WIN32) || defined(_WIN32)   // Windows system specific
#include <windows.h>
#else          // Unix based system specific
#include <sys/time.h>
#endif

#include <stdlib.h>

namespace onnxruntime {

namespace profiling {

class Timer
{
public:
    Timer();                                    // default constructor
    ~Timer();                                   // default destructor

    void   start();                             // start timer
    void   stop();                              // stop the timer
    double getElapsedTime();                    // get elapsed time in second
    double getElapsedTimeInSec();               // get elapsed time in second (same as getElapsedTime)
    double getElapsedTimeInMilliSec();          // get elapsed time in milli-second
    double getElapsedTimeInMicroSec();          // get elapsed time in micro-second


protected:


private:
    double startTimeInMicroSec;                 // starting time in micro-second
    double endTimeInMicroSec;                   // ending time in micro-second
    int    stopped;                             // stop flag 
#if defined(WIN32) || defined(_WIN32)
    LARGE_INTEGER frequency;                    // ticks per second
    LARGE_INTEGER startCount;                   //
    LARGE_INTEGER endCount;                     //
#else
    timeval startCount;                         //
    timeval endCount;                           //
#endif
};

}  // namespace profiling
}  // namespace onnxruntime

#endif // CUDA_ENERGY_PROFILER_TIMER_H


#ifndef CUDA_ENERGY_PROFILER_H
#define CUDA_ENERGY_PROFILER_H

#include <vector>
#include <unordered_map>
#include <memory>

#ifdef USE_CTPL_THREAD_POOL
namespace ctpl
{
  class thread_pool;
}
#else
#include <thread>
#endif

namespace onnxruntime {

namespace profiling {

#define DISALLOW_COPY(TypeName) TypeName(const TypeName&) = delete
#define DISALLOW_ASSIGNMENT(TypeName) TypeName& operator=(const TypeName&) = delete
#define DISALLOW_COPY_AND_ASSIGNMENT(TypeName) \
  DISALLOW_COPY(TypeName);                     \
  DISALLOW_ASSIGNMENT(TypeName)
#define DISALLOW_MOVE(TypeName)     \
  TypeName(TypeName&&) = delete;    \
  TypeName& operator=(TypeName&&) = delete
#define DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TypeName) \
  DISALLOW_COPY_AND_ASSIGNMENT(TypeName);           \
  DISALLOW_MOVE(TypeName)

class Timer;
struct GPUInfoContainer;

class GPUInspector final
{
 public:
  struct GPUInfo_t
  {
    double time_stamp{};
    double used_memory_percent{};
    double power_watt{};
    double temperature{};
    double memory_util{};
    double gpu_util{};
    double energy_since_boot{};
  };

  ~GPUInspector();
  
  static unsigned int NumTotalDevices();
  static unsigned int NumInspectedDevices();
  static void InspectedDeviceIds(std::vector<unsigned int>& device_ids);
  static GPUInfo_t GetGPUInfo(unsigned int gpu_id);

  static void StartInspect();
  static void StopInspect();
  static void ExportReadings(unsigned int gpu_id, std::vector<GPUInfo_t>& readings);
  static void ExportAllReadings(std::unordered_map<unsigned int, std::vector<GPUInfo_t>>& all_readings);

  static double CalculateEnergy(const std::vector<GPUInfo_t>& readings);
  static double CalculateEnergy(unsigned int gpu_id);
  static void CalculateEnergy(std::unordered_map<unsigned int, double>& energies);
  static double GetDurationInSec();

  static void Initialize() { Instance(); }
  static bool Reset(std::vector<unsigned int> gpu_ids = {}, double sampling_interval = 0.05);

 private:
  // sigleton
  DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GPUInspector);
  GPUInspector();
  static GPUInspector& Instance();
  // implementation
  bool _init(std::vector<unsigned int> gpu_ids = {}, double sampling_interval = 0.05);
  void _run();
  void _start_inspect();
  void _stop_inspect();

  bool running_inspect_{false};
  double sampling_interval_micro_second_{0.05 * 1000000};

#ifdef USE_CTPL_THREAD_POOL
  std::unique_ptr<ctpl::thread_pool> pthread_pool_{nullptr};
  void _thread_pool_wait_ready();
#else
  std::shared_ptr<std::thread> pthread_inspect_{nullptr};
#endif

  std::shared_ptr<Timer> timer_{nullptr};
  std::shared_ptr<GPUInfoContainer> recording_container_{nullptr};

};

using GPUInfo_t = GPUInspector::GPUInfo_t;

}  // namespace profiling
}  // namespace onnxruntime

#endif  // CUDA_ENERGY_PROFILER_H

#endif  // #if defined(USE_CUDA) && defined(GPU_ENERGY_PROFILE)