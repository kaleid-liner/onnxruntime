#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

using std::cout;
using std::endl;

using namespace std::chrono_literals;

class ThreadPool
{
public:

    ThreadPool(int num_threads = 5)
    {
        std::cout << "Hardware Concurrency = " << std::thread::hardware_concurrency() << std::endl;
        if(num_threads < 0)
        {
            num_threads = std::thread::hardware_concurrency() - 1;
        }
        for(int i = 0; i < num_threads; i++)
        {
            threads.push_back(std::thread(&ThreadPool::InfiniteLoop, this));
        }
    }

    void StartRun()
    {
        {
            std::unique_lock<std::mutex> lock(mMutexThread);
            mbRun = true;
        }
        cv.notify_one();
    }

    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(mMutexThread);
            mbTerminate = true;
        }
        cv.notify_all();
        for(std::thread& th : threads)
        {
            if(th.joinable())
            {
                th.join();
            }
        }
    }

private:

    void InfiniteLoop()
    {
        {
            std::unique_lock<std::mutex> lock(mMutexThread);
            std::cout << "Thread #" << std::this_thread::get_id() << " start." << std::endl;
        }

        while(true)
        {
            {
                std::unique_lock<std::mutex> lock(mMutexThread);
                cv.wait(lock, [this](){ return mbRun || mbTerminate; });
                if(mbTerminate)
                {
                    break;
                }
            }
            Run();
            std::this_thread::sleep_for(50ms);
        }

        {
            std::unique_lock<std::mutex> lock(mMutexThread);
            std::cout << "Thread #" << std::this_thread::get_id() << " exit." << std::endl;
        }
    }

    void Run()
    {
        std::unique_lock<std::mutex> lock(mMutexThread);
        std::cout << "Thread #" << std::this_thread::get_id() << " is running." << std::endl;
        mbRun = false;
    }

    bool mbRun{false};
    bool mbTerminate{false};
    std::mutex mMutexThread;
    std::condition_variable cv;
    std::vector<std::thread> threads;
};


int main(int argc, char** argv)
{
    if(argc != 2)
    {
        return 1;
    }
    int num_threads = std::stoi(argv[1]);
    ThreadPool pool(num_threads);
    std::this_thread::sleep_for(1s);

    for(int i = 0; i < 20; i++)
    {
        pool.StartRun();
        std::this_thread::sleep_for(1s);
    }

    return 0;
}
