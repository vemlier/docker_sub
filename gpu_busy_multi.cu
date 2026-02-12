#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <cuda_runtime.h>

typedef struct {
    int device;
    int seconds;
} gpu_task_t;

__global__ void gpu_busy_wait(unsigned long long wait_cycles) {
    unsigned long long start = clock64();
    while ((clock64() - start) < wait_cycles);
}

void* gpu_worker(void* arg) {
    gpu_task_t* task = (gpu_task_t*)arg;
    int dev = task->device;
    int seconds = task->seconds;

    cudaSetDevice(dev);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    printf("[GPU %d] %s 시작 (%d초)\n", dev, prop.name, seconds);

    dim3 blocks(prop.multiProcessorCount);
    dim3 threads(256);

    unsigned long long one_sec =
        (unsigned long long)prop.clockRate * 1000ULL;

    for (int i = 0; i < seconds; i++) {
        gpu_busy_wait<<<blocks, threads>>>(one_sec);
        cudaDeviceSynchronize();
        printf("[GPU %d] %d / %d 초 경과\n", dev, i + 1, seconds);
    }

    printf("[GPU %d] 종료\n", dev);
    return NULL;
}

int main(int argc, char *argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);

    int seconds = 3;
    if (argc > 1) seconds = atoi(argv[1]);

    int gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);

    printf("감지된 GPU 수: %d\n", gpu_count);
    if (gpu_count == 0) return 0;

    pthread_t threads[gpu_count];
    gpu_task_t tasks[gpu_count];

    for (int i = 0; i < gpu_count; i++) {
        tasks[i].device = i;
        tasks[i].seconds = seconds;
        pthread_create(&threads[i], NULL, gpu_worker, &tasks[i]);
    }

    for (int i = 0; i < gpu_count; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("모든 GPU 작업 종료\n");
    return 0;
}

