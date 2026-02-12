#include <stdio.h>
#include <cuda_runtime.h>

// n초 동안 GPU를 점유하는 커널
__global__ void gpu_busy_wait(unsigned long long wait_cycles) {
    unsigned long long start = clock64();
    unsigned long long now;

    do {
        now = clock64();
    } while ((now - start) < wait_cycles);
}

int main(int argc, char *argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // n초 설정 (기본값 3초)
    int seconds = 3;
    if (argc > 1) {
        seconds = atoi(argv[1]);
    }

    // 1. GPU 정보 출력
    printf("=== GPU 정보 ===\n");
    printf("Device ID          : %d\n", device);
    printf("GPU 이름           : %s\n", prop.name);
    printf("Compute Capability : %d.%d\n", prop.major, prop.minor);
    printf("SM 개수             : %d\n", prop.multiProcessorCount);
    printf("GPU 클럭            : %.2f MHz\n\n", prop.clockRate / 1000.0);

    // 2. n초 동안 GPU 사용
    unsigned long long wait_cycles =
        (unsigned long long)seconds *
        (unsigned long long)prop.clockRate * 1000ULL;

    dim3 blocks(prop.multiProcessorCount); // SM 수만큼
    dim3 threads(256);

    printf("=== GPU 사용 시작 ===\n");
    for (int i = 0; i < seconds; i++) {
        gpu_busy_wait<<<blocks, threads>>>(prop.clockRate * 1000ULL);
        cudaDeviceSynchronize();
        printf("  %d / %d 초 경과\n", i + 1, seconds);
    }
    printf("사용 시간          : %d 초\n", seconds);
    printf("블록 수            : %d\n", blocks.x);
    printf("블록당 스레드 수   : %d\n", threads.x);
    printf("총 스레드 수       : %d\n\n", blocks.x * threads.x);

    gpu_busy_wait<<<blocks, threads>>>(wait_cycles);
    cudaDeviceSynchronize();

    // 3. GPU 사용 관련 정보 출력
    printf("=== GPU 사용 종료 ===\n");

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);

    printf("GPU 메모리 상태:\n");
    printf("  사용 가능 메모리 : %.2f MB\n", freeMem / (1024.0 * 1024.0));
    printf("  전체 메모리      : %.2f MB\n", totalMem / (1024.0 * 1024.0));

    return 0;
}

