#include <cstdint>
#include <memory.h>
#include <cstdio>
#include <ctime>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>
#include <string>

#define RANDOM_MULTIPLIER 0x5DEECE66DULL
#define RANDOM_ADDEND 0xBULL
#define RANDOM_MASK ((1ULL << 48ULL) - 1ULL)

#define CHUNK_SEED_BOTTOM_4 (CHUNK_SEED & 0xFULL)
#define CHUNK_SEED_BIT_5 ((CHUNK_SEED >> 4ULL) & 1ULL)

#ifndef FLOOR_LEVEL
#define FLOOR_LEVEL 63LL
#endif

#ifndef WANTED_CACTUS_HEIGHT
#define WANTED_CACTUS_HEIGHT 20LL
#endif

#ifndef WORK_UNIT_SIZE
#define WORK_UNIT_SIZE (1ULL << 23ULL)
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256ULL
#endif

#ifndef GPU_COUNT
#define GPU_COUNT 1ULL
#endif

#ifndef OFFSET
#define OFFSET 0
#endif

#ifndef END
#define END (1ULL << 48ULL)
#endif

#ifndef CHUNK_SEED
#define CHUNK_SEED 9567961692053ULL
#endif

#ifndef NEIGHBOR1
#define NEIGHBOR1 856ULL
#endif

#ifndef NEIGHBOR2
#define NEIGHBOR2 344ULL
#endif

#ifndef NEIGHBOR3
#define NEIGHBOR3 840ULL
#endif

#ifndef DIAGONAL_INDEX
#define DIAGONAL_INDEX 0ULL
#endif

#ifndef CACTUS_HEIGHT
#define CACTUS_HEIGHT 12ULL
#endif

namespace java_random {

    // Random::next(bits)
    __device__ inline uint32_t next(uint64_t *random, int32_t bits) {
        *random = (*random * RANDOM_MULTIPLIER + RANDOM_ADDEND) & RANDOM_MASK;
        return (uint32_t) (*random >> (48ULL - bits));
    }

    __device__ inline int32_t next_int_unknown(uint64_t *seed, int16_t bound) {
        if ((bound & -bound) == bound) {
            *seed = (*seed * RANDOM_MULTIPLIER + RANDOM_ADDEND) & RANDOM_MASK;
            return (int32_t) ((bound * (*seed >> 17ULL)) >> 31ULL);
        }

        int32_t bits, value;
        do {
            *seed = (*seed * RANDOM_MULTIPLIER + RANDOM_ADDEND) & RANDOM_MASK;
            bits = *seed >> 17ULL;
            value = bits % bound;
        } while (bits - value + (bound - 1) < 0);
        return value;
    }

    // Random::nextInt(bound)
    __device__ inline uint32_t next_int(uint64_t *random) {
        return java_random::next(random, 31) % 3;
    }

}

__global__ __launch_bounds__(256, 2) void crack(uint64_t seed_offset, int32_t *num_seeds, uint64_t *seeds) {
    uint64_t originalSeed = ((blockIdx.x * blockDim.x + threadIdx.x + seed_offset) << 4ULL) | CHUNK_SEED_BOTTOM_4;
    uint64_t seed = originalSeed;

    int8_t heightMap[1024];

#pragma unroll
    for (int i = 0; i < 1024; i++) {
        heightMap[i] = FLOOR_LEVEL;
    }

    int32_t currentHighestPos = 0, posMap;
    int16_t initialPosX, initialPosY, initialPosZ, initialPos;
    int16_t posX, posY, posZ;

    int16_t i, a, j;
    int8_t position = -1;

    for (i = -90; i < 0; i += 9) {
        // Keep, most threads finish early this way
        if (heightMap[currentHighestPos] - WANTED_CACTUS_HEIGHT - FLOOR_LEVEL < i)
            return;

        initialPosX = java_random::next(&seed, 4) + 8;
        initialPosZ = java_random::next(&seed, 4) + 8;

        initialPos = initialPosX + initialPosZ * 32;

        if (position == -1) {
            if (initialPos == NEIGHBOR1) {
                position = 0;
            } else if (initialPos == NEIGHBOR2) {
                position = 1;
            } else if (initialPos == NEIGHBOR3) {
                position = 2;
            }

            if (position != -1) {
                uint64_t bit = (originalSeed >> 4ULL) & 1ULL;

                if (position != DIAGONAL_INDEX) {
                    if (bit == CHUNK_SEED_BIT_5) return;
                } else {
                    if (bit != CHUNK_SEED_BIT_5) return;
                }

                heightMap[initialPos] += CACTUS_HEIGHT;

                if (heightMap[currentHighestPos] < heightMap[initialPos]) {
                    currentHighestPos = initialPos;
                }
            }
        }

        initialPosY = java_random::next_int_unknown(&seed, (heightMap[initialPosX + initialPosZ * 32] + 1) * 2);

        for (a = 0; a < 10; a++) {
            posX = initialPosX + java_random::next(&seed, 3) - java_random::next(&seed, 3);
            posY = initialPosY + java_random::next(&seed, 2) - java_random::next(&seed, 2);
            posZ = initialPosZ + java_random::next(&seed, 3) - java_random::next(&seed, 3);

            posMap = posX + posZ * 32;

            if (position == -1) {
                if (posMap == NEIGHBOR1) {
                    position = 0;
                } else if (posMap == NEIGHBOR2) {
                    position = 1;
                } else if (posMap == NEIGHBOR3) {
                    position = 2;
                }

                if (position != -1) {
                    uint64_t bit = (originalSeed >> 4ULL) & 1ULL;

                    if (position != DIAGONAL_INDEX) {
                        if (bit == CHUNK_SEED_BIT_5) return;
                    } else {
                        if (bit != CHUNK_SEED_BIT_5) return;
                    }

                    heightMap[posMap] += CACTUS_HEIGHT;

                    if (heightMap[currentHighestPos] < heightMap[posMap]) {
                        currentHighestPos = posMap;
                    }
                }
            }

            // Keep
            if (posY <= heightMap[posMap])
                continue;

            for (j = 0; j < 1 + java_random::next_int_unknown(&seed, java_random::next_int(&seed) + 1); j++) {
                if ((posY + j - 1) > heightMap[posMap] || posY < 0) continue;
                if ((posY + j) <= heightMap[(posX + 1) + posZ * 32]) continue;
                if ((posY + j) <= heightMap[posX + (posZ - 1) * 32]) continue;
                if ((posY + j) <= heightMap[(posX - 1) + posZ * 32]) continue;
                if ((posY + j) <= heightMap[posX + (posZ + 1) * 32]) continue;

                heightMap[posMap]++;

                if (heightMap[currentHighestPos] < heightMap[posMap]) {
                    currentHighestPos = posMap;
                }
            }
        }

        if (heightMap[currentHighestPos] - FLOOR_LEVEL >= WANTED_CACTUS_HEIGHT) {
            uint64_t addend = 0;
            if (position == 0)
                addend = NEIGHBOR1;
            if (position == 1)
                addend = NEIGHBOR2;
            if (position == 2)
                addend = NEIGHBOR3;
            seeds[atomicAdd(num_seeds, 1)] = (addend << 48ULL) | originalSeed;
            return;
        }
    }
}

struct GPU_Node {
    int* num_seeds;
    uint64_t* seeds;
};

void setup_gpu_node(GPU_Node* node, int32_t gpu) {
    cudaSetDevice(gpu);
    cudaMallocManaged(&node->num_seeds, sizeof(*node->num_seeds));
    cudaMallocManaged(&node->seeds, 1ULL << 10ULL); // approx 1kb
}

GPU_Node nodes[GPU_COUNT];
uint64_t offset = OFFSET;
uint64_t count = 0;
std::mutex info_lock;

void gpu_manager(int32_t gpu_index) {
    std::string fileName = "kaktoos_seeds" + std::to_string(gpu_index) + ".txt";
    FILE *out_file = fopen(fileName.c_str(), "a");
    cudaSetDevice(gpu_index);
    while (offset < END) {
        *nodes[gpu_index].num_seeds = 0;
        crack<<<WORK_UNIT_SIZE / BLOCK_SIZE, BLOCK_SIZE, 0>>> (offset, nodes[gpu_index].num_seeds, nodes[gpu_index].seeds);
        info_lock.lock();
        offset += WORK_UNIT_SIZE;
        info_lock.unlock();
        cudaDeviceSynchronize();
        for (int32_t i = 0, e = *nodes[gpu_index].num_seeds; i < e; i++) {
            fprintf(out_file, "%lld\n", (long long int)nodes[gpu_index].seeds[i]);
            printf("Found seed: %lld\n", (long long int)nodes[gpu_index].seeds[i]);
        }
        fflush(out_file);
        info_lock.lock();
        count += *nodes[gpu_index].num_seeds;
        info_lock.unlock();
    }
    fclose(out_file);
}

int main() {
    printf("Searching %ld total seeds...\n", END - OFFSET);

    std::thread threads[GPU_COUNT];

    time_t startTime = time(nullptr), currentTime;
    for(int32_t i = 0; i < GPU_COUNT; i++) {
        setup_gpu_node(&nodes[i], i);
        threads[i] = std::thread(gpu_manager, i);
    }

    using namespace std::chrono_literals;

    while (offset < END) {
        time(&currentTime);
        int timeElapsed = (int)(currentTime - startTime);
        double speed = (double)(offset - OFFSET) / (double)timeElapsed / 1000000.0;
        printf("Searched %lld seeds, offset: %lld found %lld matches. Time elapsed: %ds. Speed: %.2fm seeds/s. %f%%\n",
               (long long int)(offset - OFFSET),
               (long long int)offset,
               (long long int)count,
               timeElapsed,
               speed,
               (double)(offset - OFFSET) / (END - OFFSET) * 100);

        std::this_thread::sleep_for(0.5s);
    }

    for (auto &thread : threads) {
        thread.join();
    }

    printf("Done!\n");
    printf("But, verily, it be the nature of dreams to end.\n");

}