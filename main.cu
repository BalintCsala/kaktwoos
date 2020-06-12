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
#define WANTED_CACTUS_HEIGHT 18LL
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
#define END (1ULL << 44ULL)
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

#ifdef DEBUG
// Magic values, don't touch
__device__ int64_t DEBUG_ARR[]{
        0, 18, 90, 20, 9, 308, -1, 79, 16, 78, 11, 368, -1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 76,
        10, 344, 1, 0, 12, 344, 12, 3, 12, 0, 0, 0, 0, 13, 13, 13, 13, 0, 0, 0, 0, 14, 14, 14, 14, 0, 0, 0, 0, 15, 15,
        15, 17, 80, 11, 369, 0, 1, 0, 0, 0, 0, 0, 16, 76, 3, 112, 0, 1, 0, 0, 0, 0, 0, 21, 78, 11, 373, 0, 2, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 20, 82, 10, 340, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 79, 10, 344, 15, 3, 15, 0, 0, 0,
        0, 16, 16, 16, 16, 0, 0, 0, 0, 17, 17, 17, 17, 0, 0, 0, 0, 18, 18, 18, 17, 80, 6, 209, 0, 1, 0, 0, 0, 0, 0, 22,
        81, 10, 342, 0, 1, 0, 0, 0, 0, 0, 22, 77, 8, 278, 0, 1, 0, 0, 0, 0, 0, 18, 344, 96827469838241317
};

#ifdef EXTRACT
#define ASSERT(k, val) printf("%d %lld\n", k++, (long long)val)
#else
#define ASSERT(k, val) if (DEBUG_ARR[k++] != val) printf("Error at %d, expected %lld, got %lld\n", __LINE__, (long long)DEBUG_ARR[k - 1], (long long)val)
#endif
#else
#define ASSERT(k, val)
#endif

/*inline __device__ int8_t extract(int8_t heightMap[], int32_t id) {
    return (*((int16_t*)(heightMap + ((id * 6U) >> 3U))) >> ((id * 6U) & 0b111U)) & 0b111111U;
}

inline __device__ void increase(int8_t heightMap[], int32_t id, int8_t val) {
    *((int16_t*)(heightMap + ((id * 6) >> 3U))) += val << ((id * 6) & 0b111U);
}*/

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

__global__ __launch_bounds__(BLOCK_SIZE, 2) void crack(uint64_t seed_offset, int32_t *num_seeds, uint64_t *seeds) {
#ifdef DEBUG
    int32_t debug_index = 0;
    uint64_t originalSeed = 77849775653ULL;
#else
    uint64_t originalSeed = ((blockIdx.x * blockDim.x + threadIdx.x + seed_offset) << 4ULL) | CHUNK_SEED_BOTTOM_4;
#endif
    uint64_t seed = originalSeed;

    int8_t heightMap[1024];

#pragma unroll
    for (int i = 0; i < 1024; i++) {
        heightMap[i] = 0;
    }

    int32_t currentHighestPos = 0, posMap;
    int16_t initialPosX, initialPosY, initialPosZ, initialPos;
    int16_t posX, posY, posZ;

    int8_t position = -1;

    for (int32_t i = 0; i < 10; i++) {
        ASSERT(debug_index, i);
        ASSERT(debug_index, WANTED_CACTUS_HEIGHT - heightMap[currentHighestPos]);
        ASSERT(debug_index, 9 * (10 - i));
        if (WANTED_CACTUS_HEIGHT - heightMap[currentHighestPos] > 9 * (10 - i))
            return;

        initialPosX = java_random::next(&seed, 4) + 8;
        initialPosZ = java_random::next(&seed, 4) + 8;
        initialPos = initialPosX + initialPosZ * 32;
        ASSERT(debug_index, initialPosX);
        ASSERT(debug_index, initialPosZ);
        ASSERT(debug_index, initialPos);

        if (position == -1) {
            if (initialPos == NEIGHBOR1) {
                position = 0;
            } else if (initialPos == NEIGHBOR2) {
                position = 1;
            } else if (initialPos == NEIGHBOR3) {
                position = 2;
            }
            ASSERT(debug_index, position);

            if (position != -1) {
                uint64_t bit = (originalSeed >> 4ULL) & 1ULL;
                ASSERT(debug_index, bit);

                if (position != DIAGONAL_INDEX) {
                    if (bit == CHUNK_SEED_BIT_5) return;
                } else {
                    if (bit != CHUNK_SEED_BIT_5) return;
                }

                heightMap[initialPos] += CACTUS_HEIGHT;
                ASSERT(debug_index, heightMap[initialPos]);

                if (heightMap[currentHighestPos] < heightMap[initialPos]) {
                    currentHighestPos = initialPos;
                    ASSERT(debug_index, currentHighestPos);
                }
            }
        }

        initialPosY = java_random::next_int_unknown(&seed, (heightMap[initialPos] + FLOOR_LEVEL + 1) * 2);
        ASSERT(debug_index, initialPosY);

        for (int32_t a = 0; a < 10; a++) {
            posX = initialPosX + java_random::next(&seed, 3) - java_random::next(&seed, 3);
            posY = initialPosY + java_random::next(&seed, 2) - java_random::next(&seed, 2);
            posZ = initialPosZ + java_random::next(&seed, 3) - java_random::next(&seed, 3);
            posMap = posX + posZ * 32;
            ASSERT(debug_index, posX);
            ASSERT(debug_index, posY);
            ASSERT(debug_index, posZ);
            ASSERT(debug_index, posMap);

            if (position == -1) {
                if (posMap == NEIGHBOR1) {
                    position = 0;
                } else if (posMap == NEIGHBOR2) {
                    position = 1;
                } else if (posMap == NEIGHBOR3) {
                    position = 2;
                }
                ASSERT(debug_index, position);

                if (position != -1) {
                    uint64_t bit = (originalSeed >> 4ULL) & 1ULL;
                    ASSERT(debug_index, bit);

                    if (position != DIAGONAL_INDEX) {
                        if (bit == CHUNK_SEED_BIT_5) return;
                    } else {
                        if (bit != CHUNK_SEED_BIT_5) return;
                    }

                    heightMap[posMap] += CACTUS_HEIGHT;
                    ASSERT(debug_index, heightMap[posMap]);

                    if (heightMap[currentHighestPos] < heightMap[posMap]) {
                        currentHighestPos = posMap;
                        ASSERT(debug_index, currentHighestPos);
                    }
                }
            }

            ASSERT(debug_index, heightMap[posMap]);
            if (posY <= heightMap[posMap] + FLOOR_LEVEL)
                continue;

            int32_t offset = 1 + java_random::next_int_unknown(&seed, java_random::next_int(&seed) + 1);
            ASSERT(debug_index, offset);

            for (int32_t j = 0; j < offset; j++) {
                ASSERT(debug_index, heightMap[posMap]);
                ASSERT(debug_index, heightMap[(posX + 1) + posZ * 32]);
                ASSERT(debug_index, heightMap[(posX - 1) + posZ * 32]);
                ASSERT(debug_index, heightMap[posX + (posZ + 1) * 32]);
                ASSERT(debug_index, heightMap[posX + (posZ - 1) * 32]);
                if ((posY + j - 1) > heightMap[posMap] + FLOOR_LEVEL || posY < 0) continue;
                if ((posY + j) <= heightMap[(posX + 1) + posZ * 32] + FLOOR_LEVEL) continue;
                if ((posY + j) <= heightMap[(posX - 1) + posZ * 32] + FLOOR_LEVEL) continue;
                if ((posY + j) <= heightMap[posX + (posZ + 1) * 32] + FLOOR_LEVEL) continue;
                if ((posY + j) <= heightMap[posX + (posZ - 1) * 32] + FLOOR_LEVEL) continue;

                heightMap[posMap]++;
                ASSERT(debug_index, heightMap[posMap]);

                ASSERT(debug_index, heightMap[currentHighestPos]);
                ASSERT(debug_index, heightMap[posMap]);
                if (heightMap[currentHighestPos] < heightMap[posMap]) {
                    currentHighestPos = posMap;
                    ASSERT(debug_index, currentHighestPos);
                }
            }
        }

        ASSERT(debug_index, heightMap[currentHighestPos]);
        if (heightMap[currentHighestPos] >= WANTED_CACTUS_HEIGHT) {
            uint64_t neighbor = 0;
            if (position == 0)
                neighbor = NEIGHBOR1;
            if (position == 1)
                neighbor = NEIGHBOR2;
            if (position == 2)
                neighbor = NEIGHBOR3;
            ASSERT(debug_index, neighbor);
            seeds[atomicAdd(num_seeds, 1)] = (neighbor << 48ULL) | originalSeed;
            ASSERT(debug_index, ((neighbor << 48ULL) | originalSeed));
            return;
        }
    }
}

struct GPU_Node {
    int *num_seeds;
    uint64_t *seeds;
};

void setup_gpu_node(GPU_Node *node, int32_t gpu) {
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
    FILE *out_file = fopen(fileName.c_str(), "w");
    cudaSetDevice(gpu_index);
    while (offset < END) {
        *nodes[gpu_index].num_seeds = 0;
        crack<<<WORK_UNIT_SIZE / BLOCK_SIZE, BLOCK_SIZE, 0>>>(offset, nodes[gpu_index].num_seeds,
                                                              nodes[gpu_index].seeds);
        info_lock.lock();
        offset += WORK_UNIT_SIZE;
        info_lock.unlock();
        cudaDeviceSynchronize();
        for (int32_t i = 0, e = *nodes[gpu_index].num_seeds; i < e; i++) {
#ifndef DEBUG
            fprintf(out_file, "%llu %llu\n", nodes[gpu_index].seeds[i] & RANDOM_MASK, (unsigned long long)nodes[gpu_index].seeds[i] >> 48ULL);
            printf("Found seed: %lld\n", (long long int)nodes[gpu_index].seeds[i]);
#endif
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
    for (int32_t i = 0; i < GPU_COUNT; i++) {
        setup_gpu_node(&nodes[i], i);
        threads[i] = std::thread(gpu_manager, i);
    }

    using namespace std::chrono_literals;

    while (offset < END) {
        time(&currentTime);
        int timeElapsed = (int) (currentTime - startTime);
        double speed = (double) (offset - OFFSET) / (double) timeElapsed / 1000000.0;
        printf("Searched %lld seeds, offset: %lld found %lld matches. Time elapsed: %ds. Speed: %.2fm seeds/s. %f%%\n",
               (long long int) (offset - OFFSET),
               (long long int) offset,
               (long long int) count,
               timeElapsed,
               speed,
               (double) (offset - OFFSET) / (END - OFFSET) * 100);

        std::this_thread::sleep_for(0.5s);
    }

    for (auto &thread : threads) {
        thread.join();
    }

    printf("Done!\n");
    printf("But, verily, it be the nature of dreams to end.\n");

}