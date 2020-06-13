#include <cstdint>
#include <memory.h>
#include <cstdio>
#include <ctime>
#include <thread>
#include <vector>
#include <mutex>
#include <chrono>
#include <string>
#include <iostream>

#define RANDOM_MULTIPLIER 0x5DEECE66DULL
#define RANDOM_ADDEND 0xBULL
#define RANDOM_MASK ((1ULL << 48ULL) - 1ULL)
#define WORK_UNIT_SIZE (1ULL << 23ULL)

#ifndef WANTED_CACTUS_HEIGHT
#define WANTED_CACTUS_HEIGHT 18LL
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256ULL
#endif

#ifndef GPU_COUNT
#define GPU_COUNT 1ULL
#endif

#ifndef FLOOR_LEVEL
#define FLOOR_LEVEL 63LL
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

inline __device__ uint8_t extract(uint32_t heightMap[], int32_t id) {
    return (heightMap[id / 5] >> ((id % 5) * 6)) & 0b111111U;
}

inline __device__ void increase(uint32_t heightMap[], int32_t id, int8_t val) {
    heightMap[id / 5] += val << ((id % 5) * 6);
}

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

__global__ __launch_bounds__(BLOCK_SIZE, 2)
void crack(uint64_t seed_offset, int32_t *num_seeds, uint64_t *seeds, uint8_t chunkSeedBottom4Bits, int16_t neighbor1,
           int16_t neighbor2, int16_t neighbor3, int16_t diagonalIndex, uint8_t chunkSeedBit5, uint8_t cactusHeight) {
#ifdef DEBUG
    int32_t debug_index = 0;
    uint64_t originalSeed = 77849775653ULL;
#else
    uint64_t originalSeed = ((blockIdx.x * blockDim.x + threadIdx.x + seed_offset) << 4ULL) | chunkSeedBottom4Bits;
#endif
    uint64_t seed = originalSeed;

    uint32_t heightMap[205];

#pragma unroll
    for (int i = 0; i < 205; i += 2) {
        *(int64_t *) (heightMap + i) = 0;
    }

    int32_t currentHighestPos = 0, posMap;
    int16_t initialPosX, initialPosY, initialPosZ, initialPos;
    int16_t posX, posY, posZ;

    int8_t position = -1;

    for (int32_t i = 0; i < 10; i++) {
        ASSERT(debug_index, i);
        ASSERT(debug_index, WANTED_CACTUS_HEIGHT - extract(heightMap, currentHighestPos));
        ASSERT(debug_index, 9 * (10 - i));
        if (WANTED_CACTUS_HEIGHT - extract(heightMap, currentHighestPos) > 9 * (10 - i))
            return;

        initialPosX = java_random::next(&seed, 4) + 8;
        initialPosZ = java_random::next(&seed, 4) + 8;
        initialPos = initialPosX + initialPosZ * 32;
        ASSERT(debug_index, initialPosX);
        ASSERT(debug_index, initialPosZ);
        ASSERT(debug_index, initialPos);

        if (position == -1) {
            if (initialPos == neighbor1) {
                position = 0;
            } else if (initialPos == neighbor2) {
                position = 1;
            } else if (initialPos == neighbor3) {
                position = 2;
            }
            ASSERT(debug_index, position);

            if (position != -1) {
                uint64_t bit = (originalSeed >> 4ULL) & 1ULL;
                ASSERT(debug_index, bit);

                if (position != diagonalIndex) {
                    if (bit == chunkSeedBit5) return;
                } else {
                    if (bit != chunkSeedBit5) return;
                }

                increase(heightMap, initialPos, cactusHeight);
                ASSERT(debug_index, extract(heightMap, initialPos));

                if (extract(heightMap, currentHighestPos) < extract(heightMap, initialPos)) {
                    currentHighestPos = initialPos;
                    ASSERT(debug_index, currentHighestPos);
                }
            }
        }

        initialPosY = java_random::next_int_unknown(&seed,
                                                    ((int32_t) extract(heightMap, initialPos) + FLOOR_LEVEL + 1) * 2);
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
                if (posMap == neighbor1) {
                    position = 0;
                } else if (posMap == neighbor2) {
                    position = 1;
                } else if (posMap == neighbor3) {
                    position = 2;
                }
                ASSERT(debug_index, position);

                if (position != -1) {
                    uint64_t bit = (originalSeed >> 4ULL) & 1ULL;
                    ASSERT(debug_index, bit);

                    if (position != diagonalIndex) {
                        if (bit == chunkSeedBit5) return;
                    } else {
                        if (bit != chunkSeedBit5) return;
                    }

                    increase(heightMap, posMap, cactusHeight);
                    ASSERT(debug_index, extract(heightMap, posMap));

                    if (extract(heightMap, currentHighestPos) < extract(heightMap, posMap)) {
                        currentHighestPos = posMap;
                        ASSERT(debug_index, currentHighestPos);
                    }
                }
            }

            ASSERT(debug_index, extract(heightMap, posMap));
            if (posY <= extract(heightMap, posMap) + FLOOR_LEVEL)
                continue;

            int32_t offset = 1 + java_random::next_int_unknown(&seed, java_random::next_int(&seed) + 1);
            ASSERT(debug_index, offset);

            for (int32_t j = 0; j < offset; j++) {
                ASSERT(debug_index, extract(heightMap, posMap));
                ASSERT(debug_index, extract(heightMap, (posX + 1) + posZ * 32));
                ASSERT(debug_index, extract(heightMap, (posX - 1) + posZ * 32));
                ASSERT(debug_index, extract(heightMap, posX + (posZ + 1) * 32));
                ASSERT(debug_index, extract(heightMap, posX + (posZ - 1) * 32));
                if ((posY + j - 1) > extract(heightMap, posMap) + FLOOR_LEVEL || posY < 0) continue;
                if ((posY + j) <= extract(heightMap, (posX + 1) + posZ * 32) + FLOOR_LEVEL) continue;
                if ((posY + j) <= extract(heightMap, (posX - 1) + posZ * 32) + FLOOR_LEVEL) continue;
                if ((posY + j) <= extract(heightMap, posX + (posZ + 1) * 32) + FLOOR_LEVEL) continue;
                if ((posY + j) <= extract(heightMap, posX + (posZ - 1) * 32) + FLOOR_LEVEL) continue;

                increase(heightMap, posMap, 1);
                ASSERT(debug_index, extract(heightMap, posMap));

                ASSERT(debug_index, extract(heightMap, currentHighestPos));
                ASSERT(debug_index, extract(heightMap, posMap));
                if (extract(heightMap, currentHighestPos) < extract(heightMap, posMap)) {
                    currentHighestPos = posMap;
                    ASSERT(debug_index, currentHighestPos);
                }
            }
        }

        ASSERT(debug_index, extract(heightMap, currentHighestPos));
        if (extract(heightMap, currentHighestPos) >= WANTED_CACTUS_HEIGHT) {
            uint64_t neighbor = 0;
            if (position == 0)
                neighbor = neighbor1;
            if (position == 1)
                neighbor = neighbor2;
            if (position == 2)
                neighbor = neighbor3;
            ASSERT(debug_index, neighbor);
            seeds[atomicAdd(num_seeds, 1)] =
                    ((uint64_t) extract(heightMap, currentHighestPos) << 58ULL) |
                    ((uint64_t) neighbor << 48ULL) |
                    originalSeed;
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
uint64_t chunkSeed;
uint8_t chunkSeedBottom4Bits;
uint8_t chunkSeedBit5;
int16_t neighbor1;
int16_t neighbor2;
int16_t neighbor3;
int16_t diagonalIndex;
uint8_t cactusHeight;
uint64_t start;
uint64_t end;

uint64_t offset;
uint64_t count = 0;
std::mutex info_lock;

using namespace std::chrono_literals;

void gpu_manager(int32_t gpuIndex) {
    std::string fileName = "kaktoos_seeds" + std::to_string(gpuIndex) + ".txt";
    FILE *out_file = fopen(fileName.c_str(), "w");
    cudaSetDevice(gpuIndex);
    while (offset < end) {
        *nodes[gpuIndex].num_seeds = 0;
        crack<<<WORK_UNIT_SIZE / BLOCK_SIZE, BLOCK_SIZE, 0>>>(
                offset,
                nodes[gpuIndex].num_seeds,
                nodes[gpuIndex].seeds,
                chunkSeedBottom4Bits,
                neighbor1,
                neighbor2,
                neighbor3,
                diagonalIndex,
                chunkSeedBit5,
                cactusHeight);
        info_lock.lock();
        offset += WORK_UNIT_SIZE;
        info_lock.unlock();
        cudaDeviceSynchronize();
        for (int32_t i = 0, e = *nodes[gpuIndex].num_seeds; i < e; i++) {
#ifndef DEBUG
            std::cerr << nodes[gpuIndex].seeds[i] << std::endl;
            printf("Found seed: %llu, height: %llu\n",
                   nodes[gpuIndex].seeds[i] & RANDOM_MASK,
                   (nodes[gpuIndex].seeds[i] >> 58ULL) & 63ULL);
#endif
        }
        fflush(out_file);
        info_lock.lock();
        count += *nodes[gpuIndex].num_seeds;
        info_lock.unlock();
        std::this_thread::sleep_for(0.01s);
    }
    fclose(out_file);
}

int main(int argc, char **argv) {
    printf("Searching %lld total seeds...\n", (long long) (end - start));

    int32_t gpuIndex;

    if (argc < 10) {
        throw std::invalid_argument("Not enough arguments!");
    } else {
        try {
            gpuIndex = std::stoi(argv[1]);
            start = std::stoull(argv[2]);
            end = std::stoull(argv[3]);
            chunkSeed = std::stoull(argv[4]);
            chunkSeedBottom4Bits = chunkSeed & 15U;
            chunkSeedBit5 = (chunkSeed >> 4U) & 1U;
            neighbor1 = std::stoi(argv[5]);
            neighbor2 = std::stoi(argv[6]);
            neighbor3 = std::stoi(argv[7]);
            diagonalIndex = std::stoi(argv[8]);
            cactusHeight = std::stoi(argv[9]);
            std::cout << "Received new work unit: " << chunkSeed << std::endl;
            std::cout <<
                      "Data: n1: " << neighbor1 <<
                      ", n2: " << neighbor2 <<
                      ", n3: " << neighbor3 <<
                      ", di: " << diagonalIndex <<
                      ", ch: " << (int) cactusHeight << std::endl;
        } catch (std::invalid_argument const &ex) {
            throw std::invalid_argument("Invalid argument");
        } catch (std::out_of_range const &ex) {
            throw std::invalid_argument("Invalid number size");
        }
    }

    setup_gpu_node(&nodes[0], gpuIndex);
    std::thread thr(gpu_manager, gpuIndex);
    time_t startTime = time(nullptr), currentTime;

    while (offset < end) {
        std::this_thread::sleep_for(1s);
        time(&currentTime);
        int timeElapsed = (int) (currentTime - startTime);
        double speed = (double) (offset - start) / (double) timeElapsed / 1000000.0;
        printf("Speed: %.2fm/s, Block: %.5f%%, Block time: %llds\n",
               speed,
               (double) (offset - start) / (double) (end - start) * 100,
               (long long) timeElapsed);
    }

    thr.join();

    std::cout << "Finished work unit" << std::endl;

}