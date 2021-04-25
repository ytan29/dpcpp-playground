//==============================================================
// Assignment 1 - Classification
// Source: https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761
// Quicksort implementation adapted from : https://www.programiz.com/dsa/quick-sort
// Bitonicsort implementation adapted from : https://github.com/zjin-lcf/oneAPI-DirectProgramming/tree/master/bitonic-sort-dpct
// =============================================================

#include <CL/sycl.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <ctime>

using namespace sycl;

#define DEBUG 0
#define BLOCK_SIZE 256
#define DEVICE_HOST 0
#define DEVICE_GPU 1
#define DEVICE_CPU 2

static const size_t default_power = 8; // power for Bitonic Sort
static const size_t K = 5; // k-value

/*
Load the data
Initialize K to your chosen number of neighbors
3. For each example in the data
3.1 Calculate the distance between the query example and the current example from the data.
3.2 Add the distance and the index of the example to an ordered collection
4. Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by the distances
5. Pick the first K entries from the sorted collection
6. Get the labels of the selected K entries
7. If regression, return the mean of the K labels
8. If classification, return the mode of the K labels
 */

void ParallelBitonicSort(uint32_t data_gpu[], uint32_t index[], int n, queue &q) {
  // n: the exponent used to set the array size. Array size = power(2, n)
  int size = pow(2, n);
  uint32_t *a = data_gpu;
  uint32_t *b = index;

  // step from 0, 1, 2, ...., n-1
  for (int step = 0; step < n; step++) {
    // for each step s, stage goes s, s-1, ..., 0
    for (int stage = step; stage >= 0; stage--) {
      int seq_len = pow(2, stage + 1);

      // Constant used in the kernel: 2**(step-stage).
      int two_power = 1 << (step - stage);

      // Offload the work to kernel.
      q.submit([&](auto &h) {
        h.parallel_for(range<1>(size), [=](id<1> i) {
          // Assign the bitonic sequence number.
          int seq_num = i / seq_len;

          // Variable used to identified the swapped element.
          int swapped_ele = -1;

          // Because the elements in the first half in the bitonic
          // sequence may swap with elements in the second half,
          // only the first half of elements in each sequence is
          // required (seq_len/2).
          int h_len = seq_len / 2;

          if (i < (seq_len * seq_num) + h_len) swapped_ele = i + h_len;

          // Check whether increasing or decreasing order.
          int odd = seq_num / two_power;

          // Boolean variable used to determine "increasing" or
          // "decreasing" order.
          bool increasing = ((odd % 2) == 0);

          // Swap the elements in the bitonic sequence if needed
          if (swapped_ele != -1) {
            if (((a[i] > a[swapped_ele]) && increasing) ||
                ((a[i] < a[swapped_ele]) && !increasing)) {
              uint32_t temp = a[i];
              a[i] = a[swapped_ele];
              a[swapped_ele] = temp;
                
              uint32_t temp_index = b[i];
              b[i] = b[swapped_ele];
              b[swapped_ele] = temp_index;
            }
          }
        });
      });
      q.wait();
    }  // end stage
  }    // end step
}

class custom_selector : public device_selector {
public:
    custom_selector(int type_id): typeID_(type_id){};
    int operator()(const device& dev) const override {
    int rating = 0;
    
    switch(typeID_){
        case 1:
            // looking for GPU
            if (dev.is_gpu())
                rating = 3;
            break;
        case 2:
            // Looking for CPU
            if (dev.is_cpu())
                rating = 3;
            break;
        default:
            rating = 0;
    }
    return rating;
    };
    
private:
    int typeID_;
};

int main(int argc, char* argv[]) {  
    int opt;
    int power = default_power;
    int N = 0;
    int device = DEVICE_CPU;
    std::time_t start, end;

    while((opt = getopt(argc, argv, "p:d:")) != -1) {
        switch(opt)
        {
            case 'd':
                if(strcmp("gpu", optarg)==0) {
                    device = DEVICE_GPU;
                    std::cout << "Using GPU for DPCPP" << std::endl;
                } else {
                    device = DEVICE_CPU;
                    std::cout << "Using CPU for DPCPP" << std::endl;
                }
                break;
            case 'p':
                power = atoi(optarg);
                break;
            case ':':
                std::cout << "Missing argument for -d" << std::endl;
            case '?':
            default:
                std::cout << "Use -d [cpu|gpu] to determine which device custom selector will choose" << std::endl;
                std::cout << "Use -p <power_int_value> to determine N (2 ^ power)" << std::endl;
                return -1;
        }
    }

    custom_selector selector(device);
    queue q(selector);
    
    N = 1 << power;
    
    std::cout << "Device : " << q.get_device().get_info<info::device::name>() << std::endl;
    std::cout << "N : " << N << " = 2^" << power << std::endl;
    srand(time(NULL));
    
    std::cout << "N size : " << N << std::endl;
    
    uint32_t *sample_index = static_cast<uint32_t *>(malloc_shared(N * sizeof(uint32_t), q));
    uint32_t *sample_age = static_cast<uint32_t *>(malloc_shared(N * sizeof(uint32_t), q));
    uint32_t *sample_choice = static_cast<uint32_t *>(malloc_shared(N * sizeof(uint32_t), q));
    uint32_t *calc_distance = static_cast<uint32_t *>(malloc_shared(N * sizeof(uint32_t), q));
    
    /* randomly generated data */
    for (int i = 0; i < N; i++) {
        sample_index[i] = i;
        sample_age[i] = rand() % 70 + 20; // age 20 - 90
        sample_choice[i] = rand() % 2; 
    }
    
    /* Random input */
    uint32_t input_age = rand() % 70 + 20;
    printf("Random Age: %u  \n", input_age);
    
    /* DPCPP - Parallel For */
    q.parallel_for(range<1>(N), [=](id<1> i) {
        calc_distance[i] = abs((int)sample_age[i] - (int)input_age);
    }).wait();

#if DEBUG
    printf("Before Sort:\n");
    /* Sorted distance */
    for (int i = 0; i < 5; i++) {
        printf("Age: %u, Choice: %d, Dist: %d\n", sample_age[sample_index[i]], sample_choice[sample_index[i]], calc_distance[i]);
    }
#endif
    start = std::time(nullptr);
    
    /* Using bitonicsort on the distance */
    ParallelBitonicSort(calc_distance, sample_index, power, q);
    
    end = std::time(nullptr);
    printf("Start: %ld End: %ld Spent: %ld\n", start, end, (end - start));
    
#if DEBUG
    printf("After Sort:\n");
    /* Sorted distance */
    for (int i = 0; i < 5; i++) {
        printf("Age: %u, Choice: %d, Dist: %d\n", sample_age[sample_index[i]], sample_choice[sample_index[i]], calc_distance[i]);
    }
    
    printf("calc_distance distribution...\n");
    uint32_t temp = calc_distance[0];
    int count = 0;
    for (int i = 0; i < N; i++) {
        if (temp != calc_distance[i]) {
            printf("Dist: %d, count = %d\n", temp, count);
            temp = calc_distance[i];
            count = 1;
        } else {
            count++;
        }
    }
#endif
    
    /* For classification, return mode of top 5 */
    printf("Shortlisted sample_choice...\n");
    int favor_cnt = 0;
    int unfavor_cnt = 0;
    for (int i = 0; i < K; i++) {
        printf("%d: Choice: %d\n", sample_index[i], sample_choice[sample_index[i]]);
        if (sample_choice[sample_index[i]])
            favor_cnt++;
        else
            unfavor_cnt++;
    }
    
    /* Result */
    printf("With Age %u, we predict the person %slikes pineapple on pizza\n", input_age, (favor_cnt > unfavor_cnt)? "" : "dis");
    
    printf("Start: %ld End: %ld Spent: %ld\n", start, end, (end - start));
    
    free(sample_age, q);
    free(sample_choice, q);
    free(calc_distance, q);
    
    return 0;
}
