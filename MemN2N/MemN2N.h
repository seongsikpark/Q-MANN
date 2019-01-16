#include "common.h"
#include "layer.h"
#include "sample.h"
#include "define.h"

#include <pthread.h>

bool en_gpu_model = EN_GPU_MODEL;
bool en_fpga = EN_FPGA;
bool en_cpu = EN_CPU;

typedef struct {
    FILE *f_file;
    sample *p_sam;
    float err_test;
    unsigned int num_sample_test;
} arg_stream_io;

// host -> fpga
void *stream_write(void *arg);
// fpga -> host
void *stream_read(void* arg);
