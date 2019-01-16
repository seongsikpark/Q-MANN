#ifndef __DEFINE__
#define __DEFINE__

//#define APPX_ATT_MAT_MUL // test later

//#define APPX_ATT    // approximate attention
//#define BIN_ATT    // binary attention
//#define QUANT_ATT    // quantized attention

// Attention Mode
// 1: normal (default)
// 2: quantized (fixed point)
// 3: approximate (proposed)
// 4: binary
#define ATTENTION_MODE 2

// fixed point
//#define BW_IWL 2
//#define BW_FRAC 5
//#define BW_WL 1+BW_IWL+BW_FRAC
#define BW_WL 8

// number of bits for appx_att
#define NUM_BIT_ATTENTION BW_WL

// hamming weight parameter - w = 2^(k+1-(n+hamming_weight_para))
#define HAMMING_WEIGHT_PARA 0
//#define HAMMING_WEIGHT_PARA -1

// fixed point
#define EN_FIXED_POINT true
//#define EN_FIXED_POINT false

// enalbe quantization mode
//#define EN_QUANT_MODE
#ifdef EN_QUANT_MODE
    // Quantization mode for fixed point operation
    // 0: round down
    // 1: round up
    // 2: round to nearest even
    // 3: round towards zero (default)
    // default: truncation -> to use this mode -> undefine EN_QUANT_MODE
    #define QUANT_MODE 3
#else
    // default: truncation
    #define QUANT_MODE 3
#endif

// cuda debug
//#define CUDA_DEBUG true
#define CUDA_DEBUG false

// shift based softmax
//#define EN_SHIFT_BASED_SM true
#define EN_SHIFT_BASED_SM false

// scale layer enable - softmax in attention module
//#define EN_SC_ATT true
#define EN_SC_ATT false

// attention constant scale - only in proposed method
//#define ATTENTION_CONST_SCALE 1.0
//#define ATTENTION_CONST_SCALE 0.25
//#define ATTENTION_CONST_SCALE 0.125

// 
#define ATTENTION_CONST_SCALE (-3)

// similarity analysis(softmax input)
//#define EN_SIMILARITY_ANALYSIS true
#define EN_SIMILARITY_ANALYSIS false

// save best trained model & load for test, ealry stopping
//
//#define EN_SAVE_BEST_MODEL true
#define EN_SAVE_BEST_MODEL false

// MQ: Memory controller Qunatization control
#define EN_MQ true
//#define EN_MQ false

#define COUNT_EARLY_STOPPING 5

// testing..
// one side(internal representation binarization)
// descrive about this mode !
//#define BINARY_MODE true
#define BINARY_MODE false

// enable gradient quantization
//#define EN_GRAD_QUANT

// HW MODE
// 1 : CPU only
// 2 : CPU - GPU(model, instruction level parallel)
// 21: CPU - GPU verification
// 3 : CPU - FPGA
#define HW_MODE 2

#if HW_MODE == 1
    #define EN_CPU true
    #define EN_GPU_MODEL false
    #define EN_FPGA false
#elif HW_MODE == 2
    #define EN_CPU false
    #define EN_GPU_MODEL true
    #define EN_FPGA false
#elif HW_MODE == 21
    #define EN_CPU true
    #define EN_GPU_MODEL true
    #define EN_FPGA false
#elif HW_MODE == 3
    #define EN_CPU false
    #define EN_GPU_MODEL false
    #define EN_FPGA true
#endif

#define EN_TRAIN true
#define EN_LOAD_WEIGHT false
#define EN_WRITE_WEIGHT false

#define MAX_LINE_LEN_SYS 100    // max length of a line - only used for parsing samples
#define MAX_WORD_LEN 20         // max length of a word - number of character in a word
#define MAX_LINE_LEN 7          // max length of a line - number of words in a line

// FPGA setting for each bAbI task
/*
#define DIM_FORCED false
#define EN_JOINT false
#define MAX_DICT_LEN 64         // max length of dictionary - number of words in dictionary (DIM_IN)
#define MAX_SEN_LEN 64          // max length of sentences in a sample(memory) - number of sentences(DIM_MEM)
#define DIM_EMB 16
*/

// FPGA setting for joint bAbI task
/*
#define DIM_FORCED false
#define EN_JOINT true
#define MAX_DICT_LEN 192        // max length of dictionary - number of words in dictionary (DIM_IN)
#define MAX_SEN_LEN 64          // max length of sentences in a sample(memory) - number of sentences(DIM_MEM)
#define DIM_EMB 32

#define NUM_DICT_SAMPLE 200000        // number of dict sample

#define EN_NUM_SAMPLE_FROM_FILE false //
#define NUM_SAMPLE 200000        // number of train sample
#define NUM_SAMPLE_TEST 20000    // number of test sample
*/

// same defalut setting for each bAbI task w/ "End-To-End Memory Networks"
#define DIM_FORCED false
#define EN_JOINT false
#define MAX_DICT_LEN 64         // max length of dictionary - number of words in dictionary (DIM_IN)
#define MAX_SEN_LEN 50          // max length of sentences in a sample(memory) - number of sentences(DIM_MEM)
//#define DIM_EMB 4
//#define DIM_EMB 20
//#define DIM_EMB 40
//#define DIM_EMB 50
#define DIM_EMB 60
//#define DIM_EMB 70
//#define DIM_EMB 80
//#define DIM_EMB 100
//#define DIM_EMB 150
//#define DIM_EMB 200
//#define DIM_EMB 400
//#define DIM_EMB 1000

#define EN_NUM_SAMPLE_FROM_FILE true //
#define NUM_DICT_SAMPLE 10000        // number of train sample
#define NUM_SAMPLE 10000        // number of train sample
#define NUM_SAMPLE_TEST 1000    // number of test sample
#define EN_SAMPLE_SHUFFLED false


// setting for joint bAbI task
/*
#define DIM_FORCED true
#define EN_JOINT true
#define MAX_DICT_LEN 192        // max length of dictionary - number of words in dictionary (DIM_IN)
#define MAX_SEN_LEN 64          // max length of sentences in a sample(memory) - number of sentences(DIM_MEM)
#define DIM_EMB 60

#define NUM_DICT_SAMPLE 200000        // number of dict sample

#define EN_NUM_SAMPLE_FROM_FILE true //
#define NUM_SAMPLE 200000        // number of train sample
#define NUM_SAMPLE_TEST 1000    // number of test sample

#define EN_SAMPLE_SHUFFLED true
*/

//
#define RATE_NUM_VALID_SAMPLE 0.1          // rate valid samples in training samples

// time encoding
#define EN_TIME true


// enable cosine similarity, if it is undefined, then using dot product as sim.
//#define EN_COSINE_SIM

//////////////////////////////////////////////////
// regularizer
//////////////////////////////////////////////////
// gradient clipping
#define EN_MAX_GRAD_L2_NORM true
//#define MAX_GRAD_L2_NORM 20.0
#define MAX_GRAD_L2_NORM 40.0
//#define MAX_GRAD_L2_NORM 60.0
//#define MAX_GRAD_L2_NORM 80.0

// random noise - time
//#define RAND_NOISE_TIME 0.1
#define RAND_NOISE_TIME 0.0

// linear start
//#define EN_LINEAR_START true
#define EN_LINEAR_START false

#define NUM_ITR_LINEAR_START 5

//////////////////////////////////////////////////
//
//////////////////////////////////////////////////
#define SIZE_BATCH 32


//////////////////////////////////////////////////
//
//////////////////////////////////////////////////

#define NULL_CHAR "NULL"


#if ATTENTION_MODE == 1
    // Normal attention
    //#define LAMBDA 0.0001
    #define LAMBDA 0.0000

    #define RATE_DECAY_STEP 25
    #define LEARNING_RATE 0.3
    #define NUM_ITR 100
    #define NUM_HOP 3

    //#define VERBOSE_DEBUG_TEST true
    #define VERBOSE_DEBUG_TEST false
#elif ATTENTION_MODE == 2
    // Quantized attention
    #define LAMBDA 0.000000

    #define RATE_DECAY_STEP 25
    #define LEARNING_RATE 0.3
    #define NUM_ITR 100
    #define NUM_HOP 3

    //#define VERBOSE_DEBUG_TEST true
    #define VERBOSE_DEBUG_TEST false
#elif ATTENTION_MODE == 3
    // Approximate attention
    #define LAMBDA 0.000000

    #define RATE_DECAY_STEP 25
    #define LEARNING_RATE 0.3
    #define NUM_ITR 100
    #define NUM_HOP 3

    //#define VERBOSE_DEBUG_TEST true
    #define VERBOSE_DEBUG_TEST false
#elif ATTENTION_MODE == 4
    #define LAMBDA 0.000000

    #define RATE_DECAY_STEP 20
    #define LEARNING_RATE 1.0
    #define NUM_ITR 100
    #define NUM_HOP 3

    //#define VERBOSE_DEBUG_TEST true
    #define VERBOSE_DEBUG_TEST false
#endif




// type of weight tying
// 1. adjacent
// 2. layer-wise(RNN)
#define TYPE_WEIGHT_TYING 2

// linear mapping
//#define EN_LINEAR_MAPPING false
#define EN_LINEAR_MAPPING true

// non-linearity for layer-wise weight tying
#define EN_NON_LINEARITY false
//#define EN_NON_LINEARITY true

// use position encoding
#define EN_PE false
//#define EN_PE true


#define VERBOSE true
//
#define VERBOSE_DEBUG false
//#define VERBOSE_DEBUG true


// test
//#define TEST_MAXOUT

// zeroing_null_weight
//#define ZEROING_NULL_WEIGHT false
#define ZEROING_NULL_WEIGHT true

#define EN_EXP_TABLE_BASED false


// sample bin packet out for fpga accelerator
#define EN_SAMPLE_BIN_OUT false

//
//#define PATH_DATA_SET ".\/dataset\/en_1k_parsed\/"
#define PATH_DATA_SET ".\/dataset\/en_10k_parsed\/"
#define NUM_LIST_DATA 21
//static const char *LIST_DATA_SET[] = {
static const char *LIST_DATA_SET[NUM_LIST_DATA] = {
    "qa1_single-supporting-fact",
    "qa2_two-supporting-facts",
    "qa3_three-supporting-facts",
    "qa4_two-arg-relations",
    "qa5_three-arg-relations",
    "qa6_yes-no-questions",
    "qa7_counting",
    "qa8_lists-sets",
    "qa9_simple-negation",
    "qa10_indefinite-knowledge",
    "qa11_basic-coreference",
    "qa12_conjunction",
    "qa13_compound-coreference",
    "qa14_time-reasoning",
    "qa15_basic-deduction",
    "qa16_basic-induction",
    "qa17_positional-reasoning",
    "qa18_size-reasoning",
    "qa19_path-finding",
    "qa20_agents-motivations",
    "qa_joint"
};

//////////////////////////////////////////////////
// for FPGA
//////////////////////////////////////////////////
// configuration for hex dump
#define NUM_BYTE 4

// configuration for packet
#define NUM_BIT_TYPE 4
#define NUM_BIT_ADDR 12

#define TYPE_TRAIN_SEN          0x8
#define TYPE_TRAIN_SEN_DONE     0x9
#define TYPE_TRAIN_QUEST        0xa
#define TYPE_TRAIN_QUEST_DONE   0xb
#define TYPE_TRAIN_ANS          0xc
#define TYPE_TRAIN_ANS_DONE     0xd

#define TYPE_TEST_SEN           0x0
#define TYPE_TEST_SEN_DONE      0x1
#define TYPE_TEST_QUEST         0x2
#define TYPE_TEST_QUEST_DONE    0x3
#define TYPE_TEST_ANS           0x4
#define TYPE_TEST_ANS_DONE      0x5


//
#define NUM_LAYER 10
#define NUM_LAYER_OP 7

#endif

