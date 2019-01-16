#ifndef __SAMPLE__
#define __SAMPLE__
#include "common.h"
#include "define.h"

typedef char* word;
typedef float bin;

typedef struct {
    unsigned int n;
    word words[MAX_LINE_LEN_SYS];
} line;

typedef struct {
    unsigned int n;
    int *words;
} line_i;

typedef struct {
    unsigned int n;
    bin *words;
} line_b;

typedef struct {
    unsigned int n;
    word words[MAX_DICT_LEN];
} dictionary;

typedef struct {
    unsigned short addr : NUM_BIT_ADDR, type : NUM_BIT_TYPE;
} packet_16;

typedef struct {
    packet_16 packet[2];
} words_32;

typedef struct {
    unsigned int n_sen;
    unsigned int dim_input;
    unsigned int dim_word;
    unsigned int max_word;
    unsigned int dim_dict;

    line *sentences;
    line question;
    line answer;

    // index
    line_i *sentences_i;
    line_i question_i;
    line_i answer_i;
    
    // bag of words
    bin **sentences_b;
    bin *question_b;
    bin *answer_b;
    
    // binary index packet 16
    packet_16 **sentences_i_p16;
    packet_16 *question_i_p16;
    packet_16 *answer_i_p16;

} sample;


sample* sample_constructor(char *file_name, unsigned int max_len, unsigned int *num_sample, unsigned int num_sample_define);
int sample_init(sample *sa, unsigned int num_sample, unsigned int null_ind, bool f_enable_time);
int sample_print(sample *sa, unsigned int num_sample, unsigned int mode);
int sample_vectorization(sample *sa, dictionary *dict, unsigned int *arr_ind, unsigned int num_sample, unsigned int null_ind, bool f_enable_time, bool f_train, bool f_en_pe, float **pe_w, float random_noise_time);
int sample_vectorization_destructor(sample *sa, unsigned int *arr_ind, unsigned int num_sample);
int sample_hex_dump(sample *sa, dictionary *dict, unsigned int num_sample, FILE *f_dump);
int sample_destructor();
int word_idx(dictionary* dict, word* word_t);
int dictionary_constructor(dictionary *dict, sample *sa, unsigned int num_sample);
int dictionary_print(dictionary *dict);
int dictionary_destructor(dictionary *dict);

#endif
