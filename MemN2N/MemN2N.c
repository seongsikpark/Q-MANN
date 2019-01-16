#include "MemN2N.h"

int main(int argc, char*argv[])
{
    char tmp_word[MAX_WORD_LEN];

    sample *dict_sample;
    sample *train_sample;
    sample *test_sample;

    bool mode_train;
    bool mode_valid;
    bool mode_test;

    unsigned int *ind_sample_shuffled;
    unsigned int *ind_train_shuffled;
    unsigned int *ind_valid_shuffled;
    unsigned int *ind_test_shuffled;

    unsigned int num_sample;
    unsigned int num_sample_train;
    unsigned int num_sample_valid;
    unsigned int num_dict_sample;
    unsigned int num_sample_test;

    unsigned int null_ind;

    unsigned int max_line;      // max number of sentences in a sample
    unsigned int max_word;      // max number of words in a sentence
    unsigned int dim_dict;      // max number of words in dict

    dictionary dict;

    unsigned int i, j, k, l;
    int i_hop;

    unsigned int dim_input;
    unsigned int dim_word;      // max number of words in a sentence
    //
    unsigned int num_itr;
    unsigned int num_itr_linear_start;

    bool f_enable_time = EN_TIME;
    bool f_exp_plan = EN_EXP_TABLE_BASED;
    bool f_shift_based_sm = EN_SHIFT_BASED_SM;
    bool en_max_grad_l2_norm = EN_MAX_GRAD_L2_NORM;
    float max_grad_l2_norm = MAX_GRAD_L2_NORM;

    bool f_en_sample_shuffled = EN_SAMPLE_SHUFFLED;
    unsigned int ind_sample;

    sample *p_sam;
    float **m;
    float *q;
    float *a;
    float lr;
    float lambda;

	float *m_train;
	float *q_train;
	float *a_train;

	float *m_valid;
	float *q_valid;
	float *a_valid;

	float *m_test;
	float *q_test;
	float *a_test;

    unsigned int ind_sam, ind_itr, ind_sen, ind_batch;
    float cost_train;
    float cost_valid;
    float cost_test;
    bool verbose_debug = VERBOSE_DEBUG;
    float *grad;
    unsigned int dim_emb = DIM_EMB;

    float err_train;
    float err_valid;
    //float err_valid_prev = 0.0;
    float err_test;

    float err_valid_max=1.0;
    float err_valid_min=0.0;

    float cost_valid_max=0.0;

    bool en_save_best_model = EN_SAVE_BEST_MODEL;
    float err_valid_best;
    float cost_valid_best;
    unsigned int ind_early_stopping;


    float *predict;
    unsigned int predict_i;
    unsigned int match_count_train;
    unsigned int match_count_valid;
    unsigned int match_count_test;

    unsigned int ai;

    unsigned int size_batch=SIZE_BATCH;
    unsigned int num_batch;
    unsigned int size_b;

    unsigned int n_sen;

    unsigned int ind_data_set;
    unsigned int ind_data_set_s;
    unsigned int ind_data_set_e;

    char path_dict[100];
    char path_train[100];
    char path_test[100];

    clock_t time_train_s;
    clock_t time_train_e;
    clock_t time_test_s;
    clock_t time_test_e;
    float time_test_tmp;

	clock_t time_train_data_s;
	clock_t time_train_data_e;

	float time_train_data;

    // validation
    bool en_valid;


    //////
    // time arr
    //////      //
    //  [10][7][2]
    // [10] : emb_b, emb_c, emb_q, dot_mat_vec, softmax, dot_mat_vec, sum_vec, dense, softmax, cross_entropy
    // [7] : constructor, init, in, fwd, bwd, w_up, destructor
    double time_profile_train[NUM_LAYER][NUM_LAYER_OP];
    double time_profile_test[NUM_LAYER][NUM_LAYER_OP];
    double time_profile_train_total;
    double time_profile_test_total;

    unsigned int num_task_loop;
    unsigned int ind_task_loop;

    float *time_train_arr;
    float *time_test_arr;
    float *err_train_arr;
    float *err_test_arr;

    float time_train_avg;
    float time_test_avg;
    float err_train_avg;
    float err_test_avg;

    float time_train_max;
    float time_test_max;
    float err_train_max;
    float err_test_max;

    float time_train_min;
    float time_test_min;
    float err_train_min;
    float err_test_min;

    bool verbose = VERBOSE;

    bool en_train = EN_TRAIN;
    bool en_load_weight = EN_LOAD_WEIGHT;
    bool en_write_weight = EN_WRITE_WEIGHT;

    // joint task
    bool en_joint = EN_JOINT;
    bool done_joint_training = false;

    // position encoding
    bool en_pe = EN_PE;

    // linear mappint
    bool en_lin_map = EN_LINEAR_MAPPING;

    // non-linearity
    bool en_non_lin = EN_NON_LINEARITY;

    // linear start
    bool en_linear_start = EN_LINEAR_START;
    bool en_remove_softmax = true;

    // pthread variables
    pthread_t thread_write;
    pthread_t thread_read;

    int thread_write_id;
    int thread_read_id;
    int thread_write_status;
    int thread_read_status;

    FILE *f_fpga_write;
    FILE *f_fpga_read;
    packet_16 pkt_fpga_write;
    packet_16 pkt_fpga_read;

    // fixed point
    bool f_fixed = EN_FIXED_POINT;

    float rand_noise_time = RAND_NOISE_TIME;

    // enable scale layer - softmax in attention mode
    bool f_en_sc_att = EN_SC_ATT;

    if( argc != 2 ) {
        //num_task_loop = 1;
        num_task_loop = atoi(argv[1]);
    } else {
        num_task_loop = atoi(argv[1]);
    }
    printf("< NUM TASK LOOP : %d >\n",num_task_loop);

    if(RATE_NUM_VALID_SAMPLE != 0.0) {
        en_valid = true;
    } else {
        en_valid = false;
    }

    //
    time_train_arr = (float *) malloc(num_task_loop*sizeof(float));
    time_test_arr = (float *) malloc(num_task_loop*sizeof(float));
    err_train_arr = (float *) malloc(num_task_loop*sizeof(float));
    err_test_arr = (float *) malloc(num_task_loop*sizeof(float));


    printf("< LIST DATA SET >\n");
    printf("    %2d : %s\n", 0, "TEST_ALL");
    for(i=0;i<NUM_LIST_DATA;i++) {
        printf("    %2d : %s\n",i+1, LIST_DATA_SET[i]);
    }

    //while(true) {
    /*  
        printf("\ninput number of data set : ");
        //scanf("%d", &ind_data_set);
        printf("%d\n",atoi(argv[2]));
        ind_data_set = atoi(argv[2]);

        if(ind_data_set==-1) {
            printf("TEST ALL\n");

            ind_data_set_s = 1;
            ind_data_set_e = NUM_LIST_DATA;

            //break;
        } else if(ind_data_set==0) {
            printf("Start End index ?\n");
            scanf("%d %d", &ind_data_set_s, &ind_data_set_e);
            //break;
        } else if((ind_data_set>0)&&(ind_data_set<=NUM_LIST_DATA)) {
            printf("<  Set : %2d : %s>\n",ind_data_set,LIST_DATA_SET[ind_data_set-1]);

            ind_data_set_s = ind_data_set;
            ind_data_set_e = ind_data_set;

            //break;
        } else {
            printf("*E : Number of Data Set - Out of Range\n");
        }
    //}
    */

    ind_data_set_s = atoi(argv[2]);
    ind_data_set_e = atoi(argv[3]);

    // fixed point
    unsigned int iwl_argv = atoi(argv[4]);
    unsigned int frac_argv = BW_WL-1-atoi(argv[4]);

    printf("<Test index> %d ~ %d\n",ind_data_set_s, ind_data_set_e);


    unsigned int attention_mode = ATTENTION_MODE;
    unsigned int *attention_mode_hop;
    char *name_attention_mode;
    unsigned int num_hamming_attention = NUM_BIT_ATTENTION;


    if(attention_mode==1) {
        name_attention_mode = "normal";
    } else if(attention_mode==2) {
        name_attention_mode = "quantized";
    } else if(attention_mode==3) {
        name_attention_mode = "approximate";
    } else if(attention_mode==4) {
        name_attention_mode = "binary";
    } else {
        printf(" *E: undefined attention mode\n");
        exit(0);
    }

    // print setting
    printf("< Setting >\n");
    printf("   HW_MODE : %d\n",HW_MODE);
    printf("   EN_TRAIN : %d\n", en_train);
    printf("   EN_LOAD_WEIGHT : %d\n", en_load_weight);
    printf("   EN_WRITE_WEIGHT : %d\n", en_write_weight);
    printf("   ATTENTION MODE : %s\n", name_attention_mode);

    if(attention_mode==1) {
    } else if(attention_mode==2) {
        printf("    BW_WL : %d, BW_IWL : %d, BW_FRAC : %d\n", BW_WL, iwl_argv, frac_argv);
    } else if(attention_mode==3) {
        printf("    approximate attention bit : %d\n",num_hamming_attention);
    } else if(attention_mode==4) {
    } else {
    }



    // result file
    FILE *fp_result;
    fp_result = fopen("result.csv","w");
    fclose(fp_result);

    fp_result = fopen("result_all.csv","w");

    fprintf(fp_result,"<configurations>\n");
    fprintf(fp_result,"dim embedding: %d\n",DIM_EMB);
    fprintf(fp_result,"size batch: %d\n",SIZE_BATCH);
    fprintf(fp_result,"half decay step: %d\n",RATE_DECAY_STEP);
    fprintf(fp_result,"learning rate: %f\n",LEARNING_RATE);
    fprintf(fp_result,"num itr: %d\n",NUM_ITR);
    fprintf(fp_result,"num hop: %d\n",NUM_HOP);
    fprintf(fp_result,"en sample shuffled: %s\n",BOOL_PRINTF(EN_SAMPLE_SHUFFLED));
    fprintf(fp_result,"type weight tying: %d\n",TYPE_WEIGHT_TYING);
    fprintf(fp_result,"en linear mapping: %s\n",BOOL_PRINTF(EN_LINEAR_MAPPING));
    fprintf(fp_result,"en non linearity: %s\n",BOOL_PRINTF(EN_NON_LINEARITY));

    fprintf(fp_result,"en max grad l2 norm: %s\n",BOOL_PRINTF(EN_MAX_GRAD_L2_NORM));
    fprintf(fp_result,"max grad l2 norm: %f\n",MAX_GRAD_L2_NORM);
    fprintf(fp_result,"en time encoding: %s\n",BOOL_PRINTF(EN_TIME));
    fprintf(fp_result,"en position encoding: %s\n",BOOL_PRINTF(EN_PE));
    fprintf(fp_result,"rand noise time: %f\n",RAND_NOISE_TIME);
    fprintf(fp_result,"en linear start: %s\n",BOOL_PRINTF(EN_LINEAR_START));
    fprintf(fp_result,"num itr linear start: %d\n",NUM_ITR_LINEAR_START);
    fprintf(fp_result,"zeroing null weight: %s\n",BOOL_PRINTF(ZEROING_NULL_WEIGHT));

    fprintf(fp_result,"hw mode: %d\n",HW_MODE);
    fprintf(fp_result,"en fixed point: %s\n",BOOL_PRINTF(EN_FIXED_POINT));
    fprintf(fp_result,"iwl: %d",iwl_argv);
    fprintf(fp_result,"frac: %d",frac_argv);
    fprintf(fp_result,"attention_mode: %d\n",ATTENTION_MODE);
    fprintf(fp_result,"num bit attention: %d\n",NUM_BIT_ATTENTION);
    fprintf(fp_result,"attention const scale: %s\n",BOOL_PRINTF(ATTENTION_CONST_SCALE));
    fprintf(fp_result,"en scale attention: %s\n",BOOL_PRINTF(EN_SC_ATT));
    fprintf(fp_result,"hamming weight para: %d\n",HAMMING_WEIGHT_PARA);
    fprintf(fp_result,"en shift based softmax: %s\n",BOOL_PRINTF(EN_SHIFT_BASED_SM));
    fprintf(fp_result,"attention const scale: %f\n",ATTENTION_CONST_SCALE);




    fclose(fp_result);

    // layer declaration
    dense emb_q;            // q

    dense_mat *emb_m;        // emb mem m
    dense_mat *emb_c;        // emb mem c

    dot_mat_vec *dotmv;
    scale *sc_sf_in;
    softmax *sf_in;
    dot_mat_vec *w_sum;
    sum_vec *sv;

    dense *lin_map;

    activation *non_lin;

    dense ds_ans;
    softmax sf_out;
    cross_entropy ce;

    // for multiple hops
    float *sv_in_vec_a;
    float *sv_dev_in_vec_a;
    float *sv_grad_in;
    float *sv_dev_grad_in;

    // for pe
    float **pe_w;
    float *emb_q_vec;
    float *dev_emb_q_vec;


    // for linear start
    float *dotmv_grad_in;
    float *dev_dotmv_grad_in;
    float *w_sum_in_vec;
    float *dev_w_sum_in_vec;

    // for linear mapping
    float *lin_map_in;
    float *dev_lin_map_in;
    float *lin_map_grad_in;
    float *dev_lin_map_grad_in;
    float *ds_ans_in;
    float *dev_ds_ans_in;
    float *dev_dup_grad_bwd_in;

    // tmp - duplicated layer
    float **dup_grad;
    float **dev_dup_grad;

    float *dev_m_train;
    float *dev_q_train;
    float *dev_a_train;

    float *dev_m_valid;
    float *dev_q_valid;
    float *dev_a_valid;

    float *dev_m_test;
    float *dev_q_test;
    float *dev_a_test;


    //
    float *dotmv_in;
    float *dev_dotmv_in;

    //
    float *non_lin_in;
    float *non_lin_dev_in;

    float *non_lin_grad_in;
    float *non_lin_dev_grad_in;

    // fixed point operatin
    unsigned int *iwl;
    unsigned int *frac;

    unsigned int *iwl_w;
    unsigned int *frac_w;

    unsigned int *iwl_att;
    unsigned int *frac_att;

    unsigned int iwl_ds_ans;
    unsigned int frac_ds_ans;

    unsigned int iwl_bin;
    unsigned int frac_bin;

    unsigned int *f_mode;
    unsigned int *f_mode_w;
    unsigned int *f_mode_att;
    unsigned int f_mode_ds_ans;

	unsigned int *n_sen_train_arr;
	unsigned int n_sen_train_total;
	unsigned int *addr_m_arr_train;
	unsigned int *addr_q_arr_train;
	unsigned int *addr_a_arr_train;

	unsigned int *n_sen_valid_arr;
	unsigned int n_sen_valid_total;
	unsigned int *addr_m_arr_valid;
	unsigned int *addr_q_arr_valid;
	unsigned int *addr_a_arr_valid;

	unsigned int *n_sen_test_arr;
	unsigned int n_sen_test_total;
	unsigned int *addr_m_arr_test;
	unsigned int *addr_q_arr_test;
	unsigned int *addr_a_arr_test;

	unsigned int addr_m;
	unsigned int addr_q;
	unsigned int addr_a;

    //
    float att_bwd_scale=1.0;

    // softmax input
    float *softmax_input_tmp;
    float *softmax_output_tmp;

    unsigned int total_num_softmax_input;
    unsigned int ind_sf_in;
    unsigned int ind_sf_out;

    bool en_similarity_analysis = EN_SIMILARITY_ANALYSIS;

    FILE *fp_sf_in;
    FILE *fp_sf_out;

    if(en_similarity_analysis) {
        fp_sf_in = fopen("softmax_input_0to24.csv","w");
        fp_sf_out = fopen("softmax_output_0to24.csv","w");
        fclose(fp_sf_in);
        fclose(fp_sf_out);

        fp_sf_in = fopen("softmax_input_25to49.csv","w");
        fp_sf_out = fopen("softmax_output_25to49.csv","w");
        fclose(fp_sf_in);
        fclose(fp_sf_out);

        fp_sf_in = fopen("softmax_input_50to74.csv","w");
        fp_sf_out = fopen("softmax_output_50to74.csv","w");
        fclose(fp_sf_in);
        fclose(fp_sf_out);

        fp_sf_in = fopen("softmax_input_75to99.csv","w");
        fp_sf_out = fopen("softmax_output_75to99.csv","w");
        fclose(fp_sf_in);
        fclose(fp_sf_out);
    }

    for(ind_data_set=ind_data_set_s;ind_data_set<=ind_data_set_e;ind_data_set++) {
        strcpy(path_train,PATH_DATA_SET);
        if(en_joint) {
            strcat(path_train,LIST_DATA_SET[21-1]);
        } else {
            strcat(path_train,LIST_DATA_SET[ind_data_set-1]);
        }
        strcat(path_train,"_train_set");
        strcpy(path_dict,path_train);

        strcpy(path_test,PATH_DATA_SET);
        strcat(path_test,LIST_DATA_SET[ind_data_set-1]);
        strcat(path_test,"_test_set");

        //
        if((!en_joint)||(en_joint&(!done_joint_training))) {
            printf("< Extract Data Set : %s, %s, %s>\n",path_dict, path_train, path_test);
            dict_sample = sample_constructor(&path_dict, MAX_SEN_LEN, &num_dict_sample, NUM_DICT_SAMPLE);
            train_sample = sample_constructor(&path_train, MAX_SEN_LEN, &num_sample, NUM_SAMPLE);
    
            // make dict
            printf("< Make Dictionary >\n");
            dictionary_constructor(&dict, dict_sample, num_dict_sample);
    
            null_ind = 0;
    
            dictionary_print(&dict);
    
            //
            max_line = 0;
            for(i=0;i<num_sample;i++) {
                if(train_sample[i].n_sen > max_line) {
                    max_line = train_sample[i].n_sen;
                }
            }
    
            max_word = 0;
            for(i=0;i<num_sample;i++) {
                for(j=0;j<train_sample[i].n_sen;j++) {
                    if(train_sample[i].sentences[j].n > max_word) {
                        max_word = train_sample[i].sentences[j].n;
                    }
                }
            }
    
            if(DIM_FORCED) {
                dim_dict = MAX_DICT_LEN;
                max_word = MAX_LINE_LEN;
                dim_input = MAX_DICT_LEN + MAX_SEN_LEN;
    
                if(f_enable_time) {
                    dim_word = max_word+1;
                } else {
                    dim_word = max_word;
                }
            } else {
                dim_dict = dict.n;
                if(f_enable_time) {
                    dim_input = dim_dict + max_line;
                    dim_word = max_word+1;              // +1 -> temporal encoding index
                } else {
                    dim_input = dim_dict;
                    dim_word = max_word;
                }
            }
        }

        test_sample = sample_constructor(&path_test, max_line, &num_sample_test, NUM_SAMPLE_TEST);

        //printf("%s\n",path_test);

        for(i=0;i<num_sample;i++) {
            train_sample[i].dim_input = dim_input;
            train_sample[i].dim_word = dim_word;
            train_sample[i].dim_dict = dim_dict;
        }

        for(i=0;i<num_sample_test;i++) {
            test_sample[i].dim_input = dim_input;
            test_sample[i].dim_word = dim_word;
            test_sample[i].dim_dict = dim_dict;
        }

        sample_print(train_sample, num_sample, 0);
        sample_print(test_sample, num_sample_test, 0);

        // useless
        // position encoding weight array
        pe_w = (float **) malloc(dim_input*sizeof(float *));
        pe_w[0] = (float *) malloc(dim_input*dim_word*sizeof(float));

        for(i=1;i<dim_input;i++) {
            pe_w[i] = pe_w[i-1]+dim_word;
        }

        for(i=0;i<dim_input;i++) {
            for(j=0;j<dim_word;j++) {
                pe_w[i][j] = 1.0+4.0*((float)i/(float)dim_input-0.5)*((float)j/(float)dim_word-0.5);
            }
        }

        printf("< Sample Init. >\n");
        sample_init(train_sample, num_sample, null_ind, f_enable_time);
        sample_init(test_sample, num_sample_test, null_ind, f_enable_time);

        //sample_print(train_sample, num_sample, 0);
        //sample_print(test_sample, num_sample_test, 0);

        //printf("< Hex dump >\n");
        //FILE* fp_hex_dump;
        //fp_hex_dump=fopen("qa1_test.bin","wb");

        //sample_hex_dump(train_sample, &dict, num_sample, fp_hex_dump);
        //sample_hex_dump(test_sample, &dict, num_sample_test, fp_hex_dump);

        //fclose(fp_hex_dump);


        num_sample_valid = num_sample * RATE_NUM_VALID_SAMPLE;
        num_sample_train = num_sample - num_sample_valid;

        printf("< Layer Construction >\n");

        printf("    Dim input : %d\n", dim_input);
        printf("    Dim emb   : %d\n", dim_emb);

        printf("    Number of Samples - Train : %d\n",num_sample_train);
        printf("    Number of Samples - Valid : %d\n",num_sample_valid);
        printf("    Number of Samples - Test  : %d\n",num_sample_test);
        printf("    Size of Batch : %d\n",size_batch);
        printf("    Number of Hop : %d\n",NUM_HOP);

        // init time array
        for(i=0;i<NUM_LAYER;i++) {
            for(j=0;j<NUM_LAYER_OP;j++) {
                time_profile_train[i][j] = 0.0;
            }
        }

        //
        if((!en_joint)||(en_joint&(ind_data_set==ind_data_set_s))) {
            emb_m = (dense_mat*) malloc(NUM_HOP*sizeof(dense_mat));
            emb_c = (dense_mat*) malloc(NUM_HOP*sizeof(dense_mat));
    
            dotmv = (dot_mat_vec*) malloc(NUM_HOP*sizeof(dot_mat_vec));
            sf_in = (softmax*) malloc(NUM_HOP*sizeof(softmax));
            w_sum = (dot_mat_vec*) malloc(NUM_HOP*sizeof(dot_mat_vec));
    
            if(en_lin_map) {
                lin_map = (dense *) malloc(NUM_HOP*sizeof(dense));
            }
    
            sv = (sum_vec*) malloc(NUM_HOP*sizeof(sum_vec));
    
            if(en_non_lin) {
                non_lin = (activation *) malloc(NUM_HOP*sizeof(activation));
            }
    
            // scale layer
            sc_sf_in = (scale *) malloc(NUM_HOP*sizeof(scale));
    
            iwl = (unsigned int *) malloc(NUM_HOP*sizeof(unsigned int));
            frac = (unsigned int *) malloc(NUM_HOP*sizeof(unsigned int));
    
            iwl_w = (unsigned int *) malloc(NUM_HOP*sizeof(unsigned int));
            frac_w = (unsigned int *) malloc(NUM_HOP*sizeof(unsigned int));
    
            iwl_att = (unsigned int *) malloc(NUM_HOP*sizeof(unsigned int));
            frac_att = (unsigned int *) malloc(NUM_HOP*sizeof(unsigned int));

            f_mode = (unsigned int*) malloc(NUM_HOP*sizeof(unsigned int));
            f_mode_w = (unsigned int*) malloc(NUM_HOP*sizeof(unsigned int));
            f_mode_att = (unsigned int*) malloc(NUM_HOP*sizeof(unsigned int));

            attention_mode_hop = (unsigned int *) malloc(NUM_HOP*sizeof(unsigned int));
    
    
            // last
            /*
            for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                iwl[i_hop] = 0;
                frac[i_hop] = 0;
    
                iwl_w[i_hop] = 0;
                frac_w[i_hop] = 0;
    
                iwl_att[i_hop] = 4;
                frac_att[i_hop] = 4;
            }
    
            iwl_ds_ans = 2;
            frac_ds_ans = 8;
            */
    
    
            // binary
            for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                iwl[i_hop] = iwl_argv;
                frac[i_hop] = frac_argv;

                iwl_w[i_hop] = iwl_argv;
                frac_w[i_hop] = frac_argv;

                iwl_att[i_hop] = iwl_argv;
                frac_att[i_hop] = frac_argv;
                
                /*
                iwl_w[i_hop] = 1;
                frac_w[i_hop] = 2;

                iwl_att[i_hop] = 2;
                frac_att[i_hop] = 5;
                */
    
                attention_mode_hop[i_hop] = attention_mode;

                f_mode[i_hop] = QUANT_MODE;
                f_mode_w[i_hop] = QUANT_MODE;
                f_mode_att[i_hop] = QUANT_MODE;
            }

            f_mode_ds_ans = QUANT_MODE;


            //iwl[0] = iwl_argv+1;
            //frac[0] = frac_argv-1;

            //iwl[2] = iwl_argv-1;
            //frac[2] = frac_argv+1;
    
#ifdef EN_MQ
            iwl_w[0] = iwl_w[0]+1;
            frac_w[0] = frac_w[0]-1;

            iwl_w[2] = iwl_w[2]-1;
            frac_w[2] = frac_w[2]+1;
#endif

            //f_mode_w[1]=0;
            //f_mode[1]=2;
    
            /*
            attention_mode_hop[0] = 3;
            attention_mode_hop[1] = 3;
            attention_mode_hop[2] = 3;
            */
    
            // last layer -> floating point
            iwl_ds_ans = 8;
            frac_ds_ans = 7;

            if(BINARY_MODE) {
                iwl_bin = 0;
                frac_bin = 0;
            } else {
                iwl_bin=iwl_argv;
                frac_bin=frac_argv;
            }

    
            //printf("here! : %f\n",FIXED_MAX_FLOAT(2,5));
    
            // for sweep sim
            /*
            f_fixed = true;
            for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                iwl[i_hop] = atoi(argv[3]);
                frac[i_hop] = atoi(argv[4]);
    
                iwl_w[i_hop] = atoi(argv[3]);
                frac_w[i_hop] = atoi(argv[4]);
    
                iwl_att[i_hop] = atoi(argv[3]);
                frac_att[i_hop] = atoi(argv[4]);
            }
            iwl_ds_ans = atoi(argv[3]);
            frac_ds_ans = atoi(argv[4]);
            */
    
            // fixed point att. precision test
            /*
            f_fixed = false;
            for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                iwl[i_hop] = 2;
                frac[i_hop] = 8;
    
                iwl_w[i_hop] = 1;
                frac_w[i_hop] = 1;
    
                iwl_att[i_hop] = atoi(argv[3]);
                frac_att[i_hop] = atoi(argv[4]);
                //iwl_att[i_hop] = 1;
                //frac_att[i_hop] = 1;
            }
    
            iwl_ds_ans = 2;
            frac_ds_ans = 8;
            */
    
            //printf("%d %d\n",iwl_ds_ans,frac_ds_ans);
    


            fp_result = fopen("result_all.csv","a");

            // embedding part
                //f_fixed = true;
            //f_fixed = false;
            time_profile_train[2][0] += dense_constructor(&emb_q, dim_input, dim_emb, en_max_grad_l2_norm, max_grad_l2_norm,"NULL",f_fixed,iwl_w[0],frac_w[0],iwl_w[0],frac_w[0],f_mode_w[0],fp_result);
            //time_profile_train[2][0] += dense_constructor(&emb_q, dim_input, dim_emb, en_max_grad_l2_norm, max_grad_l2_norm,"NULL",f_fixed,iwl_w[0],frac_w[0],1,2,f_mode_w[0],fp_result);
            //time_profile_train[2][0] += dense_constructor(&emb_q, dim_input, dim_emb, en_max_grad_l2_norm, max_grad_l2_norm,"NULL",f_fixed,0,0);
            //f_fixed = EN_FIXED_POINT;
    
            for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                // emb - mem part
                    //f_fixed = true;
                //f_fixed = false;
                time_profile_train[0][0] += dense_mat_constructor(&emb_m[i_hop], max_line, dim_input, dim_emb, en_max_grad_l2_norm, max_grad_l2_norm,f_fixed,iwl_w[i_hop],frac_w[i_hop],f_mode_w[i_hop],fp_result);     // mem_m
                //time_profile_train[0][0] += dense_mat_constructor(&emb_m[i_hop], max_line, dim_input, dim_emb, en_max_grad_l2_norm, max_grad_l2_norm,f_fixed,1,2,f_mode_w[i_hop],fp_result);     // mem_m
                //f_fixed = false;
                time_profile_train[1][0] += dense_mat_constructor(&emb_c[i_hop], max_line, dim_input, dim_emb, en_max_grad_l2_norm, max_grad_l2_norm,f_fixed,iwl_w[i_hop],frac_w[i_hop],f_mode_w[i_hop],fp_result);     // mem_c
                //time_profile_train[1][0] += dense_mat_constructor(&emb_c[i_hop], max_line, dim_input, dim_emb, en_max_grad_l2_norm, max_grad_l2_norm,f_fixed,1,2,f_mode_w[i_hop],fp_result);     // mem_c

                f_fixed = EN_FIXED_POINT;
    
                // attention
            // fixed point att. precision test
                //f_fixed = true;
                if(attention_mode==2) {
                    time_profile_train[3][0] += dot_mat_vec_constructor(&dotmv[i_hop], max_line, dim_emb, dim_emb, false,f_fixed,iwl_att[i_hop],frac_att[i_hop], iwl_bin, frac_bin, f_mode_att[i_hop], attention_mode_hop[i_hop],fp_result);
                } else {
                    time_profile_train[3][0] += dot_mat_vec_constructor(&dotmv[i_hop], max_line, dim_emb, dim_emb, false,f_fixed,iwl_att[i_hop],frac_att[i_hop],iwl_att[i_hop],frac_att[i_hop],f_mode_att[i_hop],attention_mode_hop[i_hop],fp_result);
                }

                if(f_en_sc_att) {
                    time_profile_train[4][0] += scale_constructor(&sc_sf_in[i_hop], max_line, f_fixed, iwl_att[i_hop], frac_att[i_hop],f_mode_att[i_hop],fp_result);
                }
                // for the proposed method
                time_profile_train[4][0] += softmax_constructor(&sf_in[i_hop], max_line, f_exp_plan, f_shift_based_sm,fp_result);
    
            // fixed point att. precision test
                //f_fixed = false;
    
                // o - weighted sum
                //f_fixed = false;
                time_profile_train[5][0] += dot_mat_vec_constructor(&w_sum[i_hop], max_line, dim_emb, max_line, true,f_fixed,iwl[i_hop],frac[i_hop], iwl[i_hop],frac[i_hop],f_mode[i_hop],attention_mode_hop[i_hop],fp_result);
                f_fixed = EN_FIXED_POINT;
    
                //f_fixed = false;
                if(en_lin_map) {
                    //time_profile_train[6][0] += dense_constructor(&lin_map[i_hop], dim_emb, dim_emb, en_max_grad_l2_norm, max_grad_l2_norm, "NULL", f_fixed,iwl[i_hop],frac[i_hop]);
                    //time_profile_train[6][0] += dense_constructor(&lin_map[i_hop], dim_emb, dim_emb, en_max_grad_l2_norm, max_grad_l2_norm, "NULL", false,iwl_bin,frac_bin,iwl[i_hop],frac[i_hop]);
                    // test_170409
                    //time_profile_train[6][0] += dense_constructor(&lin_map[i_hop], dim_emb, dim_emb, en_max_grad_l2_norm, max_grad_l2_norm/2, "NULL", false,iwl_bin,frac_bin,iwl[i_hop],frac[i_hop]);
                    //time_profile_train[6][0] += dense_constructor(&lin_map[i_hop], dim_emb, dim_emb, en_max_grad_l2_norm, max_grad_l2_norm/2, "NULL", f_fixed,iwl_bin,frac_bin,iwl[i_hop],frac[i_hop],fp_result);
                    time_profile_train[6][0] += dense_constructor(&lin_map[i_hop], dim_emb, dim_emb, en_max_grad_l2_norm, max_grad_l2_norm/2, "NULL", f_fixed,iwl_bin,frac_bin,iwl_w[i_hop],frac_w[i_hop],f_mode_w[i_hop],fp_result);
                    //time_profile_train[6][0] += dense_constructor(&lin_map[i_hop], dim_emb, dim_emb, en_max_grad_l2_norm, max_grad_l2_norm/2, "NULL", f_fixed,iwl_bin,frac_bin,1,2,f_mode_w[i_hop],fp_result);
                    //time_profile_train[6][0] += dense_constructor(&lin_map[i_hop], dim_emb, dim_emb, en_max_grad_l2_norm, max_grad_l2_norm, "RELU", f_fixed,iwl_bin,frac_bin,iwl_w[i_hop],frac_w[i_hop],fp_result);
                    //time_profile_train[6][0] += dense_constructor(&lin_map[i_hop], dim_emb, dim_emb, en_max_grad_l2_norm, max_grad_l2_norm/2, "NULL", f_fixed,iwl_bin,frac_bin,iwl_w[i_hop],frac_w[i_hop],fp_result);
                }
    
                // o + u - sum vector
                //f_fixed = true;
                /*
                if(i_hop==NUM_HOP-1) {
                    time_profile_train[6][0] += sum_vec_constructor(&sv[i_hop], dim_emb,f_fixed,0,0);
                } else {
                    time_profile_train[6][0] += sum_vec_constructor(&sv[i_hop], dim_emb,f_fixed,iwl[i_hop],frac[i_hop]);
                }
                */
                //f_fixed = false;
                time_profile_train[6][0] += sum_vec_constructor(&sv[i_hop], dim_emb,f_fixed,iwl[i_hop],frac[i_hop],f_mode[i_hop],fp_result);
                //time_profile_train[6][0] += sum_vec_constructor(&sv[i_hop], dim_emb,f_fixed,0,0);
                f_fixed = EN_FIXED_POINT;
    
                //f_fixed = false;
                if(en_non_lin) {
                    time_profile_train[6][0] += activation_constructor(&non_lin[i_hop], dim_emb, "RELU",f_fixed,iwl[i_hop],frac[i_hop],f_mode[i_hop],fp_result);
                }
            }
    
            // output part
            // dense
            //f_fixed = true;
            f_fixed = false;
            //time_profile_train[7][0] = dense_constructor(&ds_ans, dim_emb, dim_input, en_max_grad_l2_norm, max_grad_l2_norm, "NULL",f_fixed,iwl[NUM_HOP-1],frac[NUM_HOP-1]);
            
                
            time_profile_train[7][0] = dense_constructor(&ds_ans, dim_emb, dim_input, en_max_grad_l2_norm, max_grad_l2_norm, "NULL",f_fixed,iwl_ds_ans,frac_ds_ans,iwl_ds_ans,frac_ds_ans,f_mode_ds_ans,fp_result);
    
                f_fixed = EN_FIXED_POINT;
            // softmax
            time_profile_train[8][0] = softmax_constructor(&sf_out, dim_input, f_exp_plan, false,fp_result);
            // cost function - cross_entropy
            time_profile_train[9][0] = cross_entropy_constructor(&ce, dim_input,fp_result);


            fprintf(fp_result,"<train_test_result>\n");
            fprintf(fp_result,"ind_data_set,time_train_avg,time_train_max,time_train_min,err_train_avg,err_train_max,err_train_min,time_test_avg,time_test_max,time_test_min,err_test_avg,err_test_max,err_test_min,");

            for(i=0;i<num_task_loop;i++) {
                fprintf(fp_result,"%d",i);
                if(i!=num_task_loop-1) {
                    fprintf(fp_result,",");
                } else {
                    fprintf(fp_result,"\n");
                }
            }

            fclose(fp_result);
    
            // dup grad
            dup_grad = (float**) malloc(NUM_HOP*sizeof(float*));
            dup_grad[0] = (float*) malloc(NUM_HOP*dim_emb*sizeof(float));
            for(i_hop=1;i_hop<NUM_HOP;i_hop++) {
                dup_grad[i_hop] = dup_grad[i_hop-1]+dim_emb;
            }
    
            dev_dup_grad = (float**) malloc(NUM_HOP*sizeof(float*));
    
            if(en_gpu_model) {
                cuda_data_constructor
                (
                    &(dev_m_train),
                    &(dev_q_train),
                    &(dev_a_train),
                    max_line*num_sample_train,
                    dim_input,
					num_sample_train
                );

                cuda_data_constructor
                (
                    &(dev_m_valid),
                    &(dev_q_valid),
                    &(dev_a_valid),
                    max_line*num_sample_valid,
                    dim_input,
					num_sample_valid
                );

                cuda_data_constructor
                (
                    &(dev_m_test),
                    &(dev_q_test),
                    &(dev_a_test),
                    max_line*num_sample_test,
                    dim_input,
					num_sample_test
                );
    
                /*
                cuda_dup_grad_constructor
                (
                    &(dev_dup_grad),
                    NUM_HOP,
                    dim_emb
                );
                */
                for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                    cuda_dup_grad_constructor
                    (
                        &(dev_dup_grad[i_hop]),
                        1,
                        dim_emb
                    );
                }

                // 
            }
        }

        for(ind_task_loop=0;ind_task_loop<num_task_loop;ind_task_loop++) {
            printf("TASK LOOP : %d\n",ind_task_loop);

            if(en_train) {
                time_train_s = clock();

                printf("< Train Phase >\n");
                lr = LEARNING_RATE;
                lambda = LAMBDA;

                // init
                time_profile_train[2][1] += dense_init(&emb_q);
    
                for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                    time_profile_train[0][1] += dense_mat_init(&emb_m[i_hop]);
                    time_profile_train[1][1] += dense_mat_init(&emb_c[i_hop]);
    
                    time_profile_train[3][1] += dot_mat_vec_init(&dotmv[i_hop]);
    
                    if(f_en_sc_att) {
                        time_profile_train[4][1] += scale_init(&sc_sf_in[i_hop]);
                    }
    
                    time_profile_train[4][1] += softmax_init(&sf_in[i_hop]);
    
                    time_profile_train[5][1] += dot_mat_vec_init(&w_sum[i_hop]);
    
                    if(en_lin_map) {
                        time_profile_train[6][1] += dense_init(&lin_map[i_hop]);
                    }
    
                    time_profile_train[6][1] += sum_vec_init(&sv[i_hop]);
    
                    if(en_non_lin) {
                        time_profile_train[6][1] += activation_init(&non_lin[i_hop]);
                    }
                }
    
                time_profile_train[7][1] += dense_init(&ds_ans);
    
                time_profile_train[8][1] += softmax_init(&sf_out);
                time_profile_train[9][1] += cross_entropy_init(&ce);
    
                // sample order shuffle
                ind_sample_shuffled = (unsigned int*) malloc(num_sample*sizeof(unsigned int));
                ind_train_shuffled = (unsigned int*) malloc(num_sample_train*sizeof(unsigned int));
                ind_valid_shuffled = (unsigned int*) malloc(num_sample_valid*sizeof(unsigned int));
                //
                if(en_linear_start == true) {
                    num_itr = NUM_ITR + NUM_ITR_LINEAR_START;
                    num_itr_linear_start = NUM_ITR_LINEAR_START;
                } else {
                    num_itr = NUM_ITR;
                    num_itr_linear_start = 0;
                }

                for(ind_sample=0;ind_sample<num_sample;ind_sample++) {
                    ind_sample_shuffled[ind_sample] = ind_sample;
                }

                if(f_en_sample_shuffled) {
                    rand_perm(ind_sample_shuffled, num_sample);
                }



                //err_valid_prev = 0.0;
                //err_valid = 0.0;

				time_train_data = 0.0;

                err_valid_best = err_valid_max;
                cost_valid_best = cost_valid_max;
                ind_early_stopping = 0;

                for(ind_itr=0;ind_itr<num_itr;ind_itr++) {
                    /*
					if(ind_itr==19) {
						verbose_debug = true;
					} else {
                        verbose_debug = false;
                    }
                    */
         	        if(verbose == true) {
              	        printf("< ITR : %3d > ",ind_itr);
                   	}

                    // linear start
                    //if( (en_remove_softmax==true)&&(en_linear_start==true)&&((ind_itr<10)||(err_valid<=err_valid_prev)) ) {
                    //if( (en_remove_softmax==true)&&(en_linear_start==true)&&(err_valid<=err_valid_prev) ) {
                    if( (en_linear_start==true)&&(ind_itr<num_itr_linear_start) ) {
                        en_remove_softmax = true;
                        printf(" en remove sotfmax : true\n");
                        lr = LEARNING_RATE/2.0;
                        //lr = 0.005;
                    } else {
                        if(en_remove_softmax==true) {
                            lr = LEARNING_RATE;
                        }
                        en_remove_softmax = false;
                        //printf(" en remove sotfmax : false\n");

                        // learning rate decay
                        if((((ind_itr-num_itr_linear_start)%RATE_DECAY_STEP)==0)&&(ind_itr!=num_itr_linear_start)) {
                            lr = lr/2.0;

                            // test_170306
                            //att_bwd_scale = att_bwd_scale*2.0;
                        }
                    }

                    cost_train = 0.0;
                    err_train = 0.0;
                    match_count_train=0;
                    match_count_valid=0;
                    match_count_test=0;

                    num_batch = ceil((float)num_sample_train/(float)size_batch);
                    //printf("Number of Batch : %d\n",num_batch);


                    if(f_en_sample_shuffled) {
                        rand_perm(ind_train_shuffled, num_sample_train);
                    }

                	if(f_en_sample_shuffled || (ind_itr==0)) {
	                    for(ind_sample=0;ind_sample<num_sample_train;ind_sample++) {
    	                        ind_train_shuffled[ind_sample] = ind_sample_shuffled[ind_sample];
        	            }

                    	sample_vectorization(train_sample, &dict, ind_train_shuffled, num_sample_train, null_ind, f_enable_time, 1, en_pe, pe_w, rand_noise_time);

    					n_sen_train_total = 0;
    					n_sen_train_arr = (unsigned int *) malloc(num_sample_train*sizeof(unsigned int));
    
    					addr_m_arr_train = (unsigned int *) malloc(num_sample_train*sizeof(unsigned int));
    					addr_q_arr_train = (unsigned int *) malloc(num_sample_train*sizeof(unsigned int));
    					addr_a_arr_train = (unsigned int *) malloc(num_sample_train*sizeof(unsigned int));
    
    					for(ind_sam=0;ind_sam<num_sample_train;ind_sam++) {
                            p_sam = &(train_sample[ind_train_shuffled[ind_sam]]);
    
    						n_sen_train_arr[ind_sam] = p_sam->n_sen;
                            n_sen_train_total += p_sam->n_sen;
    					}
    
    					m_train = (float *) malloc(n_sen_train_total*dim_input*sizeof(float));
    					q_train = (float *) malloc(num_sample_train*dim_input*sizeof(float));
    					a_train = (float *) malloc(num_sample_train*dim_input*sizeof(float));
    
    					addr_m = 0;
    					addr_q = 0;
    					addr_a = 0;
    
    					for(ind_sam=0;ind_sam<num_sample_train;ind_sam++) {
    						addr_m_arr_train[ind_sam] = addr_m;
    						addr_q_arr_train[ind_sam] = addr_q;
    						addr_a_arr_train[ind_sam] = addr_a;
    
                            p_sam = &(train_sample[ind_train_shuffled[ind_sam]]);
                            m = p_sam->sentences_b;
                            q = p_sam->question_b;
                            a = p_sam->answer_b;
    
    						memcpy(&m_train[addr_m],m[0],n_sen_train_arr[ind_sam]*dim_input*sizeof(float));
    						memcpy(&q_train[addr_q],q,dim_input*sizeof(float));
    						memcpy(&a_train[addr_a],a,dim_input*sizeof(float));
    
    						addr_m += n_sen_train_arr[ind_sam]*dim_input;
    						addr_q += dim_input;
    						addr_a += dim_input;
    					}
    
    
                        if(en_gpu_model) {
                            cuda_data_in
                            (
                                dev_m_train,
                                dev_q_train,
                                dev_a_train,
                                m_train,
                                q_train,
                                a_train,
                                n_sen_train_total,
                                //max_line,
                                dim_input,
    							num_sample_train
                            );
                        }
					}

                    for(ind_batch=0;ind_batch<num_batch;ind_batch++) {
                        //printf("batch index : %d\n",ind_batch);
                        for(ind_sam=ind_batch*size_batch;(ind_sam<(ind_batch+1)*size_batch)&&(ind_sam<num_sample_train);ind_sam++) {
                            if(verbose_debug) {
                                printf("sample index : %d\n",ind_train_shuffled[ind_sam]);
                            }

							/*
							time_train_data_s=clock();
                            p_sam = &(train_sample[ind_train_shuffled[ind_sam]]);
                            m = p_sam->sentences_b;
                            q = p_sam->question_b;
                            a = p_sam->answer_b;
                            ai = p_sam->answer_i.words[0];

                            n_sen = p_sam->n_sen;

                            if(en_gpu_model) {
                                cuda_data_in
                                (
                                    dev_m,
                                    dev_q,
                                    dev_a,
                                    m[0],
                                    q,
                                    a,
                                    n_sen,
                                    //max_line,
                                    dim_input
                                );
                            }
							time_train_data_e=clock();

                			time_train_data += (double)(time_train_data_e-time_train_data_s)/(double)CLOCKS_PER_SEC;
							*/


							n_sen = n_sen_train_arr[ind_sam];

                            //printf("n_sen : %d\n",n_sen);

                            if(ind_batch != num_batch-1) {
                                size_b = size_batch;
                            } else {
                                size_b = num_sample_train-ind_batch*size_batch;
                            }

                            ////////////////////////////////////////
                            // input setting - train
                            ////////////////////////////////////////
                            //time_profile_train[2][2] += dense_in(&emb_q, q, dup_grad[0], dev_q, dev_dup_grad[0]);
                            time_profile_train[2][2] += dense_in(&emb_q, q, dup_grad[0], &dev_q_train[addr_q_arr_train[ind_sam]], dev_dup_grad[0]);

                            emb_q_vec = emb_q.out_vec;
                            dev_emb_q_vec = emb_q.dev_out_vec;


                            for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                               //time_profile_train[0][2] += dense_mat_in(&emb_m[i_hop], n_sen, m, dotmv[i_hop].grad_out_mat, dev_m, dotmv[i_hop].dev_grad_out_mat);
                               //time_profile_train[1][2] += dense_mat_in(&emb_c[i_hop], n_sen, m, w_sum[i_hop].grad_out_mat, dev_m, w_sum[i_hop].dev_grad_out_mat);


							    time_profile_train[0][2] += dense_mat_in(&emb_m[i_hop], n_sen, m, dotmv[i_hop].grad_out_mat, &dev_m_train[addr_m_arr_train[ind_sam]], dotmv[i_hop].dev_grad_out_mat);
                                time_profile_train[1][2] += dense_mat_in(&emb_c[i_hop], n_sen, m, w_sum[i_hop].grad_out_mat, &dev_m_train[addr_m_arr_train[ind_sam]], w_sum[i_hop].dev_grad_out_mat);

                                if(en_remove_softmax) {
                                    dotmv_grad_in = w_sum[i_hop].grad_out_vec;
                                    dev_dotmv_grad_in = w_sum[i_hop].dev_grad_out_vec;
                                } else {
                                    if(f_en_sc_att) {
                                        dotmv_grad_in = sc_sf_in[i_hop].grad_out;
                                        dev_dotmv_grad_in = sc_sf_in[i_hop].dev_grad_out;
                                    } else {
                                        dotmv_grad_in = sf_in[i_hop].grad_out;
                                        dev_dotmv_grad_in = sf_in[i_hop].dev_grad_out;
                                    }
                                }

                                if(i_hop==0) {
                                    time_profile_train[3][2] += dot_mat_vec_in(&dotmv[i_hop], n_sen, emb_m[i_hop].out_mat, emb_q_vec, dotmv_grad_in, emb_m[i_hop].dev_out_mat, dev_emb_q_vec, dev_dotmv_grad_in);
                                } else {
                                    if(en_non_lin) {
                                        dotmv_in = non_lin[i_hop-1].out;
                                        dev_dotmv_in = non_lin[i_hop-1].dev_out;
                                    } else {
                                        dotmv_in = sv[i_hop-1].out_vec;
                                        dev_dotmv_in = sv[i_hop-1].dev_out_vec;
                                    }

                                    time_profile_train[3][2] += dot_mat_vec_in(&dotmv[i_hop], n_sen, emb_m[i_hop].out_mat, dotmv_in, dotmv_grad_in, emb_m[i_hop].dev_out_mat, dev_dotmv_in, dev_dotmv_grad_in);
                                }

                                if(f_en_sc_att) {
                                    time_profile_train[4][2] += scale_in(&sc_sf_in[i_hop], n_sen, dotmv[i_hop].out_vec, sf_in[i_hop].grad_out, dotmv[i_hop].dev_out_vec, sf_in[i_hop].dev_grad_out);
                                    time_profile_train[4][2] += softmax_in(&sf_in[i_hop], n_sen, sc_sf_in[i_hop].out, w_sum[i_hop].grad_out_vec, sc_sf_in[i_hop].dev_out, w_sum[i_hop].dev_grad_out_vec);
                                } else {
                                    time_profile_train[4][2] += softmax_in(&sf_in[i_hop], n_sen, dotmv[i_hop].out_vec, w_sum[i_hop].grad_out_vec, dotmv[i_hop].dev_out_vec, w_sum[i_hop].dev_grad_out_vec);
                                }

                                if(en_remove_softmax) {
                                    w_sum_in_vec = dotmv[i_hop].out_vec;
                                    dev_w_sum_in_vec = dotmv[i_hop].dev_out_vec;
                                } else {
                                    w_sum_in_vec = sf_in[i_hop].out_vec;
                                    dev_w_sum_in_vec = sf_in[i_hop].dev_out_vec;
                                }

                                time_profile_train[5][2] += dot_mat_vec_in(&w_sum[i_hop], n_sen, emb_c[i_hop].out_mat, w_sum_in_vec, sv[i_hop].grad_out, emb_c[i_hop].dev_out_mat, dev_w_sum_in_vec, sv[i_hop].dev_grad_out);

                                if(en_lin_map) {
                                    lin_map_grad_in = sv[i_hop].grad_out;
                                    dev_lin_map_grad_in = sv[i_hop].dev_grad_out;

                                    if(i_hop==0) {
                                        lin_map_in = emb_q_vec;
                                        dev_lin_map_in = dev_emb_q_vec;
                                    } else {
                                        lin_map_in = sv[i_hop-1].out_vec;
                                        dev_lin_map_in = sv[i_hop-1].dev_out_vec;
                                    }

                                    time_profile_train[6][2] += dense_in(&lin_map[i_hop], lin_map_in, lin_map_grad_in, dev_lin_map_in, dev_lin_map_grad_in);
                                }

                                if(en_lin_map) {
                                    sv_in_vec_a = lin_map[i_hop].out_vec;
                                    sv_dev_in_vec_a = lin_map[i_hop].dev_out_vec;
                                } else {
                                    if(i_hop==0) {
                                        sv_in_vec_a = emb_q_vec;
                                        sv_dev_in_vec_a = dev_emb_q_vec;
                                    } else {
                                        sv_in_vec_a = sv[i_hop-1].out_vec;
                                        sv_dev_in_vec_a = sv[i_hop-1].dev_out_vec;
                                    }
                                }

                                if(en_non_lin) {
                                    sv_grad_in = non_lin[i_hop].grad_out;
                                    sv_dev_grad_in = non_lin[i_hop].dev_grad_out;
                                } else {
                                    if(i_hop==NUM_HOP-1) {
                                        sv_grad_in = ds_ans.grad_out;
                                        sv_dev_grad_in = ds_ans.dev_grad_out;
                                    } else {
                                        sv_grad_in = dup_grad[i_hop+1];
                                        //sv_dev_grad_in = dev_dup_grad+(i_hop+1)*dim_emb;
                                        sv_dev_grad_in = dev_dup_grad[i_hop+1];
                                    }
                                }

                                time_profile_train[6][2] += sum_vec_in(&sv[i_hop], sv_in_vec_a, w_sum[i_hop].out_vec, sv_grad_in, sv_dev_in_vec_a, w_sum[i_hop].dev_out_vec, sv_dev_grad_in);

                                if(en_non_lin) {
                                    non_lin_in = sv[i_hop].out_vec;
                                    non_lin_dev_in = sv[i_hop].dev_out_vec;

                                    if(i_hop==NUM_HOP-1) {
                                        non_lin_grad_in = ds_ans.grad_out;
                                        non_lin_dev_grad_in = ds_ans.dev_grad_out;
                                    } else {
                                        non_lin_grad_in = dup_grad[i_hop+1];
                                        //non_lin_dev_grad_in = dev_dup_grad+(i_hop+1)*dim_emb;
                                        non_lin_dev_grad_in = dev_dup_grad[i_hop+1];
                                    }

                                    time_profile_train[6][2] += activation_in(&non_lin[i_hop], non_lin_in, non_lin_grad_in, non_lin_dev_in, non_lin_dev_grad_in);
                                }
                            }

                            if(en_non_lin) {
                                ds_ans_in = non_lin[NUM_HOP-1].out;
                                dev_ds_ans_in = non_lin[NUM_HOP-1].dev_out;
                            } else {
                                ds_ans_in = sv[NUM_HOP-1].out_vec;
                                dev_ds_ans_in = sv[NUM_HOP-1].dev_out_vec;
                            }

                            time_profile_train[7][2] += dense_in(&ds_ans, ds_ans_in, ce.grad_out, dev_ds_ans_in, ce.dev_grad_out);
                            time_profile_train[8][2] += softmax_in(&sf_out, dim_input, ds_ans.out_vec, ce.grad_out, ds_ans.dev_out_vec, ce.dev_grad_out);
                            //time_profile_train[9][2] += cross_entropy_in(&ce, sf_out.out_vec, a, sf_out.dev_out_vec, dev_a);
                            time_profile_train[9][2] += cross_entropy_in(&ce, sf_out.out_vec, a, sf_out.dev_out_vec, &dev_a_train[addr_a_arr_train[ind_sam]]);

                            //printf("TEST : host : %f : dev : %f\n",sf_out.out_vec[0], sf_out.dev_out_vec[0]);
                            //printf("TEST : host : %f \n",sf_out.out_vec[0]);

                            ////////////////////////////////////////
                            // forward propagation - train
                            ////////////////////////////////////////
                            //printf("Forward Propagation\n");
                            //printf("emb_q\n");
                            time_profile_train[2][3] += dense_fwd(&emb_q, verbose_debug);

                            for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                                //printf("%d \n",n_sen);
                                
                                /*
                                if(ind_itr>0) {
                                    verbose_debug=true;
                                }
                                */

                                //printf("emb_m\n");
                                time_profile_train[0][3] += dense_mat_fwd(&emb_m[i_hop], verbose_debug);

                                //printf("emb_c\n");
                                time_profile_train[1][3] += dense_mat_fwd(&emb_c[i_hop], verbose_debug);
                                
                                /*
                                if(ind_itr>0) {
                                    verbose_debug=false;
                                }
                                */

                                //verbose_debug = true;
                                //
                                /*
                                if(ind_itr > 9) {
                                    verbose_debug = true;
                                }
                                */

                                //printf("dotmv\n");
                                time_profile_train[3][3] += dot_mat_vec_fwd(&dotmv[i_hop], verbose_debug);



                                if(!en_remove_softmax) {
                                    if(f_en_sc_att) {
                                        time_profile_train[4][3] += scale_fwd(&sc_sf_in[i_hop], verbose_debug);
                                    }
                                    //printf("sf_in\n");
                                    time_profile_train[4][3] += softmax_fwd(&sf_in[i_hop], verbose_debug);
                                }

                                // similarity analysis
                                if(en_similarity_analysis) {
                                    //if( (ind_itr==0)|(ind_itr==9)|(ind_itr==19)|(ind_itr==49)|(ind_itr==99) ) {
                                    //if((ind_itr%10)==9) {
                                    //if(ind_itr<20) {
                                    //if( (ind_itr<20)|(ind_itr==49)|(ind_itr==99) ) {
                                    if(ind_itr<100) {
                                        // softmax input analysis
                                        softmax_input_tmp = (float *) malloc(n_sen*sizeof(float));
    
                                        cuda_copy_dev2host(softmax_input_tmp,dotmv[i_hop].dev_out_vec,n_sen);

                                        if(ind_itr<25) {
                                            fp_sf_in = fopen("softmax_input_0to24.csv","a");
                                            fp_sf_out = fopen("softmax_output_0to24.csv","a");
                                        } else if(ind_itr<50) {
                                            fp_sf_in = fopen("softmax_input_25to49.csv","a");
                                            fp_sf_out = fopen("softmax_output_25to49.csv","a");
                                        } else if(ind_itr<75) {
                                            fp_sf_in = fopen("softmax_input_50to74.csv","a");
                                            fp_sf_out = fopen("softmax_output_50to74.csv","a");
                                        } else if(ind_itr<100) {
                                            fp_sf_in = fopen("softmax_input_75to99.csv","a");
                                            fp_sf_out = fopen("softmax_output_75to99.csv","a");
                                        }

                                        fprintf(fp_sf_in,"%d,%d,%d,",ind_itr,ind_sam,i_hop);
    
                                        for(ind_sf_in=0;ind_sf_in<n_sen;ind_sf_in++) {
                                            fprintf(fp_sf_in,"%f",softmax_input_tmp[ind_sf_in]);
                                            if(ind_sf_in==n_sen-1) {
                                                fprintf(fp_sf_in,"\n");
                                            } else {
                                                fprintf(fp_sf_in,",");
                                            }
                                        }
                                        fclose(fp_sf_in);
    
                                        free(softmax_input_tmp);
    
                                        // softmax output analysis
                                        softmax_output_tmp = (float *) malloc(n_sen*sizeof(float));
    
                                        cuda_copy_dev2host(softmax_output_tmp,sf_in[i_hop].dev_out_vec,n_sen);
    
                                        fprintf(fp_sf_out,"%d,%d,%d,",ind_itr,ind_sam,i_hop);
    
                                        for(ind_sf_out=0;ind_sf_out<n_sen;ind_sf_out++) {
                                            fprintf(fp_sf_out,"%f",softmax_output_tmp[ind_sf_out]);
                                            if(ind_sf_out==n_sen-1) {
                                                fprintf(fp_sf_out,"\n");
                                            } else {
                                                fprintf(fp_sf_out,",");
                                            }
                                        }
                                        fclose(fp_sf_out);
    
                                        free(softmax_output_tmp);
                                    }
                                }

                                //verbose_debug = false;

                                //printf("w_sum\n");
                                time_profile_train[5][3] += dot_mat_vec_fwd(&w_sum[i_hop], verbose_debug);


                                if(en_lin_map) {
                                    //printf("lin_map\n");
                                    time_profile_train[6][3] += dense_fwd(&lin_map[i_hop], verbose_debug);
                                }

                                //printf("sv\n");
                                time_profile_train[6][3] += sum_vec_fwd(&sv[i_hop], verbose_debug);

                                if(en_non_lin) {
                                    //printf("non_lin\n");
                                    time_profile_train[6][3] += activation_fwd(&non_lin[i_hop], verbose_debug);
                                }
                            }

                            //printf("ds_ans\n");
                            time_profile_train[7][3] += dense_fwd(&ds_ans, verbose_debug);

                            //printf("sf_out\n");
                            time_profile_train[8][3] += softmax_fwd(&sf_out, verbose_debug);

							/*
                            predict = sf_out.out_vec;

                            predict_i = 0;
                            for(j=0;j<dim_input;j++) {
                                if(predict[predict_i] < predict[j]) {
                                    predict_i = j;
                                }
                            }

                            if( predict_i == ai ) {
                                match_count++;
                            }
							*/

                            //printf("predict i : %d, ai : %d\n",predict_i, ai);

                            /*
                            printf("Answer\n");
                            for(j=0;j<dim_input;j++) {
                                printf("%.1f ",p_sam->answer_b[j]);
                            }
                            printf("\n");
                            */

                            ////////////////////////////////////////
                            // cost function
                            ////////////////////////////////////////
                            //printf("ce\n");
                            time_profile_train[9][3] += cross_entropy_run(&ce,1);
                            //cost_train += ce.cost;

                            ////////////////////////////////////////
                            // backward propagation
                            ////////////////////////////////////////
                            //printf("Backward Propagation\n");
                            //time_profile_train[7][4] += dense_bwd(&ds_ans, ce.grad, false);
                            time_profile_train[7][4] += dense_bwd(&ds_ans, false);
                            //time_profile_train[7][4] += dense_bwd(&ds_ans, true);

                            for(i_hop=NUM_HOP-1;i_hop>=0;i_hop--) {
                                if(en_non_lin) {
                                    time_profile_train[6][4] += activation_bwd(&non_lin[i_hop], verbose_debug);
                                }

                                time_profile_train[6][4] += sum_vec_bwd(&sv[i_hop]);
                                if(en_lin_map) {
                                    time_profile_train[6][4] += dense_bwd(&lin_map[i_hop], verbose_debug);
                                }

                                // test1
                                //verbose_debug = true;
                                time_profile_train[5][4] += dot_mat_vec_bwd(&w_sum[i_hop], 1.0, verbose_debug);

                                /* 
                                if(ind_itr>19) {
                                    verbose_debug = true;
                                }
                                */

                                if(!en_remove_softmax) {
                                    time_profile_train[4][4] += softmax_bwd(&sf_in[i_hop], verbose_debug);
                                    if(f_en_sc_att) {
                                        time_profile_train[4][4] += scale_bwd(&sc_sf_in[i_hop], verbose_debug);
                                    }
                                }


                                time_profile_train[3][4] += dot_mat_vec_bwd(&dotmv[i_hop], att_bwd_scale, verbose_debug);

                                //verbose_debug = false;

                                time_profile_train[1][4] += dense_mat_bwd(&emb_c[i_hop], false);
                                time_profile_train[0][4] += dense_mat_bwd(&emb_m[i_hop], false);


                                if(en_gpu_model) {
                                    if(en_lin_map) {
                                        dev_dup_grad_bwd_in = lin_map[i_hop].dev_grad_out;
                                    } else {
                                        dev_dup_grad_bwd_in = sv[i_hop].dev_grad_out;
                                    }

                                    cuda_dup_grad_bwd
                                    (
                                        //dev_dup_grad+i_hop*dim_emb,
                                        dev_dup_grad[i_hop],
                                        dotmv[i_hop].dev_grad_out_vec,
                                        dev_dup_grad_bwd_in,
                                        dup_grad[i_hop],
                                        dim_emb,
#ifdef EN_GRAD_QUANT
										f_fixed,
#else
                                        false,
#endif
                                        iwl[i_hop],
                                        frac[i_hop],
                                        f_mode[i_hop]
                                    );
                                }

                                if(en_cpu) {
                                    for(i=0;i<dim_emb;i++) {
                                        if(en_lin_map) {
                                            dup_grad[i_hop][i] = (dotmv[i_hop].grad_out_vec[i]+lin_map[i_hop].grad_out[i]);
                                        } else {
                                            dup_grad[i_hop][i] = (dotmv[i_hop].grad_out_vec[i]+sv[i_hop].grad_out[i]);
                                        }
                                    }
                                }
                            }
                            time_profile_train[2][4] += dense_bwd(&emb_q, false);

                        }

                        // weight update & weight tying
                        /*
                        //printf("w_up\n");
                        time_profile_train[2][5] += dense_w_up(&emb_q, lr, size_b, lambda, false);
                        for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                            time_profile_train[0][5] += dense_mat_w_up(&emb_m[i_hop], lr, size_b, lambda, false);
                            time_profile_train[1][5] += dense_mat_w_up(&emb_c[i_hop], lr, size_b, lambda, false);
                        }
                        time_profile_train[7][5] += dense_w_up(&ds_ans, lr, size_b, lambda, false);

                        if(en_lin_map) {
                            for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                                time_profile_train[6][5] += dense_w_up(&lin_map[i_hop], lr, size_b, lambda, false);
                            }
                        }

                        if(f_en_sc_att) {
                            for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                                scale_w_up(&sc_sf_in[i_hop], lr, size_b, lambda, false);
                            }
                        }
                        */

                        if(TYPE_WEIGHT_TYING==1) {
                            cuda_accum_mat(emb_q.dev_w_mat_del,emb_m[0].dev_w_mat_del,dim_input,dim_emb,false);
                            for(i_hop=1;i_hop<NUM_HOP;i_hop++) {
                                //printf("copy_mat_b: %d\n",i_hop);
                                cuda_accum_mat(emb_c[i_hop-1].dev_w_mat_del,emb_m[i_hop].dev_w_mat,dim_input,dim_emb,false);
                            }
                            //printf("copy_mat_c\n");
                            cuda_accum_mat(emb_c[NUM_HOP-1].dev_w_mat_del,ds_ans.dev_w_mat_del,dim_input,dim_emb,true);

                            time_profile_train[2][5] += dense_w_up(&emb_q, lr, size_b, lambda, false);
                            for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                                time_profile_train[0][5] += dense_mat_w_up(&emb_m[i_hop], lr, size_b, lambda, false);
                                time_profile_train[1][5] += dense_mat_w_up(&emb_c[i_hop], lr, size_b, lambda, false);
                            }
                            time_profile_train[7][5] += dense_w_up(&ds_ans, lr, size_b, lambda, false);


                            if(en_lin_map) {
                                for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                                    time_profile_train[6][5] += dense_w_up(&lin_map[i_hop], lr, size_b, lambda, false);
                                }
                            }
    
                            if(f_en_sc_att) {
                                for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                                    scale_w_up(&sc_sf_in[i_hop], lr, size_b, lambda, false);
                                }
                            }

                            // #1. adjacent
                            // tying
                            if(en_gpu_model) {
                                //printf("copy_mat_a\n");

                                cuda_copy_mat(emb_m[0].dev_w_mat,emb_q.dev_w_mat,dim_input,dim_emb,false);
                                for(i_hop=1;i_hop<NUM_HOP;i_hop++) {
                                    //printf("copy_mat_b: %d\n",i_hop);
                                    cuda_copy_mat(emb_m[i_hop].dev_w_mat,emb_c[i_hop-1].dev_w_mat,dim_input,dim_emb,false);
                                }
                                //printf("copy_mat_c\n");
                                cuda_copy_mat(ds_ans.dev_w_mat,emb_c[NUM_HOP-1].dev_w_mat,dim_emb,dim_input,true);

                                /*
                                cuda_copy_mat(emb_m[0].dev_w_mat,emb_q.dev_w_mat,dim_input,dim_emb,false);
                                for(i_hop=1;i_hop<NUM_HOP;i_hop++) {
                                    //printf("copy_mat_b: %d\n",i_hop);
                                    cuda_copy_mat(emb_c[i_hop-1].dev_w_mat,emb_m[i_hop].dev_w_mat,dim_input,dim_emb,false);
                                }
                                //printf("copy_mat_c\n");
                                cuda_copy_mat(emb_c[NUM_HOP-1].dev_w_mat,ds_ans.dev_w_mat,dim_input,dim_emb,dim_input,true);
                                */

                                /*
                                if(en_lin_map) {
                                    for(i_hop=1;i_hop<NUM_HOP;i_hop++) {
                                        cuda_copy_mat(lin_map[0].dev_w_mat,lin_map[i_hop].dev_w_mat,dim_emb,dim_emb,false);
                                    }
                                }

                                if(f_en_sc_att) {
                                    for(i_hop=1;i_hop<NUM_HOP;i_hop++) {
                                        cuda_copy_mat(sc_sf_in[i_hop].dev_w,sc_sf_in[i_hop-1].dev_w,n_sen,1,false);
                                    }
                                }
                                */
                            }

                            if(en_cpu) {
                                printf("NOT YET IMPLEMENTED:CPU: weight tying type 1\n");
                                /*
                                mat_copy(emb_m[0].w_mat,emb_q.w_mat,dim_emb,dim_input,false);
                                for(i_hop=1;i_hop<NUM_HOP;i_hop++) {
                                    mat_copy(emb_m[i_hop].w_mat,emb_c[i_hop-1].w_mat,dim_emb,dim_input,false);
                                }
                                mat_copy(ds_ans.w_mat,emb_c[NUM_HOP-1].w_mat,dim_input,dim_emb,true);

                                if(en_lin_map) {
                                    for(i_hop=1;i_hop<NUM_HOP;i_hop++) {
                                        mat_copy(lin_map[0].w_mat,lin_map[i_hop].w_mat,dim_emb,dim_emb,false);
                                    }
                                }
                                */
                            }
                        } else if(TYPE_WEIGHT_TYING==2) {
                            // #2. layer-wise(RNN)
                            // tying
                            for(i_hop=1;i_hop<NUM_HOP;i_hop++) {
                                cuda_accum_mat(emb_m[i_hop].dev_w_mat_del,emb_m[0].dev_w_mat_del,dim_input,dim_emb,false);
                                cuda_accum_mat(emb_c[i_hop].dev_w_mat_del,emb_c[0].dev_w_mat_del,dim_input,dim_emb,false);

                            }

                            if(en_lin_map) {
                                for(i_hop=1;i_hop<NUM_HOP;i_hop++) {
                                    cuda_accum_mat(lin_map[i_hop].dev_w_mat_del,lin_map[0].dev_w_mat_del,dim_emb,dim_emb,false);
                                }
                            }

                            /*
                            time_profile_train[2][5] += dense_w_up(&emb_q, lr, size_b, lambda, false);
                            time_profile_train[0][5] += dense_mat_w_up(&emb_m[0], lr, size_b, lambda, false);
                            time_profile_train[1][5] += dense_mat_w_up(&emb_c[0], lr, size_b, lambda, false);
                            time_profile_train[7][5] += dense_w_up(&ds_ans, lr, size_b, lambda, false);
                            */

                            time_profile_train[2][5] += dense_w_up(&emb_q, lr, size_b, lambda, false);
                            for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                                time_profile_train[0][5] += dense_mat_w_up(&emb_m[i_hop], lr, size_b, lambda, false);
                                time_profile_train[1][5] += dense_mat_w_up(&emb_c[i_hop], lr, size_b, lambda, false);
                            }
                            time_profile_train[7][5] += dense_w_up(&ds_ans, lr, size_b, lambda, false);

                            if(en_lin_map) {
                                for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                                    //time_profile_train[6][5] += dense_w_up(&lin_map[i_hop], lr, size_b, lambda, false);
                                    //test_170409
                                    time_profile_train[6][5] += dense_w_up(&lin_map[i_hop], lr*0.1, size_b, lambda, false);
                                    //time_profile_train[6][5] += dense_w_up(&lin_map[i_hop], lr*0.5, size_b, lambda, false);
                                    //time_profile_train[6][5] += dense_w_up(&lin_map[i_hop], lr*0.08, size_b, 0.000001, false);
                                }
                            }
    
                            if(f_en_sc_att) {
                                for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                                    scale_w_up(&sc_sf_in[i_hop], lr, size_b, lambda, false);
                                }
                            }

                            if(en_gpu_model) {
                                for(i_hop=1;i_hop<NUM_HOP;i_hop++) {
                                    cuda_copy_mat(emb_m[0].dev_w_mat,emb_m[i_hop].dev_w_mat,dim_input,dim_emb,false);
                                    cuda_copy_mat(emb_c[0].dev_w_mat,emb_c[i_hop].dev_w_mat,dim_input,dim_emb,false);
                                    //
                                    //cuda_copy_mat(emb_m[0].dev_w_mat,emb_m[i_hop].dev_w_mat,dim_emb,dim_input,false);
                                    //cuda_copy_mat(emb_c[0].dev_w_mat,emb_c[i_hop].dev_w_mat,dim_emb,dim_input,false);
                                }

                                /*
                                for(i_hop=0;i_hop<NUM_HOP-1;i_hop++) {
                                    cuda_copy_mat(emb_m[NUM_HOP-1].dev_w_mat,emb_m[i_hop].dev_w_mat,dim_input,dim_emb,false);
                                    cuda_copy_mat(emb_c[NUM_HOP-1].dev_w_mat,emb_c[i_hop].dev_w_mat,dim_input,dim_emb,false);
                                }
                                */

                                if(en_lin_map) {
                                    for(i_hop=1;i_hop<NUM_HOP;i_hop++) {
                                        cuda_copy_mat(lin_map[0].dev_w_mat,lin_map[i_hop].dev_w_mat,dim_emb,dim_emb,false);
                                    }
                                }

                                /*
                                if(f_en_sc_att) {
                                    for(i_hop=1;i_hop<NUM_HOP;i_hop++) {
                                        cuda_copy_mat(sc_sf_in[i_hop].dev_w,sc_sf_in[i_hop-1].dev_w,n_sen,1,false);
                                    }
                                }
                                */
                            }

                            if(en_cpu) {
                                printf("NOT YET IMPLEMENTED:CPU: weight tying type 2\n");
                                /*
                                for(i_hop=1;i_hop<NUM_HOP;i_hop++) {
                                    mat_copy(emb_m[0].w_mat,emb_m[i_hop].w_mat,dim_emb,dim_input,false);
                                    mat_copy(emb_c[0].w_mat,emb_c[i_hop].w_mat,dim_emb,dim_input,false);
                                }

                                if(en_lin_map) {
                                    for(i_hop=1;i_hop<NUM_HOP;i_hop++) {
                                        mat_copy(lin_map[0].w_mat,lin_map[i_hop].w_mat,dim_emb,dim_emb,false);
                                    }
                                }
                                */
                            }
                        } else {
                            printf("*E : type of weight tying : %d\n",TYPE_WEIGHT_TYING);
                        }

                        // zeroing null weight
                        if(ZEROING_NULL_WEIGHT) {
                            if(en_gpu_model) {
                                for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                                    cuda_set_value
                                    (
                                        emb_m[i_hop].dev_w_mat,
                                        0.0,
                                        dim_emb*dim_input,
                                        0,
                                        dim_input
                                    );
                                    cuda_set_value
                                    (
                                        emb_c[i_hop].dev_w_mat,
                                        0.0,
                                        dim_emb*dim_input,
                                        0,
                                        dim_input
                                    );
                                }
                            }

                            if(en_cpu) {
                                for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                                    for(i=0;i<dim_emb;i++) {
                                        emb_m[i_hop].w_mat[i][null_ind] = 0.0;
                                        emb_c[i_hop].w_mat[i][null_ind] = 0.0;
                                    }
                                }
                            }
                        }
                    }

                    //err_train = 1.0 - ((float)match_count/(float)num_sample_train);

                    ////////////////////////////////////////
                    // validation
                    ////////////////////////////////////////
                    //printf("validation\n");
                    if(en_valid) {
                        cost_valid = 0.0;
                        //err_valid_prev = err_valid;
                        err_valid = 0.0;
                        match_count_valid=0;

                    	if(ind_itr==0) {
			                for(ind_sample=0;ind_sample<num_sample_valid;ind_sample++) {
            			        ind_valid_shuffled[ind_sample] = ind_sample_shuffled[ind_sample+num_sample_train];
                			}

                			sample_vectorization(train_sample, &dict, ind_valid_shuffled, num_sample_valid, null_ind, f_enable_time, 0, en_pe, pe_w, rand_noise_time);

        					n_sen_valid_total = 0;
        					n_sen_valid_arr = (unsigned int *) malloc(num_sample_valid*sizeof(unsigned int));
        
        					addr_m_arr_valid = (unsigned int *) malloc(num_sample_valid*sizeof(unsigned int));
        					addr_q_arr_valid = (unsigned int *) malloc(num_sample_valid*sizeof(unsigned int));
        					addr_a_arr_valid = (unsigned int *) malloc(num_sample_valid*sizeof(unsigned int));
        
        					for(ind_sam=0;ind_sam<num_sample_valid;ind_sam++) {
                                p_sam = &(train_sample[ind_valid_shuffled[ind_sam]]);
        
        						n_sen_valid_arr[ind_sam] = p_sam->n_sen;
                                n_sen_valid_total += p_sam->n_sen;
        					}
        
        					m_valid = (float *) malloc(n_sen_valid_total*dim_input*sizeof(float));
        					q_valid = (float *) malloc(num_sample_valid*dim_input*sizeof(float));
        					a_valid = (float *) malloc(num_sample_valid*dim_input*sizeof(float));
        
        					addr_m = 0;
        					addr_q = 0;
        					addr_a = 0;
        
        					for(ind_sam=0;ind_sam<num_sample_valid;ind_sam++) {
        						addr_m_arr_valid[ind_sam] = addr_m;
        						addr_q_arr_valid[ind_sam] = addr_q;
        						addr_a_arr_valid[ind_sam] = addr_a;
        
                                p_sam = &(train_sample[ind_valid_shuffled[ind_sam]]);
                                m = p_sam->sentences_b;
                                q = p_sam->question_b;
                                a = p_sam->answer_b;
        
        						memcpy(&m_valid[addr_m],m[0],n_sen_valid_arr[ind_sam]*dim_input*sizeof(float));
        						memcpy(&q_valid[addr_q],q,dim_input*sizeof(float));
        						memcpy(&a_valid[addr_a],a,dim_input*sizeof(float));
        
        						addr_m += n_sen_valid_arr[ind_sam]*dim_input;
        						addr_q += dim_input;
        						addr_a += dim_input;
        					}
        
        
                            if(en_gpu_model) {
                                cuda_data_in
                                (
                                    dev_m_valid,
                                    dev_q_valid,
                                    dev_a_valid,
                                    m_valid,
                                    q_valid,
                                    a_valid,
                                    n_sen_valid_total,
                                    //max_line,
                                    dim_input,
        							num_sample_valid
                                );
                            }
    					}


                        for(ind_sam=0;ind_sam<num_sample_valid;ind_sam++) {
							/*
                            p_sam = &(train_sample[ind_valid_shuffled[ind_sam]]);
                            m = p_sam->sentences_b;
                            q = p_sam->question_b;
                            a = p_sam->answer_b;
                            //ai = p_sam->answer_i.words[0];

                            n_sen = p_sam->n_sen;

                            if(en_gpu_model) {
                                cuda_data_in
                                (
                                    dev_m,
                                    dev_q,
                                    dev_a,
                                    m[0],
                                    q,
                                    a,
                                    n_sen,
                                    //max_line,
                                    dim_input,
									1
                                );
                            }
							*/

							n_sen = n_sen_valid_arr[ind_sam];
                            ////////////////////////////////////////
                            // input setting - validation
                            ////////////////////////////////////////
                            time_profile_train[2][2] += dense_in(&emb_q, q, dup_grad[0], &dev_q_valid[addr_q_arr_valid[ind_sam]], dev_dup_grad[0]);

                            emb_q_vec = emb_q.out_vec;
                            dev_emb_q_vec = emb_q.dev_out_vec;

                            for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                                time_profile_train[0][2] += dense_mat_in(&emb_m[i_hop], n_sen, m, dotmv[i_hop].grad_out_mat, &dev_m_valid[addr_m_arr_valid[ind_sam]], dotmv[i_hop].dev_grad_out_mat);
                                time_profile_train[1][2] += dense_mat_in(&emb_c[i_hop], n_sen, m, w_sum[i_hop].grad_out_mat, &dev_m_valid[addr_m_arr_valid[ind_sam]], w_sum[i_hop].dev_grad_out_mat);

                                if(en_remove_softmax) {
                                    dotmv_grad_in = w_sum[i_hop].grad_out_vec;
                                    dev_dotmv_grad_in = w_sum[i_hop].dev_grad_out_vec;
                                } else {
                                    if(f_en_sc_att) {
                                        dotmv_grad_in = sc_sf_in[i_hop].grad_out;
                                        dev_dotmv_grad_in = sc_sf_in[i_hop].dev_grad_out;
                                    } else {
                                        dotmv_grad_in = sf_in[i_hop].grad_out;
                                        dev_dotmv_grad_in = sf_in[i_hop].dev_grad_out;
                                    }
                                }


                                if(i_hop==0) {
                                    time_profile_train[3][2] += dot_mat_vec_in(&dotmv[i_hop], n_sen, emb_m[i_hop].out_mat, emb_q_vec, dotmv_grad_in, emb_m[i_hop].dev_out_mat, dev_emb_q_vec, dev_dotmv_grad_in);
                                } else {
                                    if(en_non_lin) {
                                        dotmv_in = non_lin[i_hop-1].out;
                                        dev_dotmv_in = non_lin[i_hop-1].dev_out;
                                    } else {
                                        dotmv_in = sv[i_hop-1].out_vec;
                                        dev_dotmv_in = sv[i_hop-1].dev_out_vec;
                                    }

                                    time_profile_train[3][2] += dot_mat_vec_in(&dotmv[i_hop], n_sen, emb_m[i_hop].out_mat, dotmv_in, dotmv_grad_in, emb_m[i_hop].dev_out_mat, dev_dotmv_in, dev_dotmv_grad_in);
                                }

                                if(f_en_sc_att) {
                                    time_profile_train[4][2] += scale_in(&sc_sf_in[i_hop], n_sen, dotmv[i_hop].out_vec, sf_in[i_hop].grad_out, dotmv[i_hop].dev_out_vec, sf_in[i_hop].dev_grad_out);
                                    time_profile_train[4][2] += softmax_in(&sf_in[i_hop], n_sen, sc_sf_in[i_hop].out, w_sum[i_hop].grad_out_vec, sc_sf_in[i_hop].dev_out, w_sum[i_hop].dev_grad_out_vec);
                                } else {
                                    time_profile_train[4][2] += softmax_in(&sf_in[i_hop], n_sen, dotmv[i_hop].out_vec, w_sum[i_hop].grad_out_vec, dotmv[i_hop].dev_out_vec, w_sum[i_hop].dev_grad_out_vec);
                                }

                                if(en_remove_softmax) {
                                    w_sum_in_vec = dotmv[i_hop].out_vec;
                                    dev_w_sum_in_vec = dotmv[i_hop].dev_out_vec;
                                } else {
                                    w_sum_in_vec = sf_in[i_hop].out_vec;
                                    dev_w_sum_in_vec = sf_in[i_hop].dev_out_vec;
                                }

                                time_profile_train[5][2] += dot_mat_vec_in(&w_sum[i_hop], n_sen, emb_c[i_hop].out_mat, w_sum_in_vec, sv[i_hop].grad_out, emb_c[i_hop].dev_out_mat, dev_w_sum_in_vec, sv[i_hop].dev_grad_out);

                                if(en_lin_map) {
                                    lin_map_grad_in = sv[i_hop].grad_out;
                                    dev_lin_map_grad_in = sv[i_hop].dev_grad_out;

                                    if(i_hop==0) {
                                        lin_map_in = emb_q_vec;
                                        dev_lin_map_in = dev_emb_q_vec;
                                    } else {
                                        lin_map_in = sv[i_hop-1].out_vec;
                                        dev_lin_map_in = sv[i_hop-1].dev_out_vec;
                                    }

                                    time_profile_train[6][2] += dense_in(&lin_map[i_hop], lin_map_in, lin_map_grad_in, dev_lin_map_in, dev_lin_map_grad_in);
                                }

                                if(en_lin_map) {
                                    sv_in_vec_a = lin_map[i_hop].out_vec;
                                    sv_dev_in_vec_a = lin_map[i_hop].dev_out_vec;
                                } else {
                                    if(i_hop==0) {
                                        sv_in_vec_a = emb_q_vec;
                                        sv_dev_in_vec_a = dev_emb_q_vec;
                                    } else {
                                        sv_in_vec_a = sv[i_hop-1].out_vec;
                                        sv_dev_in_vec_a = sv[i_hop-1].dev_out_vec;
                                    }
                                }

                                /*
                                if(i_hop==NUM_HOP-1) {
                                    sv_grad_in = ds_ans.grad_out;
                                    sv_dev_grad_in = ds_ans.dev_grad_out;
                                } else {
                                    sv_grad_in = dup_grad[i_hop+1];
                                    sv_dev_grad_in = dev_dup_grad+(i_hop+1)*dim_emb;
                                }

                                time_profile_train[6][2] += sum_vec_in(&sv[i_hop], dim_emb, sv_in_vec_a, w_sum[i_hop].out_vec, sv_grad_in, sv_dev_in_vec_a, w_sum[i_hop].dev_out_vec, sv_dev_grad_in);
                                */

                                if(en_non_lin) {
                                    sv_grad_in = non_lin[i_hop].grad_out;
                                    sv_dev_grad_in = non_lin[i_hop].dev_grad_out;
                                } else {
                                    if(i_hop==NUM_HOP-1) {
                                        sv_grad_in = ds_ans.grad_out;
                                        sv_dev_grad_in = ds_ans.dev_grad_out;
                                    } else {
                                        sv_grad_in = dup_grad[i_hop+1];
                                        //sv_dev_grad_in = dev_dup_grad+(i_hop+1)*dim_emb;
                                        sv_dev_grad_in = dev_dup_grad[i_hop+1];
                                    }
                                }

                                time_profile_train[6][2] += sum_vec_in(&sv[i_hop], sv_in_vec_a, w_sum[i_hop].out_vec, sv_grad_in, sv_dev_in_vec_a, w_sum[i_hop].dev_out_vec, sv_dev_grad_in);

                                if(en_non_lin) {
                                    non_lin_in = sv[i_hop].out_vec;
                                    non_lin_dev_in = sv[i_hop].dev_out_vec;

                                    if(i_hop==NUM_HOP-1) {
                                        non_lin_grad_in = ds_ans.grad_out;
                                        non_lin_dev_grad_in = ds_ans.dev_grad_out;
                                    } else {
                                        non_lin_grad_in = dup_grad[i_hop+1];
                                        //non_lin_dev_grad_in = dev_dup_grad+(i_hop+1)*dim_emb;
                                        non_lin_dev_grad_in = dev_dup_grad[i_hop+1];
                                    }

                                    time_profile_train[6][2] += activation_in(&non_lin[i_hop], non_lin_in, non_lin_grad_in, non_lin_dev_in, non_lin_dev_grad_in);
                                }
                            }

                            if(en_non_lin) {
                                ds_ans_in = non_lin[NUM_HOP-1].out;
                                dev_ds_ans_in = non_lin[NUM_HOP-1].dev_out;
                            } else {
                                ds_ans_in = sv[NUM_HOP-1].out_vec;
                                dev_ds_ans_in = sv[NUM_HOP-1].dev_out_vec;
                            }

                            time_profile_train[7][2] += dense_in(&ds_ans, ds_ans_in, ce.grad_out, dev_ds_ans_in, ce.dev_grad_out);
                            time_profile_train[8][2] += softmax_in(&sf_out, dim_input, ds_ans.out_vec, ce.grad_out, ds_ans.dev_out_vec, ce.dev_grad_out);
                            time_profile_train[9][2] += cross_entropy_in(&ce, sf_out.out_vec, a, sf_out.dev_out_vec, &dev_a_valid[addr_a_arr_valid[ind_sam]]);

                            //printf("TEST : host : %f : dev : %f\n",sf_out.out_vec[0], sf_out.dev_out_vec[0]);
                            //printf("TEST : host : %f \n",sf_out.out_vec[0]);

                            ////////////////////////////////////////
                            // forward propagation - validation
                            ////////////////////////////////////////
                            //printf("Forward Propagation\n");
                            time_profile_train[2][3] += dense_fwd(&emb_q, verbose_debug);

                            for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                                time_profile_train[0][3] += dense_mat_fwd(&emb_m[i_hop], verbose_debug);
                                time_profile_train[1][3] += dense_mat_fwd(&emb_c[i_hop], verbose_debug);
                                time_profile_train[3][3] += dot_mat_vec_fwd(&dotmv[i_hop], verbose_debug);
                                if(!en_remove_softmax) {
                                    if(f_en_sc_att) {
                                        time_profile_train[4][3] += scale_fwd(&sc_sf_in[i_hop], verbose_debug);
                                    }
                                    time_profile_train[4][3] += softmax_fwd(&sf_in[i_hop], verbose_debug);
                                }

                                time_profile_train[5][3] += dot_mat_vec_fwd(&w_sum[i_hop], verbose_debug);
                                if(en_lin_map) {
                                    time_profile_train[6][3] += dense_fwd(&lin_map[i_hop], verbose_debug);
                                }
                                time_profile_train[6][3] += sum_vec_fwd(&sv[i_hop], verbose_debug);
                                if(en_non_lin) {
                                    time_profile_train[6][3] += activation_fwd(&non_lin[i_hop], verbose_debug);
                                }
                            }

                            time_profile_train[7][3] += dense_fwd(&ds_ans, verbose_debug);

                            time_profile_train[8][3] += softmax_fwd(&sf_out, verbose_debug);

							/*
                            predict = sf_out.out_vec;

                            predict_i = 0;
                            for(j=0;j<dim_input;j++) {
                                if(predict[predict_i] < predict[j]) {
                                    predict_i = j;
                                }
                            }

                            if( predict_i == ai ) {
                                match_count++;
                            }
							*/

                            ////////////////////////////////////////
                            // cost function
                            ////////////////////////////////////////
                            time_profile_train[9][3] += cross_entropy_run(&ce,2);
                            //cost_valid += ce.cost;
                        }

                        //err_valid = 1.0-((float)match_count/(float)num_sample_valid);
                    }

					cross_entropy_cost_load(&ce,&cost_train,&cost_valid,&cost_test);
					cross_entropy_m_cnt_load(&ce, &match_count_train, &match_count_valid, &match_count_test);

                    err_train = 1.0 - ((float)match_count_train/(float)num_sample_train);
                    err_valid = 1.0-((float)match_count_valid/(float)num_sample_valid);

                    if((err_valid <= err_valid_best)&&(cost_valid <= cost_valid_best)) {
                        ind_early_stopping = ind_itr;
                        err_valid_best = err_valid;
                        cost_valid_best = cost_valid;

                        // save best trained model
                        if(en_save_best_model) {
                            if(en_gpu_model) {
                                cuda_copy_mat(emb_q.dev_w_mat,emb_q.dev_w_mat_best,dim_input,dim_emb,false);

                                for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                                    cuda_copy_mat(emb_m[i_hop].dev_w_mat,emb_m[i_hop].dev_w_mat_best,dim_input,dim_emb,false);
                                    cuda_copy_mat(emb_c[i_hop].dev_w_mat,emb_c[i_hop].dev_w_mat_best,dim_input,dim_emb,false);

                                    if(!en_remove_softmax) {
                                        if(f_en_sc_att) {
                                            cuda_copy_mat(sc_sf_in[i_hop].dev_w,sc_sf_in[i_hop].dev_w_best,n_sen,1,false);
                                        }
                                    }

                                    if(en_lin_map) {
                                        cuda_copy_mat(lin_map[i_hop].dev_w_mat,lin_map[i_hop].dev_w_mat_best,dim_emb,dim_emb,false);
                                    }
                                }

                                cuda_copy_mat(ds_ans.dev_w_mat,ds_ans.dev_w_mat_best,dim_emb,dim_input,false);
                                //cuda_copy_mat(ds_ans.dev_w_mat,ds_ans.dev_w_mat_best,dim_input,dim_emb,false);
                            }
                        }
                    }

                    if(verbose==true) {
                        if(en_valid) {
                            printf(" (train,valid,valid_best) - loss: %f, %f, %f, error: %f, %f, %f\n",cost_train,cost_valid,cost_valid_best,err_train,err_valid,err_valid_best);
                        } else {
                            printf(" (train) - loss: %f, error: %f\n",cost_train,err_train);
                        }
                    }

                    // training stopping, ealry stopping
                    //if( (err_valid > (err_valid_best+0.1)) || (err_valid==err_valid_min) ) {
                    /*
                    if( (err_valid > (err_valid_best+0.3)) ) {
                        ind_itr = num_itr-1;
                    }
                    */

                    if(en_save_best_model) {
                        if(((ind_itr-ind_early_stopping) > COUNT_EARLY_STOPPING) && (err_valid > (err_valid_best+0.3))){
                            ind_itr = num_itr-1;
                        }
                    }

                    //
                	if(f_en_sample_shuffled || (ind_itr==num_itr-1)) {
						sample_vectorization_destructor(train_sample, ind_train_shuffled, num_sample_train);

    					free(m_train);
    					free(q_train);
    					free(a_train);
    
    					free(n_sen_train_arr);
    					free(addr_m_arr_train);
    					free(addr_q_arr_train);
    					free(addr_a_arr_train);
					}
                }

                time_train_e = clock();

                time_train_arr[ind_task_loop] = (double)(time_train_e-time_train_s)/(double)CLOCKS_PER_SEC;
                err_train_arr[ind_task_loop] = err_train;

                if(en_joint) {
                    done_joint_training = true;
                    en_train = ( en_train && (!done_joint_training) );
                }

                //
                sample_vectorization_destructor(train_sample, ind_valid_shuffled, num_sample_valid);

    			free(m_valid);
    			free(q_valid);
    			free(a_valid);
    
    			free(n_sen_valid_arr);
    			free(addr_m_arr_valid);
    			free(addr_q_arr_valid);
    			free(addr_a_arr_valid);

                free(ind_sample_shuffled);
                free(ind_train_shuffled);
                free(ind_valid_shuffled);
            }

            // test
            printf("< Test Phase >\n");

            // init time array
            for(i=0;i<NUM_LAYER;i++) {
                for(j=0;j<NUM_LAYER_OP;j++) {
                    time_profile_test[i][j] = 0.0;
                }
            }

            // fpga
            if(en_fpga) {
                f_fpga_write = fopen("/dev/xillybus_write_16", "wb");
                f_fpga_read = fopen("/dev/xillybus_read_16", "rb");
            }

            time_test_tmp = 0.0;
            time_test_s = clock();


            ind_test_shuffled = (unsigned int*) malloc(num_sample_test*sizeof(unsigned int));

            err_test = 0.0;
            match_count_test=0;

            for(ind_sample=0;ind_sample<num_sample_test;ind_sample++) {
                ind_test_shuffled[ind_sample] = ind_sample;
            }

            sample_vectorization(test_sample, &dict, ind_test_shuffled, num_sample_test, null_ind, f_enable_time, 0, en_pe, pe_w, RAND_NOISE_TIME);

			n_sen_test_total = 0;
			n_sen_test_arr = (unsigned int *) malloc(num_sample_test*sizeof(unsigned int));

			addr_m_arr_test = (unsigned int *) malloc(num_sample_test*sizeof(unsigned int));
			addr_q_arr_test = (unsigned int *) malloc(num_sample_test*sizeof(unsigned int));
			addr_a_arr_test = (unsigned int *) malloc(num_sample_test*sizeof(unsigned int));

			for(ind_sam=0;ind_sam<num_sample_test;ind_sam++) {
	            p_sam = &(test_sample[ind_test_shuffled[ind_sam]]);

				n_sen_test_arr[ind_sam] = p_sam->n_sen;
    	        n_sen_test_total += p_sam->n_sen;
			}

			m_test = (float *) malloc(n_sen_test_total*dim_input*sizeof(float));
			q_test = (float *) malloc(num_sample_test*dim_input*sizeof(float));
			a_test = (float *) malloc(num_sample_test*dim_input*sizeof(float));

			addr_m = 0;
			addr_q = 0;
			addr_a = 0;

			for(ind_sam=0;ind_sam<num_sample_test;ind_sam++) {
				addr_m_arr_test[ind_sam] = addr_m;
				addr_q_arr_test[ind_sam] = addr_q;
				addr_a_arr_test[ind_sam] = addr_a;

                p_sam = &(test_sample[ind_test_shuffled[ind_sam]]);
                m = p_sam->sentences_b;
                q = p_sam->question_b;
                a = p_sam->answer_b;

				memcpy(&m_test[addr_m],m[0],n_sen_test_arr[ind_sam]*dim_input*sizeof(float));
				memcpy(&q_test[addr_q],q,dim_input*sizeof(float));
				memcpy(&a_test[addr_a],a,dim_input*sizeof(float));

				addr_m += n_sen_test_arr[ind_sam]*dim_input;
				addr_q += dim_input;
				addr_a += dim_input;
			}


            if(en_gpu_model) {
                cuda_data_in
                (
                    dev_m_test,
                    dev_q_test,
                    dev_a_test,
                    m_test,
                    q_test,
                    a_test,
                    n_sen_test_total,
                    //max_line,
                    dim_input,
					num_sample_test
                );
            }

            // load saved model
            if(en_save_best_model) {
                if(en_gpu_model) {
                    cuda_copy_mat(emb_q.dev_w_mat_best,emb_q.dev_w_mat,dim_input,dim_emb,false);

                    for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                        cuda_copy_mat(emb_m[i_hop].dev_w_mat_best,emb_m[i_hop].dev_w_mat,dim_input,dim_emb,false);
                        cuda_copy_mat(emb_c[i_hop].dev_w_mat_best,emb_c[i_hop].dev_w_mat,dim_input,dim_emb,false);

                        if(!en_remove_softmax) {
                            if(f_en_sc_att) {
                                cuda_copy_mat(sc_sf_in[i_hop].dev_w_best,sc_sf_in[i_hop].dev_w,n_sen,1,false);
                            }
                        }

                        if(en_lin_map) {
                            cuda_copy_mat(lin_map[i_hop].dev_w_mat_best,lin_map[i_hop].dev_w_mat,dim_emb,dim_emb,false);
                        }
                    }

                    cuda_copy_mat(ds_ans.dev_w_mat_best,ds_ans.dev_w_mat,dim_emb,dim_input,false);
                    //cuda_copy_mat(ds_ans.dev_w_mat_best,ds_ans.dev_w_mat,dim_input,dim_emb,false);
                }
            }

            if(en_cpu || en_gpu_model) {
                for(ind_sam=0;ind_sam<num_sample_test;ind_sam++) {
					/*
                    p_sam = &(test_sample[ind_sam]);
                    m = p_sam->sentences_b;
                    q = p_sam->question_b;
                    a = p_sam->answer_b;
                    //ai = p_sam->answer_i.words[0];

                    //printf("%d\n",ai);

                    n_sen = p_sam->n_sen;

                    if(en_gpu_model) {
                        cuda_data_in
                        (
                            dev_m,
                            dev_q,
                            dev_a,
                            m[0],
                            q,
                            a,
                            n_sen,
                            dim_input,
							1
                        );
                    }
					*/

					n_sen = n_sen_test_arr[ind_sam];
                    ////////////////////////////////////////
                    // input setting - test
                    ////////////////////////////////////////
                    time_profile_test[2][2] += dense_in(&emb_q, q, dup_grad[0], &dev_q_test[addr_q_arr_test[ind_sam]], dev_dup_grad[0]);

                    emb_q_vec = emb_q.out_vec;
                    dev_emb_q_vec = emb_q.dev_out_vec;

                    for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                       time_profile_test[0][2] += dense_mat_in(&emb_m[i_hop], n_sen, m, dotmv[i_hop].grad_out_mat, &dev_m_test[addr_m_arr_test[ind_sam]], dotmv[i_hop].dev_grad_out_mat);
                       time_profile_test[1][2] += dense_mat_in(&emb_c[i_hop], n_sen, m, w_sum[i_hop].grad_out_mat, &dev_m_test[addr_m_arr_test[ind_sam]], w_sum[i_hop].dev_grad_out_mat);

                        if(en_remove_softmax) {
                            dotmv_grad_in = w_sum[i_hop].grad_out_vec;
                            dev_dotmv_grad_in = w_sum[i_hop].dev_grad_out_vec;
                        } else {
                            if(f_en_sc_att) {
                                dotmv_grad_in = sc_sf_in[i_hop].grad_out;
                                dev_dotmv_grad_in = sc_sf_in[i_hop].dev_grad_out;
                            } else {
                                dotmv_grad_in = sf_in[i_hop].grad_out;
                                dev_dotmv_grad_in = sf_in[i_hop].dev_grad_out;
                            }
                        }

                        if(i_hop==0) {
                            time_profile_test[3][2] += dot_mat_vec_in(&dotmv[i_hop], n_sen, emb_m[i_hop].out_mat, emb_q_vec, dotmv_grad_in, emb_m[i_hop].dev_out_mat, dev_emb_q_vec, dev_dotmv_grad_in);
                        } else {
                            if(en_non_lin) {
                                dotmv_in = non_lin[i_hop-1].out;
                                dev_dotmv_in = non_lin[i_hop-1].dev_out;
                            } else {
                                dotmv_in = sv[i_hop-1].out_vec;
                                dev_dotmv_in = sv[i_hop-1].dev_out_vec;
                            }

                            time_profile_test[3][2] += dot_mat_vec_in(&dotmv[i_hop], n_sen, emb_m[i_hop].out_mat, dotmv_in, dotmv_grad_in, emb_m[i_hop].dev_out_mat, dev_dotmv_in, dev_dotmv_grad_in);
                        }

                        if(f_en_sc_att) {
                            time_profile_test[4][2] += scale_in(&sc_sf_in[i_hop], n_sen, dotmv[i_hop].out_vec, sf_in[i_hop].grad_out, dotmv[i_hop].dev_out_vec, sf_in[i_hop].dev_grad_out);
                            time_profile_test[4][2] += softmax_in(&sf_in[i_hop], n_sen, sc_sf_in[i_hop].out, w_sum[i_hop].grad_out_vec, sc_sf_in[i_hop].dev_out, w_sum[i_hop].dev_grad_out_vec);
                        } else {
                            time_profile_test[4][2] += softmax_in(&sf_in[i_hop], n_sen, dotmv[i_hop].out_vec, w_sum[i_hop].grad_out_vec, dotmv[i_hop].dev_out_vec, w_sum[i_hop].dev_grad_out_vec);
                        }

                        if(en_remove_softmax) {
                            w_sum_in_vec = dotmv[i_hop].out_vec;
                            dev_w_sum_in_vec = dotmv[i_hop].dev_out_vec;
                        } else {
                            w_sum_in_vec = sf_in[i_hop].out_vec;
                            dev_w_sum_in_vec = sf_in[i_hop].dev_out_vec;
                        }

                        time_profile_test[5][2] += dot_mat_vec_in(&w_sum[i_hop], n_sen, emb_c[i_hop].out_mat, w_sum_in_vec, sv[i_hop].grad_out, emb_c[i_hop].dev_out_mat, dev_w_sum_in_vec, sv[i_hop].dev_grad_out);

                        if(en_lin_map) {
                            lin_map_grad_in = sv[i_hop].grad_out;
                            dev_lin_map_grad_in = sv[i_hop].dev_grad_out;

                            if(i_hop==0) {
                                lin_map_in = emb_q_vec;
                                dev_lin_map_in = dev_emb_q_vec;
                            } else {
                                lin_map_in = sv[i_hop-1].out_vec;
                                dev_lin_map_in = sv[i_hop-1].dev_out_vec;
                            }

                            time_profile_test[6][2] += dense_in(&lin_map[i_hop], lin_map_in, lin_map_grad_in, dev_lin_map_in, dev_lin_map_grad_in);
                        }

                        if(en_lin_map) {
                            sv_in_vec_a = lin_map[i_hop].out_vec;
                            sv_dev_in_vec_a = lin_map[i_hop].dev_out_vec;
                        } else {
                            if(i_hop==0) {
                                sv_in_vec_a = emb_q_vec;
                                sv_dev_in_vec_a = dev_emb_q_vec;
                            } else {
                                sv_in_vec_a = sv[i_hop-1].out_vec;
                                sv_dev_in_vec_a = sv[i_hop-1].dev_out_vec;
                            }
                        }

                        if(en_non_lin) {
                            // test_170406
                            /*
                            if(i_hop==NUM_HOP-1) {
                                sv_grad_in = ds_ans.grad_out;
                                sv_dev_grad_in = ds_ans.dev_grad_out;
                            } else {
                                sv_grad_in = non_lin[i_hop].grad_out;
                                sv_dev_grad_in = non_lin[i_hop].dev_grad_out;
                            }
                            */
                            sv_grad_in = non_lin[i_hop].grad_out;
                            sv_dev_grad_in = non_lin[i_hop].dev_grad_out;
                        } else {
                            if(i_hop==NUM_HOP-1) {
                                sv_grad_in = ds_ans.grad_out;
                                sv_dev_grad_in = ds_ans.dev_grad_out;
                            } else {
                                sv_grad_in = dup_grad[i_hop+1];
                                //sv_dev_grad_in = dev_dup_grad+(i_hop+1)*dim_emb;
                                sv_dev_grad_in = dev_dup_grad[i_hop+1];
                            }
                        }

                        time_profile_test[6][2] += sum_vec_in(&sv[i_hop], sv_in_vec_a, w_sum[i_hop].out_vec, sv_grad_in, sv_dev_in_vec_a, w_sum[i_hop].dev_out_vec, sv_dev_grad_in);

                        if(en_non_lin) {
                            non_lin_in = sv[i_hop].out_vec;
                            non_lin_dev_in = sv[i_hop].dev_out_vec;

                            if(i_hop==NUM_HOP-1) {
                                non_lin_grad_in = ds_ans.grad_out;
                                non_lin_dev_grad_in = ds_ans.dev_grad_out;
                            } else {
                                non_lin_grad_in = dup_grad[i_hop+1];
                                //non_lin_dev_grad_in = dev_dup_grad+(i_hop+1)*dim_emb;
                                non_lin_dev_grad_in = dev_dup_grad[i_hop+1];
                            }

                            time_profile_test[6][2] += activation_in(&non_lin[i_hop], non_lin_in, non_lin_grad_in, non_lin_dev_in, non_lin_dev_grad_in);
                        }
                    }

                    // test_170406
                    if(en_non_lin) {
                        ds_ans_in = non_lin[NUM_HOP-1].out;
                        dev_ds_ans_in = non_lin[NUM_HOP-1].dev_out;
                    } else {
                        ds_ans_in = sv[NUM_HOP-1].out_vec;
                        dev_ds_ans_in = sv[NUM_HOP-1].dev_out_vec;
                    }
                    //ds_ans_in = sv[NUM_HOP-1].out_vec;
                    //dev_ds_ans_in = sv[NUM_HOP-1].dev_out_vec;


                    time_profile_test[7][2] += dense_in(&ds_ans, ds_ans_in, ce.grad_out, dev_ds_ans_in, ce.dev_grad_out);
                    time_profile_test[8][2] += softmax_in(&sf_out, dim_input, ds_ans.out_vec, ce.grad_out, ds_ans.dev_out_vec, ce.dev_grad_out);
                    time_profile_test[9][2] += cross_entropy_in(&ce, sf_out.out_vec, a, sf_out.dev_out_vec, &dev_a_test[addr_a_arr_test[ind_sam]]);

                    time_test_e = clock();
                    time_test_tmp += (double)(time_test_e-time_test_s)/(double)CLOCKS_PER_SEC;

                    /*
                    // load weight from file
                    if(ind_sam==0) {
                        if(en_load_weight) {
                            // Weight Load - floating point
                            printf("< Weight Load - floating point >\n");
                            FILE *f_w_float;
                            float tmp_float;

                            // binary file format
                                // emb a
                                f_w_float = fopen("./w_emb_a_float.bin","rb");
                                //f_w_float = fopen("./result_emb_a_b.bin","rb");

                                for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                                    for(j=0;j<emb_m[i_hop].dim_in;j++) {
                                        for(i=0;i<emb_m[i_hop].dim_out;i++) {
                                            fread(&tmp_float,sizeof(tmp_float),1,f_w_float);
                                            emb_m[i_hop].w_mat[i][j] = tmp_float;
                                        }
                                    }
                                }

                                fclose(f_w_float);

                                // emb c
                                f_w_float = fopen("./w_emb_c_float.bin","rb");
                                //f_w_float = fopen("./result_emb_c_b.bin","rb");

                                for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                                    for(j=0;j<emb_c[i_hop].dim_in;j++) {
                                        for(i=0;i<emb_c[i_hop].dim_out;i++) {
                                            fread(&tmp_float,sizeof(tmp_float),1,f_w_float);
                                            emb_c[i_hop].w_mat[i][j] = tmp_float;
                                        }
                                    }
                                }

                                fclose(f_w_float);

                                // emb q
                                f_w_float = fopen("./w_emb_q_float.bin","rb");
                                //f_w_float = fopen("./result_emb_q_b.bin","rb");

                                for(j=0;j<emb_q.dim_in;j++) {
                                    for(i=0;i<emb_q.dim_out;i++) {
                                        fread(&tmp_float,sizeof(tmp_float),1,f_w_float);
                                        emb_q.w_mat[i][j] = tmp_float;
                                    }
                                }
                                fclose(f_w_float);

                                // w
                                f_w_float = fopen("./w_float.bin","rb");
                                //f_w_float = fopen("./result_w_b.bin","rb");

                                for(j=0;j<ds_ans.dim_in;j++) {
                                    for(i=0;i<ds_ans.dim_out;i++) {
                                        fread(&tmp_float,sizeof(tmp_float),1,f_w_float);
                                        ds_ans.w_mat[i][j] = tmp_float;
                                    }
                                }
                                fclose(f_w_float);
                        }
                    }
                    */

                    time_test_s = clock();
                    ////////////////////////////////////////
                    // forward propagation - test
                    ////////////////////////////////////////
                    //printf("Forward Propagation\n");
                    //printf("emb_q\n");
                    time_profile_test[2][3] += dense_fwd(&emb_q, verbose_debug);

                    for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                        //printf("emb_m\n");
                        time_profile_test[0][3] += dense_mat_fwd(&emb_m[i_hop], verbose_debug);

                        //printf("emb_c\n");
                        time_profile_test[1][3] += dense_mat_fwd(&emb_c[i_hop], verbose_debug);

                        //printf("dotmv\n");
                        //verbose_debug = true;
                        //
                        /*
                        if(ind_itr > 9) {
                            verbose_debug = true;
                        }
                        */

                        time_profile_test[3][3] += dot_mat_vec_fwd(&dotmv[i_hop], verbose_debug);

                        if(!en_remove_softmax) {
                            if(f_en_sc_att) {
                                time_profile_test[4][3] += scale_fwd(&sc_sf_in[i_hop], verbose_debug);
                            }
                            //printf("sf_in\n");
                            time_profile_test[4][3] += softmax_fwd(&sf_in[i_hop], verbose_debug);
                        }

                        //verbose_debug = false;

                        //printf("w_sum\n");
                        time_profile_test[5][3] += dot_mat_vec_fwd(&w_sum[i_hop], verbose_debug);


                        if(en_lin_map) {
                            //printf("lin_map\n");
                            time_profile_test[6][3] += dense_fwd(&lin_map[i_hop], verbose_debug);
                        }

                        //printf("sv\n");
                        time_profile_test[6][3] += sum_vec_fwd(&sv[i_hop], verbose_debug);

                        if(en_non_lin) {
                            //printf("non_lin\n");
                            time_profile_test[6][3] += activation_fwd(&non_lin[i_hop], verbose_debug);
                        }
                    }

                    time_profile_test[7][3] += dense_fwd(&ds_ans, verbose_debug);

                    time_profile_test[8][3] += softmax_fwd(&sf_out, verbose_debug);

					/*
                    predict = sf_out.out_vec;

                    predict_i = 0;
                    for(j=0;j<dim_input;j++) {
                        //printf("%f %f\n",predict[predict_i],predict[j]);
                        //printf("%d %d %d\n",j,ai,predict_i);
                        if(predict[predict_i] < predict[j]) {
                            predict_i = j;
                        }
                    }

                    if( predict_i == ai ) {
                        match_count++;
                    }
					*/
                    ////////////////////////////////////////
                    // cost function
                    ////////////////////////////////////////
                    time_profile_test[9][3] += cross_entropy_run(&ce,3);

                }

				cross_entropy_cost_load(&ce,&cost_train,&cost_valid,&cost_test);
				cross_entropy_m_cnt_load(&ce, &match_count_train, &match_count_valid, &match_count_test);
                err_test = 1.0-((float)match_count_test/(float)num_sample_test);


            } else if(en_fpga) {
                arg_stream_io arg_write;
                arg_stream_io arg_read;

                arg_write.f_file = f_fpga_write;
                arg_write.p_sam = test_sample;        // modify_point
                //arg_write.p_sam = train_sample;        // modify_point
                arg_write.num_sample_test = num_sample_test;

                arg_read.f_file = f_fpga_read;
                arg_read.p_sam = test_sample;         // modify_point
                //arg_read.p_sam = train_sample;         // modify_point
                arg_read.num_sample_test = num_sample_test;

                // threading
                printf("multi threading...\n");
                thread_write_id = pthread_create(&thread_write, NULL, stream_write, (void *)&arg_write);
                if(thread_write_id<0) {
                    perror("*E : thread write create\n");
                    exit(0);
                }

                thread_read_id = pthread_create(&thread_read, NULL, stream_read, (void *)&arg_read);
                if(thread_read_id<0) {
                    perror("*E : thread read create\n");
                    exit(0);
                }

                pthread_join(thread_write, (void **)&thread_write_status);
                pthread_join(thread_read, (void **)&thread_read_status);

                err_test = arg_read.err_test;
            }

            time_test_e = clock();


            printf("err(test) : %f\n",err_test);

            time_test_arr[ind_task_loop] = (double)(time_test_e-time_test_s)/(double)CLOCKS_PER_SEC + (double)time_test_tmp;
            err_test_arr[ind_task_loop] = err_test;

            if(en_fpga) {
                fclose(f_fpga_write);
                fclose(f_fpga_read);
            }

            sample_vectorization_destructor(test_sample, ind_test_shuffled, num_sample_test);

            free(ind_test_shuffled);

  			free(m_test);
   			free(q_test);
   			free(a_test);
    
   			free(n_sen_test_arr);
   			free(addr_m_arr_test);
   			free(addr_q_arr_test);
   			free(addr_a_arr_test);
        }

        /*
        // binarization test
        FILE *ftest;
        ftest = fopen("tt.bin","wb");

        packet_16 pkt_fpga_write;

        for(i=0;i<num_sample_test;i++) {
            for(j=0;j<test_sample[i].n_sen;j++) {
                for(k=0;k<test_sample[i].sentences[j].n+1;k++) {
                    pkt_fpga_write = test_sample[i].sentences_i_p16[j][k];
                    fwrite(&pkt_fpga_write,sizeof(pkt_fpga_write),1,ftest);
                }
            }

            for(j=0;j<test_sample[i].question.n;j++) {
                pkt_fpga_write = test_sample[i].question_i_p16[j];
                fwrite(&pkt_fpga_write,sizeof(pkt_fpga_write),1,ftest);
            }

            for(j=0;j<test_sample[i].answer.n;j++) {
                pkt_fpga_write = test_sample[i].answer_i_p16[j];
                fwrite(&pkt_fpga_write,sizeof(pkt_fpga_write),1,ftest);
            }
        }
        fclose(ftest);
        */


        //
        time_train_avg = 0.0;
        time_test_avg = 0.0;
        err_train_avg = 0.0;
        err_test_avg = 0.0;

        time_train_max = 0.0;
        time_test_max = 0.0;
        err_train_max = 0.0;
        err_test_max = 0.0;

        time_train_min = time_train_arr[0];
        time_test_min = time_test_arr[0];
        err_train_min = err_train_arr[0];
        err_test_min = err_test_arr[0];


        for(ind_task_loop=0;ind_task_loop<num_task_loop;ind_task_loop++) {
            time_train_avg += time_train_arr[ind_task_loop];
            time_test_avg += time_test_arr[ind_task_loop];
            err_train_avg += err_train_arr[ind_task_loop];
            err_test_avg += err_test_arr[ind_task_loop];

            if(time_train_max < time_train_arr[ind_task_loop]) {
                time_train_max = time_train_arr[ind_task_loop];
            }

            if(time_test_max < time_test_arr[ind_task_loop]) {
                time_test_max = time_test_arr[ind_task_loop];
            }

            if(err_train_max < err_train_arr[ind_task_loop]) {
                err_train_max = err_train_arr[ind_task_loop];
            }

            if(err_test_max < err_test_arr[ind_task_loop]) {
                err_test_max = err_test_arr[ind_task_loop];
            }


            if(time_train_min > time_train_arr[ind_task_loop]) {
                time_train_min = time_train_arr[ind_task_loop];
            }

            if(time_test_min > time_test_arr[ind_task_loop]) {
                time_test_min = time_test_arr[ind_task_loop];
            }

            if(err_train_min > err_train_arr[ind_task_loop]) {
                err_train_min = err_train_arr[ind_task_loop];
            }

            if(err_test_min > err_test_arr[ind_task_loop]) {
                err_test_min = err_test_arr[ind_task_loop];
            }
        }

        if(en_write_weight) {
            // Weight Write - floating point
			printf("write_weight: should be modified\n");
			/*
            printf("< Weight Write - floating point >\n");
            FILE *f_w_float;
            float tmp_float;

            // emb a
            f_w_float = fopen("./w_emb_a_float.bin","wb");

            for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                for(j=0;j<emb_m[i_hop].dim_in;j++) {
                    for(i=0;i<emb_m[i_hop].dim_out;i++) {
                        tmp_float = emb_m[i_hop].w_mat[i][j];
                        fwrite(&tmp_float,sizeof(tmp_float),1,f_w_float);
                    }
                }
            }
            fclose(f_w_float);

            // emb c
            f_w_float = fopen("./w_emb_c_float.bin","wb");

            for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                for(j=0;j<emb_c[i_hop].dim_in;j++) {
                    for(i=0;i<emb_c[i_hop].dim_out;i++) {
                        tmp_float = emb_c[i_hop].w_mat[i][j];
                        fwrite(&tmp_float,sizeof(tmp_float),1,f_w_float);
                    }
                }
            }
            fclose(f_w_float);

            // emb q
            f_w_float = fopen("./w_emb_q_float.bin","wb");

            for(j=0;j<emb_q.dim_in;j++) {
                for(i=0;i<emb_q.dim_out;i++) {
                    tmp_float = emb_q.w_mat[i][j];
                    fwrite(&tmp_float,sizeof(tmp_float),1,f_w_float);
                }
            }
            fclose(f_w_float);

            // w
            f_w_float = fopen("./w_float.bin","wb");

            for(j=0;j<ds_ans.dim_in;j++) {
                for(i=0;i<ds_ans.dim_out;i++) {
                    tmp_float = ds_ans.w_mat[i][j];
                    fwrite(&tmp_float,sizeof(tmp_float),1,f_w_float);
                }
            }

            fclose(f_w_float);


            // Weight Write - fixed point
            printf("< Weight Write - fixed point >\n");
            FILE *f_w_fixed;
            int tmp_fixed;

            // emb a
            f_w_fixed = fopen("./w_emb_a_fixed.bin","wb");

            for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                for(j=0;j<emb_m[i_hop].dim_in;j++) {
                    for(i=0;i<emb_m[i_hop].dim_out;i++) {
                        //printf("%08x ",FLOAT2FIXED(emb_m[0].w_mat[i][j]));

                        tmp_fixed = FLOAT2FIXED(emb_m[i_hop].w_mat[i][j]);
                        fwrite(&tmp_fixed,sizeof(tmp_fixed),1,f_w_fixed);
                    }
                    //printf("\n");
                }
            }
            fclose(f_w_fixed);

            // emb c
            f_w_fixed = fopen("./w_emb_c_fixed.bin","wb");

            for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                for(j=0;j<emb_c[i_hop].dim_in;j++) {
                    for(i=0;i<emb_c[i_hop].dim_out;i++) {
                        //printf("%08x ",FLOAT2FIXED(emb_c[0].w_mat[i][j]));

                        tmp_fixed = FLOAT2FIXED(emb_c[i_hop].w_mat[i][j]);
                        fwrite(&tmp_fixed,sizeof(tmp_fixed),1,f_w_fixed);
                    }
                    //printf("\n");
                }
            }
            fclose(f_w_fixed);

            // emb q
            f_w_fixed = fopen("./w_emb_q_fixed.bin","wb");

            for(j=0;j<emb_q.dim_in;j++) {
                for(i=0;i<emb_q.dim_out;i++) {
                    //printf("%08x ",FLOAT2FIXED(emb_q.w_mat[i][j]));

                    tmp_fixed = FLOAT2FIXED(emb_q.w_mat[i][j]);
                    fwrite(&tmp_fixed,sizeof(tmp_fixed),1,f_w_fixed);
                }
                //printf("\n");
            }

            fclose(f_w_fixed);

            // w
            f_w_fixed = fopen("./w_fixed.bin","wb");

            for(j=0;j<ds_ans.dim_in;j++) {
                for(i=0;i<ds_ans.dim_out;i++) {
                    //printf("%08x ",FLOAT2FIXED(ds_ans.w_mat[i][j]));

                    tmp_fixed = FLOAT2FIXED(ds_ans.w_mat[i][j]);
                    fwrite(&tmp_fixed,sizeof(tmp_fixed),1,f_w_fixed);
                }
                //printf("\n");
            }

            fclose(f_w_fixed);
			*/
        }

        time_train_avg = time_train_avg/(float)num_task_loop;
        time_test_avg = time_test_avg/(float)num_task_loop;
        err_train_avg = err_train_avg/(float)num_task_loop;
        err_test_avg = err_test_avg/(float)num_task_loop;


        time_profile_train_total = 0.0;
        for(i=0;i<NUM_LAYER;i++) {
            for(j=0;j<NUM_LAYER_OP;j++) {
                time_profile_train_total += time_profile_train[i][j];
            }
        }

        time_profile_test_total = 0.0;
        for(i=0;i<NUM_LAYER;i++) {
            for(j=0;j<NUM_LAYER_OP;j++) {
                time_profile_test_total += time_profile_test[i][j];
            }
        }

        printf("< TIME PROFILE >\n");
        printf(" TRAIN \n");
        printf("Layer : Constructor : Init : In : Fwd : Bwd : W_up : Destructor\n");
        for(i=0;i<NUM_LAYER;i++) {
            //printf("[%d] : %s",i, layer_name[i]);
            printf("[%d] : ",i);
            for(j=0;j<NUM_LAYER_OP;j++) {
                printf("[%d] %f  ",j,time_profile_train[i][j]);
            }
            printf("\n");
        }


        printf("< TIME PROFILE >\n");
        printf(" TEST \n");
        for(i=0;i<NUM_LAYER;i++) {
            printf("[%d]",i);
            for(j=0;j<NUM_LAYER_OP;j++) {
                printf("[%d] %f  ",j,time_profile_test[i][j]);
            }
            printf("\n");
        }

        //
        //< Info. >
        printf("< Setting >\n");
        printf("   NUM_TAKS : %d\n",ind_data_set);
        printf("   HW_MODE : %d\n",HW_MODE);
        printf("   EN_TRAIN : %d\n", en_train);
        printf("   EN_LOAD_WEIGHT : %d\n", en_load_weight);
        printf("   EN_WRITE_WEIGHT : %d\n", en_write_weight);
        printf("   ATTENTION MODE : %s\n", name_attention_mode);

        if(attention_mode==1) {
        } else if(attention_mode==2) {
            printf("    BW_WL : %d, BW_IWL : %d, BW_FRAC : %d\n", BW_WL, iwl_argv, frac_argv);
        } else if(attention_mode==3) {
            printf("    approximate attention bit : %d\n",num_hamming_attention);
        } else if(attention_mode==4) {
        } else {
        }

        printf("< Result - Train >\n");
        printf(" -TIME\n");
        printf("   avg : %f\n",time_train_avg);
        printf("   max : %f\n",time_train_max);
        printf("   min : %f\n",time_train_min);
        printf("   profile total : %f\n",time_profile_train_total);
		printf("   data transfer time: %f\n",time_train_data);
        printf(" -ERROR RATE\n");
        printf("   avg : %f\n",err_train_avg);
        printf("   max : %f\n",err_train_max);
        printf("   min : %f\n",err_train_min);

        printf("< Result - Test >\n");
        printf(" -TIME\n");
        printf("   avg : %f\n",time_test_avg);
        printf("   max : %f\n",time_test_max);
        printf("   min : %f\n",time_test_min);
        printf("   profile total : %f\n",time_profile_test_total);
        printf(" -ERROR RATE\n");
        printf("   avg : %f\n",err_test_avg);
        printf("   max : %f\n",err_test_max);
        printf("   min : %f\n",err_test_min);

        // file io
        fp_result = fopen("result_all.csv","a");
        fprintf(fp_result,"%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,"
                ,ind_data_set
                ,time_train_avg,time_train_max,time_train_min
                ,err_train_avg,err_train_max,err_train_min
                ,time_test_avg,time_test_max,time_test_min
                ,err_test_avg,err_test_max,err_test_min);

        for(ind_task_loop=0;ind_task_loop<num_task_loop;ind_task_loop++) {
            fprintf(fp_result,"%f",err_test_arr[ind_task_loop]);
            if(ind_task_loop!=num_task_loop-1) {
                fprintf(fp_result,",");
            } else {
                fprintf(fp_result,"\n");
            }
        }
        fclose(fp_result);

        fp_result = fopen("result.csv","a");
        fprintf(fp_result,"%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,"
                ,ind_data_set
                ,time_train_avg,time_train_max,time_train_min
                ,err_train_avg,err_train_max,err_train_min
                ,time_test_avg,time_test_max,time_test_min
                ,err_test_avg,err_test_max,err_test_min);

        for(ind_task_loop=0;ind_task_loop<num_task_loop;ind_task_loop++) {
            fprintf(fp_result,"%f",err_test_arr[ind_task_loop]);
            if(ind_task_loop!=num_task_loop-1) {
                fprintf(fp_result,",");
            } else {
                fprintf(fp_result,"\n");
            }
        }

        fclose(fp_result);

        //
        if((!en_joint)||(en_joint&(ind_data_set==ind_data_set_e))) {
            dense_destructor(&emb_q);
            for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                dense_mat_destructor(&emb_m[i_hop]);
                dense_mat_destructor(&emb_c[i_hop]);
                dot_mat_vec_destructor(&dotmv[i_hop]);
    
                if(f_en_sc_att) {
                    scale_destructor(&sc_sf_in[i_hop]);
                }
                softmax_destructor(&sf_in[i_hop]);
                dot_mat_vec_destructor(&w_sum[i_hop]);
                if(en_lin_map) {
                    dense_destructor(&lin_map[i_hop]);
                }
                sum_vec_destructor(&sv[i_hop]);
    
                if(en_non_lin) {
                    activation_destructor(&non_lin[i_hop]);
                }
            }
    
            dense_destructor(&ds_ans);
            softmax_destructor(&sf_out);
            cross_entropy_destructor(&ce);
    
            //
            dictionary_destructor(&dict);

            //
            if(en_gpu_model) {
                cuda_data_destructor
                (
                    dev_m_train,
                    dev_q_train,
                    dev_a_train
                );

                cuda_data_destructor
                (
                    dev_m_valid,
                    dev_q_valid,
                    dev_a_valid
                );

                cuda_data_destructor
                (
                    dev_m_test,
                    dev_q_test,
                    dev_a_test
                );
    
                for(i_hop=0;i_hop<NUM_HOP;i_hop++) {
                    cuda_dup_grad_destructor
                    (
                        dev_dup_grad[i_hop]
                    );
                }
            }
    
            free(dup_grad[0]);
            free(dup_grad);
    
            free(pe_w[0]);
            free(pe_w);
    
            free(emb_m);
            free(emb_c);
    
            free(dotmv);
            free(sf_in);
            free(w_sum);
    
            if(en_lin_map) {
                free(lin_map);
            }
    
            free(sv);
    
            if(en_non_lin) {
                free(non_lin);
            }

        }

    }

    //
    free(time_train_arr);
    free(time_test_arr);
    free(err_train_arr);
    free(err_test_arr);

    return 0;
}

// host -> fpga
void *stream_write(void *arg) {
    printf("thread_write : start stream_write\n");
    arg_stream_io *p_arg = (arg_stream_io *) arg;
    FILE *f_fpga_write = p_arg->f_file;
    sample *p_sam = p_arg->p_sam;
    unsigned int num_sample_test = p_arg->num_sample_test;

    unsigned int ind_sam, j, k;
    int rc;

    packet_16 pkt_fpga_write;

    printf(" stream_write : num_sample_test : %d\n",num_sample_test);

    for(ind_sam=0;ind_sam<num_sample_test;ind_sam++) {
        for(j=0;j<p_sam[ind_sam].n_sen;j++) {
            for(k=0;k<p_sam[ind_sam].sentences[j].n+1;k++) {
                pkt_fpga_write = p_sam[ind_sam].sentences_i_p16[j][k];
                fwrite(&pkt_fpga_write,sizeof(pkt_fpga_write),1,f_fpga_write);
                //printf("%04x\n",pkt_fpga_write);
            }
        }

        for(j=0;j<p_sam[ind_sam].question.n;j++) {
            pkt_fpga_write = p_sam[ind_sam].question_i_p16[j];
            fwrite(&pkt_fpga_write,sizeof(pkt_fpga_write),1,f_fpga_write);
                //printf("%04x\n",pkt_fpga_write);
        }

        for(j=0;j<p_sam[ind_sam].answer.n;j++) {
            pkt_fpga_write = p_sam[ind_sam].answer_i_p16[j];
            fwrite(&pkt_fpga_write,sizeof(pkt_fpga_write),1,f_fpga_write);
                //printf("%04x\n",pkt_fpga_write);
        }
    }

    //dummy
    for(ind_sam=0;ind_sam<num_sample_test;ind_sam++) {
        for(j=0;j<p_sam[ind_sam].n_sen;j++) {
            for(k=0;k<p_sam[ind_sam].sentences[j].n+1;k++) {
                pkt_fpga_write = p_sam[ind_sam].sentences_i_p16[j][k];
                fwrite(&pkt_fpga_write,sizeof(pkt_fpga_write),1,f_fpga_write);
            }
        }

        for(j=0;j<p_sam[ind_sam].question.n;j++) {
            pkt_fpga_write = p_sam[ind_sam].question_i_p16[j];
            fwrite(&pkt_fpga_write,sizeof(pkt_fpga_write),1,f_fpga_write);
        }

        for(j=0;j<p_sam[ind_sam].answer.n;j++) {
            pkt_fpga_write = p_sam[ind_sam].answer_i_p16[j];
            fwrite(&pkt_fpga_write,sizeof(pkt_fpga_write),1,f_fpga_write);
        }
    }
}

// fpga -> host
void *stream_read(void* arg) {
    printf("thread_read : start stream_read\n");
    arg_stream_io *p_arg = (arg_stream_io *) arg;
    FILE *f_fpga_read = p_arg->f_file;
    sample *p_sam = p_arg->p_sam;
    unsigned int num_sample_test = p_arg->num_sample_test;
    packet_16 pkt_fpga_read;

    unsigned int ind_sam;
    unsigned int match_count=0;

    printf("num_sample_test : %d\n",num_sample_test);

    fread(&pkt_fpga_read,sizeof(pkt_fpga_read),1,f_fpga_read);
    for(ind_sam=0;ind_sam<num_sample_test;ind_sam++) {
        fread(&pkt_fpga_read,sizeof(pkt_fpga_read),1,f_fpga_read);

        printf("[%04d] : %d : %d\n",ind_sam, p_sam[ind_sam].answer_i.words[0], pkt_fpga_read.addr);

        if(p_sam[ind_sam].answer_i.words[0]==(int)pkt_fpga_read.addr) {
            match_count++;
        }
    }


    p_arg->err_test= 1.0-((float)match_count/(float)num_sample_test);


    printf("thread_read : read_done\n");
}

