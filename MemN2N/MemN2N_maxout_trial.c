#include "MemN2N.h"

int main(int argc, char*argv[])
{
    char tmp_word[MAX_WORD_LEN];

    sample *train_sample;
    sample *test_sample;

    unsigned int null_ind;

    unsigned int max_line;      // max number of sentences in a sample
    unsigned int max_word;      // max number of words in a sentence
    unsigned int dim_dict;      // max number of words in dict

    unsigned int i, j, k, l;

    unsigned int dim_input;
    unsigned int dim_word;


    bool f_enable_time = EN_TIME;
    bool f_exp_plan = EN_EXP_TABLE_BASED;
    bool en_max_grad_l2_norm = EN_MAX_GRAD_L2_NORM;
    float max_grad_l2_norm = MAX_GRAD_L2_NORM;

    sample *p_sam;
    float **m;
    float *q;
    float *a;
    float lr;
    float lambda;

    float **c;
    unsigned int ind_sam, ind_itr, ind_sen, ind_batch;
    float cost;
    bool verbose = false;
    float *grad;
    unsigned int dim_emb = DIM_EMB;

    float acc_train;
    float acc_test;

    float *predict;
    unsigned int predict_i;
    unsigned int match_count;

    unsigned int ai;

    unsigned int size_batch=32;
    unsigned int num_batch;
    unsigned int size_b;

    unsigned int n_sen;

    unsigned int n_data;

    char path_train[100];
    char path_test[100];

    clock_t time_train_s;
    clock_t time_train_e;
    clock_t time_test_s;
    clock_t time_test_e;
    
    unsigned int num_test_loop;
    unsigned int ind_test_loop;

    float *time_train_arr;
    float *time_test_arr;
    float *acc_train_arr;
    float *acc_test_arr;

    float time_train_avg;
    float time_test_avg;
    float acc_train_avg;
    float acc_test_avg;

    float time_train_max;
    float time_test_max;
    float acc_train_max;
    float acc_test_max;
    
    float time_train_min;
    float time_test_min;
    float acc_train_min;
    float acc_test_min;

    
    bool f_msg = true;

    


    if( argc != 2 ) {
        num_test_loop = 1;
    } else {
        num_test_loop = atoi(argv[1]);
    }
    printf("< NUM TEST LOOP : %d >\n",num_test_loop);

    //
    time_train_arr = (float *) malloc(num_test_loop*sizeof(float));
    time_test_arr = (float *) malloc(num_test_loop*sizeof(float));
    acc_train_arr = (float *) malloc(num_test_loop*sizeof(float));
    acc_test_arr = (float *) malloc(num_test_loop*sizeof(float));
    


    printf("< LIST DATA SET >\n");
    printf("    %2d : %s\n", 0, "TEST_ALL(NOT YET)");
    for(i=0;i<NUM_LIST_DATA;i++) {
        printf("    %2d : %s\n",i+1, LIST_DATA_SET[i]);
    }

    while(true) {
        printf("\ninput number of data set : ");
        scanf("%d", &n_data);

        if( !((n_data > 0)&&(n_data<=NUM_LIST_DATA))) {
            printf("*E : Number of Data Set - Out of Range\n");
        } else {
            printf("<  Set : %2d : %s>\n",n_data,LIST_DATA_SET[n_data-1]);
            break;
        }
    }
    
    strcpy(path_train,PATH_DATA_SET);
    strcat(path_train,LIST_DATA_SET[n_data-1]);
    strcat(path_train,"_train_set");

    strcpy(path_test,PATH_DATA_SET);
    strcat(path_test,LIST_DATA_SET[n_data-1]);
    strcat(path_test,"_test_set");

 
    //
    printf("< Extract Test Set : %s >\n",path_train);
    
    train_sample = sample_constructor(&path_train, MAX_SEN_LEN);
    
    
    // make dict
    printf("< Make Dictionary >\n");
    dictionary dict;
    dictionary_constructor(&dict, train_sample, NUM_SAMPLE);

    null_ind = 0;

    dictionary_print(&dict);


    //
    max_line = 0;
    for(i=0;i<NUM_SAMPLE;i++) {
        if(train_sample[i].n_sen > max_line) {
            max_line = train_sample[i].n_sen;
        }
    }

    max_word = 0;
    for(i=0;i<NUM_SAMPLE;i++) {
        for(j=0;j<train_sample[i].n_sen;j++) {
            if(train_sample[i].sentences[j].n > max_word) {
                max_word = train_sample[i].sentences[j].n;
            }
        }
    }

    dim_dict = dict.n;
    if(f_enable_time) {
        dim_input = dim_dict + max_line;
        dim_word = max_word+1;              // +1 -> temporal encoding index
    } else {
        dim_input = dim_dict;
        dim_word = max_word;
    }

    test_sample = sample_constructor(path_test, max_line);
    
    for(i=0;i<NUM_SAMPLE;i++) {
        train_sample[i].dim_input = dim_input;
        train_sample[i].dim_word = dim_word;
        train_sample[i].max_word = max_word;
        train_sample[i].dim_dict = dim_dict;
    }

    for(i=0;i<NUM_SAMPLE_TEST;i++) {
        test_sample[i].dim_input = dim_input;
        test_sample[i].dim_word = dim_word;
        test_sample[i].max_word = max_word;
        test_sample[i].dim_dict = dim_dict;
    }

    sample_print(train_sample, NUM_SAMPLE, 0);
    sample_print(test_sample, NUM_SAMPLE_TEST, 0);

    printf("< Vetorization >\n");
    sample_vectorization(train_sample, &dict, NUM_SAMPLE, null_ind, f_enable_time);
    sample_vectorization(test_sample, &dict, NUM_SAMPLE_TEST, null_ind, f_enable_time);
    
    //sample_print(train_sample, NUM_SAMPLE, 0);
    //sample_print(test_sample, NUM_SAMPLE_TEST, 0);


    printf("< Layer Construction >\n");

    printf("    Number of Samples : %d\n",NUM_SAMPLE);
    printf("    Size of Batch : %d\n",size_batch);
    printf("    Number of Hop : %d\n",NUM_HOP);
    
    // embedding part
    // 
    dense_mat emb_m;        // m
    dense_mat emb_c;        // c
    dense emb_q;            // q
    dense_mat_constructor(&emb_m, max_line, dim_input, dim_emb, en_max_grad_l2_norm, max_grad_l2_norm);

    dense_mat_constructor(&emb_c, max_line, dim_input, dim_emb, en_max_grad_l2_norm, max_grad_l2_norm);
    dense_constructor(&emb_q, dim_input, dim_emb, en_max_grad_l2_norm, max_grad_l2_norm);
    
    // memory part
    dot_mat_vec dotmv;
    dot_mat_vec_constructor(&dotmv, max_line, dim_emb, dim_emb, false);

    // attention 
#ifdef TEST_MAXOUT
    unsigned int dim_maxout_in=1;
    unsigned int dim_maxout_hd=5;
    unsigned int dim_maxout_out=1;
    unsigned int ind_maxout;

    float attention[MAX_SEN_LEN];
    float attention_grad[MAX_SEN_LEN];
    float attention_total;

    maxout mxout;
    maxout_constructor(&mxout, dim_maxout_in, dim_maxout_hd, dim_maxout_out);
#else
    softmax sf_in;
    softmax_constructor(&sf_in, max_line, f_exp_plan);
#endif

    // o - weighted sum
    dot_mat_vec w_sum;
    dot_mat_vec_constructor(&w_sum, max_line, dim_emb, max_line, true);

    // o + u - sum vector
    sum_vec sv;
    sum_vec_constructor(&sv, dim_emb, dim_emb);
    
    // output part
    // dense
    dense ds_ans;
    dense_constructor(&ds_ans, dim_emb, dim_input, en_max_grad_l2_norm, max_grad_l2_norm);
    // softmax
    softmax sf_out;
    softmax_constructor(&sf_out, dim_input, false);
    // cost function - cross_entropy
    cross_entropy ce;
    cross_entropy_constructor(&ce, dim_input);

    
    for(ind_test_loop=0;ind_test_loop<num_test_loop;ind_test_loop++) {
        printf("TEST LOOP : %d\n",ind_test_loop);
        // init
        dense_mat_init(&emb_m);
        dense_mat_init(&emb_c);
        dense_init(&emb_q);
        
        dot_mat_vec_init(&dotmv);
#ifdef TEST_MAXOUT    
        maxout_init(&mxout);
        mxout.w_mat[0][0][0] = 0.083914;
        mxout.w_mat[0][1][0] = 5339.823730;
        mxout.w_mat[0][2][0] = 420.621429;
        mxout.w_mat[0][3][0] = 5.772651;
        mxout.w_mat[0][4][0] = 39.996742;

        mxout.b_mat[0][0] = 0.577625;
        mxout.b_mat[0][1] = -38234.390625;
        mxout.b_mat[0][2] = -1958.250000;
        mxout.b_mat[0][3] = -3.199552;
        mxout.b_mat[0][4] = -94.767494;
#else
        softmax_init(&sf_in);
#endif
    
        dot_mat_vec_init(&w_sum);
    
        sum_vec_init(&sv);
    
        dense_init(&ds_ans);
        
        softmax_init(&sf_out);
        cross_entropy_init(&ce);
        
        time_train_s = clock();

        lr = LEARNING_RATE;
        lambda = LAMBDA;
    
        for(ind_itr=0;ind_itr<NUM_ITR;ind_itr++) {
            if( (ind_itr % RATE_DECAY_STEP == 0)&&(ind_itr!=0) ) {
                lr = lr/2.0;       
            }
            
            if(f_msg == true) {
                printf("ITR : %d\n",ind_itr);
            }

            cost = 0.0;
            acc_train = 0.0;
            match_count=0;
    
            num_batch = ceil((float)NUM_SAMPLE/(float)size_batch);
            //printf("Number of Batch : %d\n",num_batch);
            
            for(ind_batch=0;ind_batch<num_batch;ind_batch++) {
                for(ind_sam=ind_batch*size_batch;(ind_sam<(ind_batch+1)*size_batch)&&(ind_sam<NUM_SAMPLE);ind_sam++) {
                    p_sam = &(train_sample[ind_sam]);
                    m = p_sam->sentences_b;
                    q = p_sam->question_b;
                    a = p_sam->answer_b;
                    ai = p_sam->answer_i.words[0];
                    
                    n_sen = p_sam->n_sen;
                    
                    // embedding c
                    c = p_sam->sentences_b;
    
                    if(ind_batch != num_batch-1) {
                        size_b = size_batch;
                    } else {
                        size_b = NUM_SAMPLE-ind_batch*size_batch;
                    }
    
                    // input
                    dense_mat_in(&emb_m, n_sen, dim_input, dim_emb, m);
                    dense_mat_in(&emb_c, n_sen, dim_input, dim_emb, m);
                    dense_in(&emb_q, dim_input, dim_emb, q);
        
                    dot_mat_vec_in(&dotmv, n_sen, dim_emb, dim_emb, emb_m.out_mat, emb_q.out_vec);
#ifdef TEST_MAXOUT
                    for(ind_maxout=0;ind_maxout<n_sen;ind_maxout++) {
                        attention[ind_maxout] = 0.0;
                        attention_grad[ind_maxout] = 0.0;
                    }
                    
                    dot_mat_vec_in(&w_sum, n_sen, dim_emb, n_sen, emb_c.out_mat, &attention);
#else
                    softmax_in(&sf_in, n_sen, dotmv.out_vec);
                    dot_mat_vec_in(&w_sum, n_sen, dim_emb, n_sen, emb_c.out_mat, sf_in.out_vec);
#endif
                    sum_vec_in(&sv, dim_emb, dim_emb, emb_q.out_vec, w_sum.out_vec);
    
                    dense_in(&ds_ans, dim_emb, dim_input, sv.out_vec);
                    softmax_in(&sf_out, dim_input, ds_ans.out_vec);
                    cross_entropy_in(&ce, dim_input, sf_out.out_vec, a);
                    
                    // weight tying
                    
                    ////////////////////////////////////////
                    // forward propagation
                    ////////////////////////////////////////
                    //printf("Forward Propagation\n");
                    dense_mat_fwd(&emb_m, verbose);
                    dense_mat_fwd(&emb_c, verbose);
                    dense_fwd(&emb_q, verbose);
                    
                    // hop 1
                    dot_mat_vec_fwd(&dotmv, verbose);
#ifdef TEST_MAXOUT
                    for(ind_maxout=0;ind_maxout<n_sen;ind_maxout++) {
                        maxout_in(&mxout, dim_maxout_in, dim_maxout_hd, dim_maxout_out, &dotmv.out_vec[ind_maxout]);
                        maxout_fwd(&mxout);
                        attention[ind_maxout] = mxout.out_vec[0];
                    }

                    attention_total = 0.0;
                    for(ind_maxout=0;ind_maxout<n_sen;ind_maxout++) {
                        attention_total += attention[ind_maxout];
                    }

                    for(ind_maxout=0;ind_maxout<n_sen;ind_maxout++) {
                        attention[ind_maxout] = attention[ind_maxout]/attention_total;
                    }
                    printf("attention\n");
                    for(ind_maxout=0;ind_maxout<n_sen;ind_maxout++) {
                        printf("%f ",attention[ind_maxout]);
                    }
                    printf("\n");
#else
                    softmax_fwd(&sf_in, false);
#endif
                    dot_mat_vec_fwd(&w_sum, verbose);
                    sum_vec_fwd(&sv, verbose);
                     
                    
                    for(i=1;i<NUM_HOP;i++) {
                        dot_mat_vec_in(&dotmv, n_sen, dim_emb, dim_emb, emb_m.out_mat, sv.out_vec);
        
                        dot_mat_vec_fwd(&dotmv, verbose);
#ifdef TEST_MAXOUT
                        for(ind_maxout=0;ind_maxout<n_sen;ind_maxout++) {
                            maxout_in(&mxout, dim_maxout_in, dim_maxout_hd, dim_maxout_out, &dotmv.out_vec[ind_maxout]);
                            maxout_fwd(&mxout);
                            attention[ind_maxout] = mxout.out_vec[0];
                        }

                        attention_total = 0.0;
                        for(ind_maxout=0;ind_maxout<n_sen;ind_maxout++) {
                            attention_total += attention[ind_maxout];
                        }

                        for(ind_maxout=0;ind_maxout<n_sen;ind_maxout++) {
                            attention[ind_maxout] = attention[ind_maxout]/attention_total;
                        }
#else
                        softmax_fwd(&sf_in, verbose);
#endif
                        dot_mat_vec_fwd(&w_sum, verbose);
                        sum_vec_fwd(&sv, verbose);
                    }
    
                    // 
                    dense_fwd(&ds_ans, verbose);
                    softmax_fwd(&sf_out, verbose);
        
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
                    cross_entropy_run(&ce);
                    cost += ce.cost;
    
                    ////////////////////////////////////////
                    // backward propagation
                    ////////////////////////////////////////
                    //printf("Backward Propagation\n");
                    dense_bwd(&ds_ans, ce.grad, false);
    
                    // tmp - duplicated layer
                    float *dup_grad = (float *) malloc(dim_emb*sizeof(float));
    
                    // hop 1
                    sum_vec_bwd(&sv, ds_ans.grad);
                    dot_mat_vec_bwd(&w_sum, sv.grad);
#ifdef TEST_MAXOUT
                    for(ind_maxout=0;ind_maxout<n_sen;ind_maxout++) {
                        maxout_bwd(&mxout, &w_sum.grad_vec[ind_maxout]);
                        attention_grad[ind_maxout] = mxout.grad[0];
                    }
                    
                    printf("attention_grad\n");
                    for(ind_maxout=0;ind_maxout<n_sen;ind_maxout++) {
                        printf("%f ",attention_grad[ind_maxout]);
                    }
                    printf("\n");

                    dot_mat_vec_bwd(&dotmv, &attention_grad);
#else
                    softmax_bwd(&sf_in, w_sum.grad_vec, verbose);

                    /*
                    printf("attention_grad\n");
                    for(i=0;i<n_sen;i++){
                        printf("%f ",sf_in.grad[i]);
                    }
                    */

                    dot_mat_vec_bwd(&dotmv, sf_in.grad);
#endif
                    
                    for(j=1;j<NUM_HOP;j++) {
                        for(i=0;i<dim_emb;i++) {
                            dup_grad[i] = (dotmv.grad_vec[i]+sv.grad[i]);
                        }
    
                        sum_vec_bwd(&sv, dup_grad);
                        dot_mat_vec_bwd(&w_sum, sv.grad);

#ifdef TEST_MAXOUT
                        for(ind_maxout=0;ind_maxout<n_sen;ind_maxout++) {
                            maxout_bwd(&mxout, &w_sum.grad_vec[ind_maxout]);
                            attention_grad[ind_maxout] = mxout.grad[0];
                        }
                        dot_mat_vec_bwd(&dotmv, &attention_grad);
#else
                        softmax_bwd(&sf_in, w_sum.grad_vec, verbose);
                        dot_mat_vec_bwd(&dotmv, sf_in.grad);
#endif
                    }
    
                    for(i=0;i<dim_emb;i++) {
                        dup_grad[i] = (dotmv.grad_vec[i]+sv.grad[i]);
                    }
    
                    dense_bwd(&emb_q, dup_grad, false);
                    dense_mat_bwd(&emb_c, w_sum.grad_mat, false);
                    dense_mat_bwd(&emb_m, dotmv.grad_mat, false);
                }
    
                // weight update
                //printf("Weight Update\n");
                dense_w_up(&ds_ans, lr, size_b, lambda, false);
                dense_w_up(&emb_q, lr, size_b, lambda, false);
                dense_mat_w_up(&emb_c, lr, size_b, lambda, false);
                dense_mat_w_up(&emb_m, lr, size_b, lambda, false);
                
                // null char weight -> 0.0
                for(i=0;i<dim_emb;i++) {
                    ds_ans.w_mat[null_ind][i] = 0.0;
                    //emb_q.w_mat[i][null_ind] = 0.0;
                    emb_c.w_mat[i][null_ind] = 0.0;
                    emb_m.w_mat[i][null_ind] = 0.0;
                }
            }
    
            acc_train = (float) match_count / (float) NUM_SAMPLE;
            
            if(f_msg==true) {
                printf("Cost(loss) : %f\n",cost);
                printf("Acc(train) : %f\n",acc_train);
            }
        }
    
        time_train_e = clock();
        
        time_train_arr[ind_test_loop] = (double)(time_train_e-time_train_s)/(double)CLOCKS_PER_SEC;
        acc_train_arr[ind_test_loop] = acc_train;
    
    
        // test
        if(f_msg==true) {
            printf("Test\n");
        }

        acc_test = 0.0;
        match_count=0;
    
        time_test_s = clock();
    
        for(ind_sam=0;ind_sam<NUM_SAMPLE_TEST;ind_sam++) {
            p_sam = &(test_sample[ind_sam]);
            m = p_sam->sentences_b;
            q = p_sam->question_b;
            a = p_sam->answer_b;
            ai = p_sam->answer_i.words[0];
            
            n_sen = p_sam->n_sen;
            
            // embedding c
            c = p_sam->sentences_b;
                
            /*
            printf("n sen : %d\n",n_sen);
            printf("dim_input: %d\n",dim_input);
            printf("dim_emb : %d\n",dim_emb);
            */
    

            // input
            dense_mat_in(&emb_m, n_sen, dim_input, dim_emb, m);
            dense_mat_in(&emb_c, n_sen, dim_input, dim_emb, m);
            dense_in(&emb_q, dim_input, dim_emb, q);

            dot_mat_vec_in(&dotmv, n_sen, dim_emb, dim_emb, emb_m.out_mat, emb_q.out_vec);
#ifdef TEST_MAXOUT
            for(ind_maxout=0;ind_maxout<n_sen;ind_maxout++) {
                attention[ind_maxout] = 0.0;
                attention_grad[ind_maxout] = 0.0;
            }
            
            dot_mat_vec_in(&w_sum, n_sen, dim_emb, n_sen, emb_c.out_mat, &attention);
#else
            softmax_in(&sf_in, n_sen, dotmv.out_vec);
            dot_mat_vec_in(&w_sum, n_sen, dim_emb, n_sen, emb_c.out_mat, sf_in.out_vec);
#endif
            sum_vec_in(&sv, dim_emb, dim_emb, emb_q.out_vec, w_sum.out_vec);
    
            dense_in(&ds_ans, dim_emb, dim_input, sv.out_vec);
            softmax_in(&sf_out, dim_input, ds_ans.out_vec);
            cross_entropy_in(&ce, dim_input, sf_out.out_vec, a);
            
            // weight tying
            
            ////////////////////////////////////////
            // forward propagation
            ////////////////////////////////////////
            //printf("Forward Propagation\n");
            dense_mat_fwd(&emb_m, verbose);
            dense_mat_fwd(&emb_c, verbose);
            dense_fwd(&emb_q, verbose);
            
            // hop 1
            dot_mat_vec_fwd(&dotmv, verbose);
#ifdef TEST_MAXOUT
            for(ind_maxout=0;ind_maxout<n_sen;ind_maxout++) {
                maxout_in(&mxout, dim_maxout_in, dim_maxout_hd, dim_maxout_out, &dotmv.out_vec[ind_maxout]);
                maxout_fwd(&mxout);
                attention[ind_maxout] = mxout.out_vec[0];
            }
#else
            softmax_fwd(&sf_in, verbose);
#endif
            dot_mat_vec_fwd(&w_sum, verbose);
            sum_vec_fwd(&sv, verbose);
             
            
            for(i=1;i<NUM_HOP;i++) {
                dot_mat_vec_in(&dotmv, n_sen, dim_emb, dim_emb, emb_m.out_mat, sv.out_vec);

                dot_mat_vec_fwd(&dotmv, verbose);
#ifdef TEST_MAXOUT
                for(ind_maxout=0;ind_maxout<n_sen;ind_maxout++) {
                    maxout_in(&mxout, dim_maxout_in, dim_maxout_hd, dim_maxout_out, &dotmv.out_vec[ind_maxout]);
                    maxout_fwd(&mxout);
                    attention[ind_maxout] = mxout.out_vec[0];
                }
#else
                softmax_fwd(&sf_in, verbose);
#endif
                dot_mat_vec_fwd(&w_sum, verbose);
                sum_vec_fwd(&sv, verbose);
            }
    
            // 
            dense_fwd(&ds_ans, verbose);
            softmax_fwd(&sf_out, verbose);
        
/*

            // input
            dense_mat_in(&emb_m, n_sen, dim_input, dim_emb, m);
            dense_mat_in(&emb_c, n_sen, dim_input, dim_emb, m);
            dense_in(&emb_q, dim_input, dim_emb, q);
        
            dot_mat_vec_in(&dotmv, n_sen, dim_emb, dim_emb, emb_m.out_mat, emb_q.out_vec);
#ifdef TEST_MAXOUT
                    for(ind_maxout=0;ind_maxout<n_sen;ind_maxout++) {
                        attention[ind_maxout] = 0.0;
                        attention_grad[ind_maxout] = 0.0;
                    }
                    
                    dot_mat_vec_in(&w_sum, n_sen, dim_emb, n_sen, emb_c.out_mat, &attention);
#else
                    softmax_in(&sf_in, n_sen, dotmv.out_vec);
                    dot_mat_vec_in(&w_sum, n_sen, dim_emb, n_sen, emb_c.out_mat, sf_in.out_vec);
#endif
            sum_vec_in(&sv, dim_emb, dim_emb, emb_q.out_vec, w_sum.out_vec);
            dense_in(&ds_ans, dim_emb, dim_input, sv.out_vec);
            softmax_in(&sf_out, dim_input, ds_ans.out_vec);
            cross_entropy_in(&ce, dim_input, sf_out.out_vec, a);
            
            // weight tying
    
            // forward propagation
            dense_mat_fwd(&emb_m, verbose);
            dense_mat_fwd(&emb_c, verbose);
            dense_fwd(&emb_q, verbose);
            
            dot_mat_vec_fwd(&dotmv, verbose);
            softmax_fwd(&sf_in, verbose);
            dot_mat_vec_fwd(&w_sum, verbose);
            sum_vec_fwd(&sv, verbose);
            dense_fwd(&ds_ans, verbose);
            softmax_fwd(&sf_out, verbose);
 

*/






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
            
            acc_test = (float) match_count / (float) NUM_SAMPLE_TEST;
    
            /*
            printf("predict i : %d, ai : %d\n",predict_i, ai);
            
            printf("Answer\n");
            for(j=0;j<dim_input;j++) {
                printf("%.1f ",p_sam->answer_b[j]);
            }
            printf("\n"); 
            */
        }
        
        time_test_e = clock();
    
        printf("Acc(test) : %f\n",acc_test);
    
        time_test_arr[ind_test_loop] = (double)(time_test_e-time_test_s)/(double)CLOCKS_PER_SEC;
        acc_test_arr[ind_test_loop] = acc_test;
    
    }

    //
    time_train_avg = 0.0;
    time_test_avg = 0.0;
    acc_train_avg = 0.0;
    acc_test_avg = 0.0;

    time_train_max = 0.0;
    time_test_max = 0.0;
    acc_train_max = 0.0;
    acc_test_max = 0.0;
    
    time_train_min = time_train_arr[0];
    time_test_min = time_test_arr[0];
    acc_train_min = acc_train_arr[0];
    acc_test_min = acc_test_arr[0];


    for(ind_test_loop=0;ind_test_loop<num_test_loop;ind_test_loop++) {
        time_train_avg += time_train_arr[ind_test_loop];
        time_test_avg += time_test_arr[ind_test_loop];
        acc_train_avg += acc_train_arr[ind_test_loop];
        acc_test_avg += acc_test_arr[ind_test_loop];

        if(time_train_max < time_train_arr[ind_test_loop]) {
            time_train_max = time_train_arr[ind_test_loop];
        }

        if(time_test_max < time_test_arr[ind_test_loop]) {
            time_test_max = time_test_arr[ind_test_loop];
        }

        if(acc_train_max < acc_train_arr[ind_test_loop]) {
            acc_train_max = acc_train_arr[ind_test_loop];
        }

        if(acc_test_max < acc_test_arr[ind_test_loop]) {
            acc_test_max = acc_test_arr[ind_test_loop];
        }


        if(time_train_min > time_train_arr[ind_test_loop]) {
            time_train_min = time_train_arr[ind_test_loop];
        }

        if(time_test_min > time_test_arr[ind_test_loop]) {
            time_test_min = time_test_arr[ind_test_loop];
        }

        if(acc_train_min > acc_train_arr[ind_test_loop]) {
            acc_train_min = acc_train_arr[ind_test_loop];
        }

        if(acc_test_min > acc_test_arr[ind_test_loop]) {
            acc_test_min = acc_test_arr[ind_test_loop];
        }


    }

    time_train_avg = time_train_avg/(float)num_test_loop;
    time_test_avg = time_test_avg/(float)num_test_loop;
    acc_train_avg = acc_train_avg/(float)num_test_loop;
    acc_test_avg = acc_test_avg/(float)num_test_loop;

    printf("< Result - Train >\n");
    printf(" -TIME\n");
    printf("   avg : %f\n",time_train_avg);
    printf("   max : %f\n",time_train_max);
    printf("   min : %f\n",time_train_min);
    printf(" -ACCURACY\n");
    printf("   avg : %f\n",acc_train_avg);
    printf("   max : %f\n",acc_train_max);
    printf("   min : %f\n",acc_train_min);
 
    printf("< Result - Test >\n");
    printf(" -TIME\n");
    printf("   avg : %f\n",time_test_avg);
    printf("   max : %f\n",time_test_max);
    printf("   min : %f\n",time_test_min);
    printf(" -ACCURACY\n");
    printf("   avg : %f\n",acc_test_avg);
    printf("   max : %f\n",acc_test_max);
    printf("   min : %f\n",acc_test_min);
 

    // free - dynamically alocated memory

    //
    /*
    for(i=0;i<NUM_SAMPLE;i++) {
        for(j=0;j<train_sample[i].n_sen;j++) {
            free(train_sample[i].sentences[j]);
        }
    }
    */

    //
    free(time_train_arr);
    free(time_test_arr);
    free(acc_train_arr);
    free(acc_test_arr);

    return 0;
}  


/*
    line *sentences;
    line question;
    line answer;

    line_i *sentences_i;
    line_i question_i;
    line_i answer_i;

    bin **sentences_b;
    bin *question_b;
    bin *answer_b;
} sample;

*/
