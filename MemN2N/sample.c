#include "sample.h"

// file_name
// max_len: # of memory
// num_sample
/*
sample* sample_constructor_pentree(char *file_name, unsigned int max_len, unsigned int *num_sample)
{
    FILE *fp;
    char *line_tmp = NULL;
    size_t len = 0;
    ssize_t read;

    unsigned int n_s;

    unsigned int ind_sa;
    unsigned int n_sen;
    unsigned int n_sen_ori;
    unsigned int i, j, k;
    unsigned int ind_word;
    char *line_p;
    char *p_char=NULL;
    word tmp_word;

    char tmp_line[1000];

    sample *sa;

    unsigned int num_tot_words;

    fp= fopen(file_name,"r");

    srand(time(NULL));

    if(fp_train == NULL) {
        printf("*E : sample_constructor_pentree : file open error : %s\n",file_name);
        exit(1);
    }

    //printf("number of samples : %d\n",n_s);

    sa = (sample *) malloc(sizeof(sample));

    // find total number of words
    // your code here
    // num_tot_words = ?
    //

    sa->n_sen = 1;
    sa->sentences = (line *) malloc(sizeof(line));
    (sa->sentences)->n = num_tot_words;
    (sa->question).n = 1;
    (sa->answer).n = 1;

    while((read = getline(&line_tmp, &len, fp)) != -1) {

        if((line_p=strchr(line_tmp,'\n'))!=NULL) *line_p='\0';
        //printf("%d: %d: %s\n",ind_sa,i,line_tmp);
        strcpy(tmp_line, line_tmp);
        p_char = strtok(tmp_line," ");
        //while(p_char != NULL && sa[ind_sa].sentences[i].n<=MAX_LINE_LEN) {
        while(p_char != NULL) {
            //printf("%d, %s\n",strlen(p_char),p_char);
            ind_word = sa[ind_sa].sentences[i].n;
            //printf("1 : ind_sa : %d : i : %d : ind_word : %d\n",ind_sa, i, ind_word);
            tmp_word = (char *) malloc((strlen(p_char)+1)*sizeof(char));
            strcpy(tmp_word, p_char);
            sa[0]].sentences[0].words[ind_word] = tmp_word;
            sa[ind_sa].sentences[i].n++;
            p_char = strtok(NULL," ");
        }
    }
    
    // question
    
    
    // answer

    }

    fclose(fp);

    return sa;
}
*/

sample* sample_constructor(char *file_name, unsigned int max_len, unsigned int *num_sample, unsigned int num_sample_define) {
    FILE *fp_train;
    char *line_tmp = NULL;
    size_t len = 0;
    ssize_t read;

    unsigned int n_s;

    unsigned int ind_sa;
    unsigned int n_sen;
    unsigned int n_sen_ori;
    unsigned int i, j, k;
    unsigned int ind_word;
    char *line_p;
    char *p_char=NULL;
    word tmp_word;

    char tmp_line[1000];

    sample *sa;


    fp_train = fopen(file_name,"r");

    srand(time(NULL));

    if(fp_train == NULL) {
        printf("*E : sample_constructor : file open error : %s\n",file_name);
        exit(1);
    }

    read = getline(&line_tmp, &len, fp_train);
    read = getline(&line_tmp, &len, fp_train);
    read = getline(&line_tmp, &len, fp_train);
    
    if(EN_NUM_SAMPLE_FROM_FILE) {
        n_s = atoi(line_tmp);
    } else {
        n_s = num_sample_define;
    }

    *num_sample = n_s;

    printf("number of samples : %d\n",n_s);


    sa = (sample *) malloc(n_s * sizeof(sample));
   

    unsigned int itr_num = 0;
    
    while((read = getline(&line_tmp, &len, fp_train)) != -1) {
        read = getline(&line_tmp, &len, fp_train);
        read = getline(&line_tmp, &len, fp_train);
        
        //ind_sa = atoi(line_tmp);

        ind_sa = itr_num;
    
        read = getline(&line_tmp, &len, fp_train);
        read = getline(&line_tmp, &len, fp_train);
    
        n_sen_ori = atoi(line_tmp);


        if(n_sen_ori > max_len) {
            n_sen = max_len;

        } else {
            n_sen = n_sen_ori;
        }
        
        sa[ind_sa].n_sen = n_sen;
        sa[ind_sa].sentences = (line *) malloc(n_sen*sizeof(line));
        
        if(n_sen_ori > max_len) {
            for(i=0;i<n_sen_ori-max_len;i++) {
                read = getline(&line_tmp, &len, fp_train);
            }
        }

        for(i=0;i<n_sen;i++) {
            sa[ind_sa].sentences[i].n = 0;
        }
        sa[ind_sa].question.n = 0;
        sa[ind_sa].answer.n = 0;

        //printf("sentences\n");

        for(i=0;i<n_sen;i++) {
            read = getline(&line_tmp, &len, fp_train);
            if((line_p=strchr(line_tmp,'\n'))!=NULL) *line_p='\0';
            //printf("%d: %d: %s\n",ind_sa,i,line_tmp);
            strcpy(tmp_line, line_tmp);
            p_char = strtok(tmp_line," ");
            //while(p_char != NULL && sa[ind_sa].sentences[i].n<=MAX_LINE_LEN) {
            while(p_char != NULL) {
                //printf("%d, %s\n",strlen(p_char),p_char);
                
                ind_word = sa[ind_sa].sentences[i].n;
                //printf("1 : ind_sa : %d : i : %d : ind_word : %d\n",ind_sa, i, ind_word);
                tmp_word = (char *) malloc((strlen(p_char)+1)*sizeof(char));
                strcpy(tmp_word, p_char);
                sa[ind_sa].sentences[i].words[ind_word] = tmp_word;
                sa[ind_sa].sentences[i].n++;
                p_char = strtok(NULL," ");
            }
        }

        //printf("question\n");
        // question
        read = getline(&line_tmp, &len, fp_train);
        read = getline(&line_tmp, &len, fp_train);
        if((line_p=strchr(line_tmp,'\n'))!=NULL) *line_p='\0';
        //printf("%s\n",line_tmp);
        strcpy(tmp_line, line_tmp);
        p_char = strtok(tmp_line," ");
        //while(p_char != NULL && sa[ind_sa].question.n<=MAX_LINE_LEN) {
        while(p_char != NULL) {
            //printf("%d, %s\n",strlen(p_char),p_char);
            
            ind_word = sa[ind_sa].question.n;
            tmp_word = (char *) malloc((strlen(p_char)+1)*sizeof(char));
            strcpy(tmp_word, p_char);
            sa[ind_sa].question.words[ind_word] = tmp_word;
            sa[ind_sa].question.n++;
            p_char = strtok(NULL," ");
        }

        //printf("answer\n");
        // answer
        read = getline(&line_tmp, &len, fp_train);
        read = getline(&line_tmp, &len, fp_train);
        if((line_p=strchr(line_tmp,'\n'))!=NULL) *line_p='\0';
        //printf("%s\n",line_tmp);
        strcpy(tmp_line, line_tmp);
        p_char = strtok(tmp_line," ");
        //while(p_char != NULL && sa[ind_sa].answer.n<=MAX_LINE_LEN) {
        while(p_char != NULL) {
            //printf("%d, %s\n",strlen(p_char),p_char);
            
            ind_word = sa[ind_sa].answer.n;
            tmp_word = (char *) malloc((strlen(p_char)+1)*sizeof(char));
            strcpy(tmp_word, p_char);
            sa[ind_sa].answer.words[ind_word] = tmp_word;
            sa[ind_sa].answer.n++;
            p_char = strtok(NULL," ");
        }


        //printf("%zu\n", read);
        //printf("%s",line_tmp);
            
        itr_num++;
        if(itr_num >= NUM_SAMPLE) {
            break;
        }
    }

    fclose(fp_train);

    return sa;
}

// mode 0 : info
// mode 1 : sentence
// mode 2 : sentence index
// mode 3 : sentence binary
int sample_print(sample *sa, unsigned int num_sample, unsigned int mode) {
    
    unsigned int i, j, k;

    if(mode == 0 ) {
        printf("< Sample Info. >\n");
        printf("    Number of Samples : %d\n",num_sample);
        printf("    Dim of Input : %d\n",sa[0].dim_input);
        printf("    Dim of Word : %d\n",sa[0].dim_word);
        //printf("    Max Word : %d\n",sa[0].max_word);
        printf("    Dim of Dict : %d\n",sa[0].dim_dict);
    } else if(mode == 0) {
        for(i=0;i<num_sample;i++) {
            printf("Sample %d\n",i);
            printf("Sentences\n");
            for(j=0;j<sa[i].n_sen;j++) {
                for(k=0;k<sa[i].sentences[j].n;k++) {
                    printf("%s ",sa[i].sentences[j].words[k]);
                }
                printf("\n");
            }
            printf("Question\n");
            
            for(j=0;j<sa[i].question.n;j++) {
                printf("%s ",sa[i].question.words[j]);
            }
            printf("\n");
    
            printf("Answer\n");
            for(j=0;j<sa[i].answer.n;j++) {
                printf("%s ",sa[i].answer.words[j]);
            }
            printf("\n");
        }
    } else if(mode == 2) {
        for(i=0;i<num_sample;i++) {
            printf("Sentences\n");
            for(j=0;j<sa[i].n_sen;j++) {
                for(k=0;k<sa[i].sentences_i[j].n;k++) {
                    printf("%d ",sa[i].sentences_i[j].words[k]);
                }
                printf("\n");
            }
            printf("Question\n");
            for(k=0;k<sa[i].question_i.n;k++) {
                printf("%d ",sa[i].question_i.words[k]);
            }
            printf("\n");
    
            printf("Answer\n");
            for(k=0;k<sa[i].answer_i.n;k++) {
                printf("%d ",sa[i].answer_i.words[k]);
            }
            printf("\n");
        }
    } else if(mode == 3) {
        for(i=0;i<num_sample;i++) {
            printf("Sentences\n");
            for(j=0;j<sa[i].n_sen;j++) {
                for(k=0;k<sa->dim_input;k++) {
                    printf("%.1f ",sa[i].sentences_b[j][k]);
                }
                printf("\n");
            }
    
            printf("Question\n");
            for(k=0;k<sa->dim_input;k++) {
                printf("%.1f ",sa[i].question_b[k]);
            }
            printf("\n");
    
            printf("Answer\n");
            for(k=0;k<sa->dim_input;k++) {
                printf("%.1f ",sa[i].answer_b[k]);
            }
            printf("\n");
        }
    } 

    return 0;
}

int sample_init(sample *sa, unsigned int num_sample, unsigned int null_ind, bool f_enable_time) {
    unsigned int i, j, k;
    
    for(i=0;i<num_sample;i++) {
        sa[i].sentences_i = (line_i *) malloc(sa[i].n_sen*sizeof(line_i));
        for(j=0;j<sa[i].n_sen;j++) {
            if(f_enable_time) {
                if(sa[i].sentences[j].n > sa->dim_word-1) {
                    sa[i].sentences_i[j].n = sa->dim_word;
                } else {
                    sa[i].sentences_i[j].n = sa[i].sentences[j].n+1;
                }
            } else {
                if(sa[i].sentences[j].n > sa->dim_word) {
                    sa[i].sentences_i[j].n = sa->dim_word;
                } else {
                    sa[i].sentences_i[j].n = sa[i].sentences[j].n;
                }
            }

            sa[i].sentences_i[j].words = (int *) malloc(sa[i].sentences_i[j].n*sizeof(int));

            for(k=0;k<sa[i].sentences_i[j].n;k++) {
                sa[i].sentences_i[j].words[k] = null_ind;
            }
        }
        
        //sa[i].question_i.n = sa->max_word;
        //sa[i].question_i.n = sa[i].question.n;
        
        if(f_enable_time) {
            if(sa[i].question.n > sa->dim_word-1) {
                sa[i].question_i.n = sa->dim_word-1;
            } else {
                sa[i].question_i.n = sa[i].question.n;
            }
        } else {
            if(sa[i].question.n > sa->dim_word) {
                sa[i].question_i.n = sa->dim_word;
            } else {
                sa[i].question_i.n = sa[i].question.n;
            }
        }

        sa[i].question_i.words = (int *) malloc(sa[i].question_i.n*sizeof(int));

        for(j=0;j<sa[i].question_i.n;j++) {
            sa[i].question_i.words[j] = null_ind;
        }

        //sa[i].answer_i.n = sa->max_word;
        //sa[i].answer_i.n = sa[i].answer.n;

        if(f_enable_time) {
            if(sa[i].answer.n > sa->dim_word-1) {
                sa[i].answer_i.n = sa->dim_word-1;
            } else {
                sa[i].answer_i.n = sa[i].answer.n;
            }
        } else {
            if(sa[i].answer.n > sa->dim_word) {
                sa[i].answer_i.n = sa->dim_word;
            } else {
                sa[i].answer_i.n = sa[i].answer.n;
            }
        }

        sa[i].answer_i.words = (int *) malloc(sa[i].answer_i.n*sizeof(int));

        for(j=0;j<sa[i].answer_i.n;j++) {
            sa[i].answer_i.words[j] = null_ind;
        }
    }
    return 0;
}

int sample_vectorization(sample *sa, dictionary *dict, unsigned int *arr_ind, unsigned int num_sample, unsigned int null_ind, bool f_enable_time, bool f_train, bool f_en_pe, float **pe_w, float random_noise_time) {
    unsigned int i, j, k;
    
    unsigned int n_words;
    
    // word to index
    unsigned int n_noise;
    unsigned int *arr_te;
    unsigned int i_rn;

    unsigned int ind_sample;

    bool f_random_noise_time = (f_train==true)&&(random_noise_time!=0.0);

    for(i=0;i<num_sample;i++) {
        ind_sample=arr_ind[i];

        if(f_random_noise_time) {
            n_noise = rand()%(int)(sa[ind_sample].n_sen*random_noise_time+1);

            arr_te = (unsigned int*) malloc((sa[ind_sample].n_sen+n_noise)*sizeof(unsigned int));
            
            /*
            for(i_rn=0;i_rn<(sa[ind_sample].n_sen+n_noise);i_rn++) {
                arr_te[i_rn] = i_rn;
            }
            */
            
            rand_perm(arr_te, (sa[ind_sample].n_sen+n_noise));

            //printf(" n_noise : %d\n",n_noise);

            for(i_rn=0;i_rn<(sa[ind_sample].n_sen+n_noise);i_rn++) {
                if(arr_te[i_rn] >= MAX_SEN_LEN) {
                    arr_te[i_rn] = MAX_SEN_LEN-1;
                }
                //printf("%d : %d\n",i_rn,arr_te[i_rn]);
            }

            qsort(arr_te, sa[ind_sample].n_sen+n_noise, sizeof(unsigned int), compare_function);
            
            /*
            printf(" after sort \n");
            for(i_rn=0;i_rn<(sa[ind_sample].n_sen+n_noise);i_rn++) {
                printf("%d : %d\n",i_rn,arr_te[i_rn]);
            }
            */

            //printf("test : %f : %d\n",sa[ind_sample].n_sen*random_noise_time, rand()%(int)(sa[ind_sample].n_sen*random_noise_time+1));
            //printf("test : %f : %d\n",sa[ind_sample].n_sen*random_noise_time, rand()%3.0);
            //printf("test : %f : %f : %d\n",sa[ind_sample].n_sen*random_noise_time, rand()%(int)ceil(0.1), (int)(-2.8));
        }

        for(j=0;j<sa[ind_sample].n_sen;j++) {
            if(f_enable_time) {
                for(k=0;k<sa[ind_sample].sentences_i[j].n-1;k++) {
                    sa[ind_sample].sentences_i[j].words[k] = word_idx(dict,sa[ind_sample].sentences[j].words[k]);
                }
                if(f_random_noise_time) {
                    sa[ind_sample].sentences_i[j].words[k] = sa->dim_dict + arr_te[sa[ind_sample].n_sen+n_noise-j-1];     // te
                } else {
                    sa[ind_sample].sentences_i[j].words[k] = sa->dim_dict + sa[ind_sample].n_sen - j - 1;     // te
                }
                //printf("%d\n",sa[ind_sample].sentences_i[j].words[k]);
            } else {
                for(k=0;k<sa[ind_sample].sentences_i[j].n;k++) {
                    sa[ind_sample].sentences_i[j].words[k] = word_idx(dict,sa[ind_sample].sentences[j].words[k]);
                }
            }
        }
        
        if(f_random_noise_time) {
            free(arr_te);
        }


        //
        for(j=0;j<sa[ind_sample].question_i.n;j++) {
            sa[ind_sample].question_i.words[j] = word_idx(dict,sa[ind_sample].question.words[j]);
        }

        for(j=0;j<sa[ind_sample].answer_i.n;j++) {
            sa[ind_sample].answer_i.words[j] = word_idx(dict,sa[ind_sample].answer.words[j]);
        }
    }

    // index to vector
    for(i=0;i<num_sample;i++) {
        ind_sample=arr_ind[i];

        sa[ind_sample].sentences_b = (bin **) malloc(sa[ind_sample].n_sen*sizeof(bin*));
        sa[ind_sample].sentences_b[0] = (bin *) malloc(sa[ind_sample].n_sen*sa->dim_input*sizeof(bin));
        for(j=1;j<sa[ind_sample].n_sen;j++) {
            sa[ind_sample].sentences_b[j] = sa[ind_sample].sentences_b[j-1] + sa->dim_input;
        }

        sa[ind_sample].question_b = (bin *) malloc(sa->dim_input*sizeof(bin));
        sa[ind_sample].answer_b = (bin *) malloc(sa->dim_input*sizeof(bin));

        for(j=0;j<sa[ind_sample].n_sen;j++) {
            for(k=0;k<sa->dim_input;k++) {
                sa[ind_sample].sentences_b[j][k] = 0.0;
            }
        }

        for(j=0;j<sa->dim_input;j++) {
            sa[ind_sample].question_b[j] = 0.0;
            sa[ind_sample].answer_b[j] = 0.0;
        }
    }

    for(i=0;i<num_sample;i++) {
        ind_sample=arr_ind[i];

        for(j=0;j<sa[ind_sample].n_sen;j++) {
            n_words = sa[ind_sample].sentences_i[j].n;
            if(f_en_pe) {
                if(f_enable_time) {
                    for(k=0;k<n_words-1;k++) {
                        //sa[ind_sample].sentences_b[j][sa[ind_sample].sentences_i[j].words[k]] = pe_w[sa[ind_sample].sentences_i[j].words[k]][k];
                        sa[ind_sample].sentences_b[j][sa[ind_sample].sentences_i[j].words[k]] += 1.0;
                    }
                    sa[ind_sample].sentences_b[j][sa[ind_sample].sentences_i[j].words[k]] = 1.0; // te
                } else {
                    for(k=0;k<n_words;k++) {
                        //sa[ind_sample].sentences_b[j][sa[ind_sample].sentences_i[j].words[k]] = pe_w[sa[ind_sample].sentences_i[j].words[k]][k];
                        sa[ind_sample].sentences_b[j][sa[ind_sample].sentences_i[j].words[k]] += 1.0;
                    }

                }
            } else {
                if(f_enable_time) {
                    for(k=0;k<n_words-1;k++) {
                        sa[ind_sample].sentences_b[j][sa[ind_sample].sentences_i[j].words[k]] += 1.0;
                    }
                    sa[ind_sample].sentences_b[j][sa[ind_sample].sentences_i[j].words[k]] = 1.0; // te
                } else {
                    for(k=0;k<n_words;k++) {
                        sa[ind_sample].sentences_b[j][sa[ind_sample].sentences_i[j].words[k]] += 1.0;
                    }
                }
            }
        }
        
        n_words = sa[ind_sample].question_i.n;
        for(j=0;j<n_words;j++) {
            if(f_en_pe) {
                sa[ind_sample].question_b[sa[ind_sample].question_i.words[j]] = pe_w[sa[ind_sample].question_i.words[j]][j];
            } else {
                sa[ind_sample].question_b[sa[ind_sample].question_i.words[j]] += 1.0;
            }
            
        }
        //sa[ind_sample].question_b[null_ind] = 0.0;

        
        n_words = sa[ind_sample].answer_i.n;
        for(j=0;j<n_words;j++) {
            sa[ind_sample].answer_b[sa[ind_sample].answer_i.words[j]] += 1.0;
        }
        //sa[ind_sample].answer_b[null_ind] = 0.0;
    }

    // index to binary
    if(EN_SAMPLE_BIN_OUT) {
        packet_16 tmp_pkt;
        unsigned short tmp_pkt_short;
        FILE *f_dump;
        f_dump = fopen("qa1_test.bin","wb");
    
        for(i=0;i<num_sample;i++) {
            ind_sample=arr_ind[i];
    
            sa[ind_sample].sentences_i_p16 = (packet_16 **) malloc(sa[ind_sample].n_sen*sizeof(packet_16*));
            for(j=0;j<sa[ind_sample].n_sen;j++) {
                sa[ind_sample].sentences_i_p16[j] = (packet_16*) malloc((sa[ind_sample].sentences_i[j].n)*sizeof(packet_16));
                if(f_enable_time) {
                    for(k=0;k<sa[ind_sample].sentences_i[j].n-1;k++) {
                        if(f_train) {
                            tmp_pkt.type = TYPE_TRAIN_SEN;   
                        } else {
                            tmp_pkt.type = TYPE_TEST_SEN;   
                        }
                        tmp_pkt.addr = sa[ind_sample].sentences_i[j].words[k];
        
                        sa[ind_sample].sentences_i_p16[j][k]=tmp_pkt;
                        //printf("%04X\n",tmp_pkt);
                        
                        
                        tmp_pkt_short = ENDIAN_CHANGE_16(TYPE_CAST_PKT16_SHORT(tmp_pkt));
                        fwrite(&tmp_pkt_short,sizeof(tmp_pkt),1,f_dump);
                    }
        
                    // TE
                    if(f_train) {
                        tmp_pkt.type = TYPE_TRAIN_SEN_DONE;
                    } else {
                        tmp_pkt.type = TYPE_TEST_SEN_DONE;
                    }
                    //tmp_pkt.addr = sa[ind_sample].sentences_i[j].words[sa->dim_word-1];
                    tmp_pkt.addr = sa[ind_sample].sentences_i[j].words[sa[ind_sample].sentences_i[j].n-1];
                    
                    sa[ind_sample].sentences_i_p16[j][k]=tmp_pkt;
                    //printf("%04X\n",tmp_pkt);
                    
                    tmp_pkt_short = ENDIAN_CHANGE_16(TYPE_CAST_PKT16_SHORT(tmp_pkt));
                    fwrite(&tmp_pkt_short,sizeof(tmp_pkt),1,f_dump);
    
                } else {
                    for(k=0;k<sa[ind_sample].sentences_i[j].n;k++) {
                        if(f_train) {
                            tmp_pkt.type = TYPE_TRAIN_SEN;   
                        } else {
                            tmp_pkt.type = TYPE_TEST_SEN;   
                        }
                        tmp_pkt.addr = sa[ind_sample].sentences_i[j].words[k];
        
                        sa[ind_sample].sentences_i_p16[j][k]=tmp_pkt;
                        //printf("%04X\n",tmp_pkt);
                        
                        tmp_pkt_short = ENDIAN_CHANGE_16(TYPE_CAST_PKT16_SHORT(tmp_pkt));
                        fwrite(&tmp_pkt_short,sizeof(tmp_pkt),1,f_dump);
                    }
                }
            }
            
            sa[ind_sample].question_i_p16 = (packet_16*) malloc(sa[ind_sample].question.n*sizeof(packet_16));
            for(j=0;j<sa[ind_sample].question.n;j++) {
                if(f_train) {
                    if(j==sa[ind_sample].question.n-1) {
                        tmp_pkt.type = TYPE_TRAIN_QUEST_DONE;
                    } else {
                        tmp_pkt.type = TYPE_TRAIN_QUEST;
                    }
                } else {
                    if(j==sa[ind_sample].question.n-1) {
                        tmp_pkt.type = TYPE_TEST_QUEST_DONE;
                    } else {
                        tmp_pkt.type = TYPE_TEST_QUEST;
                    }
                }
                tmp_pkt.addr = sa[ind_sample].question_i.words[j];
                sa[ind_sample].question_i_p16[j]=tmp_pkt;
                        
                tmp_pkt_short = ENDIAN_CHANGE_16(TYPE_CAST_PKT16_SHORT(tmp_pkt));
                fwrite(&tmp_pkt_short,sizeof(tmp_pkt),1,f_dump);
            }
    
            sa[ind_sample].answer_i_p16 = (packet_16*) malloc(sa[ind_sample].answer.n*sizeof(packet_16));
            for(j=0;j<sa[ind_sample].answer.n;j++) {
                if(f_train) {
                    if(j==sa[ind_sample].answer.n-1) {
                        tmp_pkt.type = TYPE_TRAIN_ANS_DONE;
                    } else {
                        tmp_pkt.type = TYPE_TRAIN_ANS;
                    }
                } else {
                    if(j==sa[ind_sample].answer.n-1) {
                        tmp_pkt.type = TYPE_TEST_ANS_DONE;
                    } else {
                        tmp_pkt.type = TYPE_TEST_ANS;
                    }
                }
    
                tmp_pkt.addr = sa[ind_sample].answer_i.words[j];
                sa[ind_sample].answer_i_p16[j]=tmp_pkt;
                
                tmp_pkt_short = ENDIAN_CHANGE_16(TYPE_CAST_PKT16_SHORT(tmp_pkt));
                fwrite(&tmp_pkt_short,sizeof(tmp_pkt),1,f_dump);
            }
    
        }
    
        fclose(f_dump);
    }

    return 0;
}

int sample_vectorization_destructor(sample *sa, unsigned int *arr_ind, unsigned int num_sample) {
    
    unsigned int i, j, k;
    unsigned int ind_sample;

    // index to vector
    for(i=0;i<num_sample;i++) {
        ind_sample=arr_ind[i];
        
        free(sa[ind_sample].sentences_b[0]);
        free(sa[ind_sample].sentences_b);  
        
        free(sa[ind_sample].question_b);
        free(sa[ind_sample].answer_b);
    }

    // index to binary
    if(EN_SAMPLE_BIN_OUT) {
        for(i=0;i<num_sample;i++) {
            ind_sample=arr_ind[i];
            for(j=0;j<sa[ind_sample].n_sen;j++) {
                free(sa[ind_sample].sentences_i_p16[j]);
            }
            free(sa[ind_sample].sentences_i_p16); 
            free(sa[ind_sample].question_i_p16);
            free(sa[ind_sample].answer_i_p16);
        }
    }

    return 0;
}

int sample_hex_dump(sample*sa, dictionary *dict, unsigned int num_sample, FILE *f_dump) {

    unsigned char tmp_dump[NUM_BYTE];

    // for bAbI task #1
    // status       2 bit   - sentence 00 / question 01 / answer 10
    // word dict    20 bit
    // TE           10 bit

    // for general
    // total        128 bit
    // status       2   bit
    // word dict    62  bit
    // TE           64  bit


    // sparse representation
    // packet           16 bit
    // word(2 packets)  32 bit
    
    // packet           16 bit
    //      type         4 bit
    //          sentence        0000
    //          sentence_done   0001
    //          question        0010
    //          question_done   0011
    //          answer          0100
    //          answer_done     0101
    //      addr        12 bit

    
    
    unsigned int i, j, k, l;

    unsigned int count_byte;
    unsigned int ind_byte;

    unsigned char tmp_byte;

    count_byte = 0;
    ind_byte = NUM_BYTE-1;
    tmp_byte = tmp_byte & 0x00;


    unsigned int tot_n_sen;
    
    tot_n_sen = 0;
    
    for(i=0;i<NUM_BYTE;i++) {
        tmp_dump[i] = tmp_dump[i] & 0x00;
    }
   
    packet_16 tmp_pkt;
    unsigned short tmp_pkt_short;


    for(i=0;i<num_sample;i++) {
        for(j=0;j<sa[i].n_sen;j++) {
            for(k=0;k<sa[i].sentences[j].n;k++) {
                tmp_pkt.type = TYPE_TRAIN_SEN;   
                tmp_pkt.addr = sa[i].sentences_i[j].words[k];
                tmp_pkt_short = ENDIAN_CHANGE_16(TYPE_CAST_PKT16_SHORT(tmp_pkt));

                fwrite(&tmp_pkt_short,sizeof(tmp_pkt),1,f_dump);
                //printf("%04X\n",tmp_pkt);
            }

            // TE
            tmp_pkt.type = TYPE_TRAIN_SEN_DONE;
            tmp_pkt.addr = sa[i].sentences_i[j].words[sa->dim_word-1];
            
            tmp_pkt_short = ENDIAN_CHANGE_16(TYPE_CAST_PKT16_SHORT(tmp_pkt));
            fwrite(&tmp_pkt_short,sizeof(tmp_pkt),1,f_dump);
            //printf("%04X\n",tmp_pkt);
        }

        for(j=0;j<sa[i].question.n;j++) {
            if(j==sa[i].question.n-1) {
                tmp_pkt.type = TYPE_TRAIN_QUEST_DONE;
            } else {
                tmp_pkt.type = TYPE_TRAIN_QUEST;
            }
            tmp_pkt.addr = sa[i].question_i.words[j];
            tmp_pkt_short = ENDIAN_CHANGE_16(TYPE_CAST_PKT16_SHORT(tmp_pkt));
            fwrite(&tmp_pkt_short,sizeof(tmp_pkt),1,f_dump);
        }

        for(j=0;j<sa[i].answer.n;j++) {
            if(j==sa[i].answer.n-1) {
                tmp_pkt.type = TYPE_TRAIN_ANS_DONE;
            } else {
                tmp_pkt.type = TYPE_TRAIN_ANS;
            }
            tmp_pkt.addr = sa[i].answer_i.words[j];
            tmp_pkt_short = ENDIAN_CHANGE_16(TYPE_CAST_PKT16_SHORT(tmp_pkt));
            fwrite(&tmp_pkt_short,sizeof(tmp_pkt),1,f_dump);
        }
    }

    return 0;
}




int sample_destructor() {

    return 0;
}





int word_idx(dictionary* dict, word* word_t) {
    unsigned int i;

    for(i=0;i<dict->n;i++) {
        if( strcasecmp(dict->words[i],word_t) == 0 ) {
            return i;
        }
    }
    printf("NO WORD IN DICT : %s\n",word_t);

    return -1;
}


int dictionary_constructor(dictionary *dict, sample *sa, unsigned int num_sample) {
    unsigned int i,j,k,l;
    char tmp_word[MAX_WORD_LEN];
    bool f_match = false;

    dict->n = 0;
    // index 0 - NULL
    dict->words[dict->n] = (char *) malloc((strlen(NULL_CHAR)+1)*sizeof(char));
    strcpy(dict->words[dict->n],NULL_CHAR);
    dict->n++;

    for(i=0;i<num_sample;i++) {
        //sentence
        for(j=0;j<sa[i].n_sen;j++) {
            for(k=0;k<sa[i].sentences[j].n;k++) {
                f_match = false;

                strcpy(tmp_word,sa[i].sentences[j].words[k]);

                for(l=0;l<dict->n;l++) {
                    if( strcasecmp( dict->words[l],tmp_word ) == 0 ) {
                        f_match = true;
                        break;
                    }
                }
                
                if(!f_match) {
                    dict->words[dict->n] = (char *) malloc((strlen(tmp_word)+1)*sizeof(char));
                    strcpy(dict->words[dict->n],tmp_word);
                    dict->n++;
                } 
            }
        }

        //question
        for(j=0;j<sa[i].question.n;j++) {
            f_match = false;

            strcpy(tmp_word,sa[i].question.words[j]);

            for(l=0;l<dict->n;l++) {
                //if( strcmp(dict->words[l],tmp_word) == 0 ) {
                if( strcasecmp(dict->words[l],tmp_word) == 0 ) {
                    f_match = true;
                    break;
                }
            }
            
            if(!f_match) {
                dict->words[dict->n] = (char *) malloc((strlen(tmp_word)+1)*sizeof(char));
                strcpy(dict->words[dict->n],tmp_word);
                dict->n++;
            } 
        }

        //answer
        for(j=0;j<sa[i].answer.n;j++) {
            f_match = false;

            strcpy(tmp_word,sa[i].answer.words[j]);

            for(l=0;l<dict->n;l++) {
                //if( strcmp(dict->words[l],tmp_word) == 0 ) {
                if( strcasecmp(dict->words[l],tmp_word) == 0 ) {
                    f_match = true;
                    break;
                }
            }
            
            if(!f_match) {
                dict->words[dict->n] = (char *) malloc((strlen(tmp_word)+1)*sizeof(char));
                strcpy(dict->words[dict->n],tmp_word);
                dict->n++;
            } 
        }

    }

    return 0;
}

int dictionary_print(dictionary *dict) {

    unsigned int i;
    printf("< Dictionary Info. >\n");
    // print dict
    printf("    Dictionary : %d words\n    ",dict->n);
    for(i=0;i<dict->n;i++) {
        printf("%s ",dict->words[i]);
    }
    printf("\n");


    return 0;
}

int dictionary_destructor(dictionary *dict) {
    unsigned int i;

    for(i=0;i<dict->n;i++) {
        free(dict->words[i]);
    }
    
    return 0;
}

