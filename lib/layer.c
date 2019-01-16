#include "layer.h"

////////////////////////////////////////////////////////////////////////////////
// dot - mat vec
////////////////////////////////////////////////////////////////////////////////
double dot_mat_vec_constructor
(
    dot_mat_vec *dot,
    unsigned int dim_mat_x_max,
    unsigned int dim_mat_y,
    unsigned int dim_vec,
    bool f_trans,
    bool f_fixed,
    unsigned int iwl_m,
    unsigned int frac_m,
    unsigned int iwl_v,
    unsigned int frac_v,
	unsigned int f_mode,
    unsigned int attention_mode,
	FILE *fp_result
)
{
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j;

    dot->dim_mat_x_max = dim_mat_x_max;
    dot->dim_mat_y = dim_mat_y;
    dot->dim_vec = dim_vec;
    dot->dim_mat_x = 0;

    dot->f_trans = f_trans;
    dot->f_fixed = f_fixed;
    dot->iwl_m = iwl_m;
    dot->frac_m = frac_m;
    dot->iwl_v = iwl_v;
    dot->frac_v = frac_v;
	dot->f_mode = f_mode;

    dot->attention_mode = attention_mode;

    if(f_trans) {
        dot->out_vec = (float *) malloc(dim_mat_y*sizeof(float));
        dot->grad_out_vec = (float *) malloc(dim_mat_x_max*sizeof(float));
    } else {
        dot->out_vec = (float *) malloc(dim_mat_x_max*sizeof(float));
        dot->grad_out_vec = (float *) malloc(dim_mat_y*sizeof(float));
    }

    dot->grad_out_mat = (float **) malloc(dim_mat_x_max*sizeof(float*));
    dot->grad_out_mat[0] = (float *) malloc(dim_mat_x_max*dim_mat_y*sizeof(float));
    for(i=1;i<dot->dim_mat_x_max;i++) {
        dot->grad_out_mat[i] = dot->grad_out_mat[i-1] + dim_mat_y;
    }

    if(en_gpu_model) {
        cuda_dot_mat_vec_constructor
        (
            &(dot->dev_out_vec),
            &(dot->dev_grad_out_vec),
            &(dot->dev_grad_out_mat),
            &(dot->dev_f_overflow),
            &(dot->dev_cliff_marker),
            dot->dim_mat_x_max,
            dot->dim_mat_y,
            dot->f_trans
        );
    }

    time_e = clock();

    printf(" < dot_mat_vec_constructor > - dim_x_max: %d, dim_y: %d, dim_vec: %d, f_fixed: %s, iwl_m: %d, frac_m: %d, iwl_v: %d, frac_v: %d, f_mode: %d, attention_mode: %d\n",dim_mat_x_max,dim_mat_y,dim_vec,BOOL_PRINTF(f_fixed),iwl_m,frac_m,iwl_v,frac_v,f_mode,attention_mode);
    fprintf(fp_result," < dot_mat_vec_constructor > - dim_x_max: %d, dim_y: %d, dim_vec: %d, f_fixed: %s, iwl_m: %d, frac_m: %d, iwl_v: %d, frac_v: %d, f_mode: %d, attention_mode: %d\n",dim_mat_x_max,dim_mat_y,dim_vec,BOOL_PRINTF(f_fixed),iwl_m,frac_m,iwl_v,frac_v,f_mode,attention_mode);

    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double dot_mat_vec_init(dot_mat_vec *dot) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j;

    if(dot->f_trans) {
        for(i=0;i<dot->dim_mat_y;i++) {
            dot->out_vec[i] = 0.0;
        }

        for(i=0;i<dot->dim_mat_x_max;i++) {
            dot->grad_out_vec[i] = 0.0;
        }
    } else {
        for(i=0;i<dot->dim_mat_x_max;i++) {
            dot->out_vec[i] = 0.0;
        }

        for(i=0;i<dot->dim_mat_y;i++) {
            dot->grad_out_vec[i] = 0.0;
        }
    }

    for(i=0;i<dot->dim_mat_x_max;i++) {
        for(j=0;j<dot->dim_mat_y;j++) {
            dot->grad_out_mat[i][j] = 0.0;
        }
    }

    if(en_gpu_model) {
        cuda_dot_mat_vec_init
        (
            dot->dev_out_vec,
            dot->dev_grad_out_vec,
            dot->dev_grad_out_mat,
            dot->dev_f_overflow,
            dot->dev_cliff_marker,
            dot->dim_mat_x_max,
            dot->dim_mat_y,
            dot->f_trans
        );
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double dot_mat_vec_in
(
    dot_mat_vec *dot,
    unsigned int dim_mat_x,
    float **in_mat,
    float *in_vec,
    float *grad_in,
    float *dev_in_mat,
    float *dev_in_vec,
    float *dev_grad_in
)
{
    clock_t time_s, time_e;
    time_s = clock();

    if( (dot->dim_mat_x_max < dim_mat_x)
      ) {
        printf("*E : dot_mat_vec_in : exceed max dim\n");
        exit(1);
        return -1;
    }

    dot->dim_mat_x = dim_mat_x;

    dot->in_mat = in_mat;
    dot->in_vec = in_vec;

    dot->grad_in = grad_in;

    if(en_gpu_model) {
        dot->dev_in_mat = dev_in_mat;
        dot->dev_in_vec = dev_in_vec;

        dot->dev_grad_in = dev_grad_in;
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double dot_mat_vec_fwd(dot_mat_vec *dot, bool verbose) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j;

    unsigned int d_m_x = dot->dim_mat_x;
    unsigned int d_m_y = dot->dim_mat_y;

    if(en_gpu_model) {
        if(dot->attention_mode==1) {
            cuda_dot_mat_vec_fwd
            (
                dot->dev_in_mat,
                dot->dev_in_vec,
                dot->dev_out_vec,
                dot->dev_f_overflow,
                dot->dim_mat_x,
                dot->dim_mat_y,
                dot->f_trans,
                //dot->f_fixed,
                false,
                dot->iwl_m,
                dot->frac_m,
                dot->iwl_v,
                dot->frac_v,
				dot->f_mode,
                verbose
            );
        } else if(dot->attention_mode==2) {
            cuda_dot_mat_vec_fwd
            (
                dot->dev_in_mat,
                dot->dev_in_vec,
                dot->dev_out_vec,
                dot->dev_f_overflow,
                dot->dim_mat_x,
                dot->dim_mat_y,
                dot->f_trans,
                //dot->f_fixed,
                true,
                dot->iwl_m,
                dot->frac_m,
				dot->iwl_v,
				dot->frac_v,
				dot->f_mode,
                verbose
            );
        } else if(dot->attention_mode==3) {
            cuda_dot_mat_vec_fwd_appx
            (
                dot->dev_in_mat,
                dot->dev_in_vec,
                dot->dev_out_vec,
                dot->dev_f_overflow,
                dot->dev_cliff_marker,
                dot->dim_mat_x,
                dot->dim_mat_y,
                dot->f_fixed,
                dot->iwl_m,
                dot->frac_m,
				dot->f_mode,
                //NUM_BIT_ATTENTION,
                (1+dot->iwl_m+dot->frac_m),
                dot->f_trans,
                verbose
            );
        } else if(dot->attention_mode==4) {
            printf(" not implemented binary att mode yet. use attention mode 2 \n");
            /*
            if(!(dot->f_trans)) {
                cuda_binarization
                (
                    dot->dev_in_mat,
                    dot->dim_mat_x*dot->dim_mat_y
                );

                cuda_binarization
                (
                    dot->dev_in_vec,
                    dot->dim_mat_y
                );
            }
            */
        }
    }

    if(en_cpu) {
		printf("NOT YET FIX CPU MODE\n");
		/*
        if(dot->f_trans) {
            for(i=0;i<d_m_y;i++) {
                dot->out_vec[i] = 0.0;
            }

            if(dot->f_fixed) {
                for(i=0;i<d_m_x;i++) {
                    for(j=0;j<d_m_y;j++) {
                        //dot->out_vec[j] += mult_fixed_32(dot->in_mat[i][j],dot->in_vec[i]);
                        FIXED_MAC(dot->out_vec[j],dot->in_mat[i][j],dot->in_vec[i],dot->iwl,dot->frac);
                    }
                }
            } else {
                for(i=0;i<d_m_x;i++) {
                    for(j=0;j<d_m_y;j++) {
                        dot->out_vec[j] += dot->in_mat[i][j]*dot->in_vec[i];
                    }
                }
            }
        } else {
            if(dot->attention_mode==1) {
                for(i=0;i<d_m_x;i++) {
                    dot->out_vec[i] = 0.0;
                }

                if(dot->f_fixed) {
                    for(i=0;i<d_m_x;i++) {
                        for(j=0;j<d_m_y;j++) {
                            //dot->out_vec[i] += FLOAT_QUANT(dot->in_mat[i][j])*FLOAT_QUANT(dot->in_vec[j]);
                            FIXED_MAC(dot->out_vec[i],dot->in_mat[i][j],dot->in_vec[j],dot->iwl,dot->frac);
                        }
                        FLOAT_QUANT(dot->out_vec[i],dot->iwl,dot->frac);
                    }
                } else {
                    for(i=0;i<d_m_x;i++) {
                        for(j=0;j<d_m_y;j++) {
                            dot->out_vec[i] += dot->in_mat[i][j]*dot->in_vec[j];
                        }
                    }
                }
            } else if(dot->attention_mode==2) {
                for(i=0;i<d_m_x;i++) {
                    dot->out_vec[i] = 0.0;
                }

                for(i=0;i<d_m_x;i++) {
                    for(j=0;j<d_m_y;j++) {

                        //dot->out_vec[i] += mult_fixed_32(dot->in_mat[i][j],dot->in_vec[j]);
                        //dot->out_vec[i] += dot->in_mat[i][j]*dot->in_vec[j];

                        //printf("float a : %f, fixed a : %f, err : %f\n",float_a_32,FLOAT_QUANT(float_a_32),float_a_32-FLOAT_QUANT(float_a_32));
                        //printf("float b : %f, fixed b : %f, err : %f\n",float_b_32,FLOAT_QUANT(float_b_32),float_b_32-FLOAT_QUANT(float_b_32));

                        //printf("%f : %f -> %f : %f\n",dot->in_mat[i]

                        // last

                        dot->in_mat[i][j] = FLOAT_QUANT(dot->in_mat[i][j],dot->iwl,dot->frac);
                        dot->in_vec[j] = FLOAT_QUANT(dot->in_vec[j],dot->iwl,dot->frac);

                        dot->out_vec[i] += FLOAT_QUANT(dot->in_mat[i][j]*dot->in_vec[j],dot->iwl,dot->frac);
                        dot->out_vec[i] = FLOAT_QUANT(dot->out_vec[i],dot->iwl,dot->frac);
                    }
                }
            } else if(dot->attention_mode==3) {
                for(i=0;i<d_m_x;i++) {
                    dot->out_vec[i] = 0.0;
                }

                float tmp_a;
                float tmp_b;

                for(i=0;i<d_m_x;i++) {
                    for(j=0;j<d_m_y;j++) {
                        //dot->out_vec[i] += hamming_similarity(FLOAT2FIXED(dot->in_mat[i][j]),FLOAT2FIXED(dot->in_vec[j]),2);
                        //dot->out_vec[i] += (float)hamming_similarity_w(FLOAT2FIXED(dot->in_mat[i][j]),FLOAT2FIXED(dot->in_vec[j]),NUM_BIT_ATTENTION);
                        //dot->out_vec[i] += (float)hamming_similarity_w(FLOAT2FIXED(dot->in_mat[i][j]),FLOAT2FIXED(dot->in_vec[j]),NUM_BIT_ATTENTION)/pow(2,NUM_BIT_ATTENTION+1)/d_m_y;
                        //dot->out_vec[i] += hamming_similarity_w(FLOAT2FIXED(dot->in_mat[i][j]),FLOAT2FIXED(dot->in_vec[j]),NUM_BIT_ATTENTION)/pow(2,NUM_BIT_ATTENTION+1);

                        // proposed method
                        // current version
                        dot->out_vec[i] += hamming_similarity_w(FLOAT2FIXED(dot->in_mat[i][j],dot->iwl,dot->frac),FLOAT2FIXED(dot->in_vec[j],dot->iwl,dot->frac),NUM_BIT_ATTENTION,false);

                        //dot->out_vec[i] += (float)hamming_similarity_w(FLOAT2FIXED(dot->in_mat[i][j]),FLOAT2FIXED(dot->in_vec[j]),NUM_BIT_ATTENTION)/d_m_y;
                        //printf("%f\n",(float)hamming_similarity_w(FLOAT2FIXED(dot->in_mat[i][j]),FLOAT2FIXED(dot->in_vec[j]),NUM_BIT_ATTENTION)/pow(2,NUM_BIT_ATTENTION+1));
                        //dot->out_vec[i] += tmp_a*tmp_b;


                        //printf("[%2d][%2d] : %f : %f : -> : %08x : %08x : -> %f\n",i,j,dot->in_mat[i][j],dot->in_vec[j],FLOAT2FIXED(dot->in_mat[i][j]),FLOAT2FIXED(dot->in_vec[j]),(float)hamming_similarity_w(FLOAT2FIXED(dot->in_mat[i][j]),FLOAT2FIXED(dot->in_vec[j]),NUM_BIT_ATTENTION,false));
                    }

                    //printf("out_vec[%2d]%f\n",i,dot->out_vec[i]);
                    //dot->out_vec[i] = dot->out_vec[i]/pow(2,(int)(NUM_BIT_ATTENTION));

                    // quantization
                    //dot->out_vec[i] = floor(dot->out_vec[i]);
                    //dot->out_vec[i] = ceil(dot->out_vec[i]);
                    //printf("out_vec[%2d]%f\n",i,dot->out_vec[i]);
                }
            } else if(dot->attention_mode==4) {
                for(i=0;i<d_m_x;i++) {
                    dot->out_vec[i] = 0.0;
                }

                if(dot->f_fixed) {
                    for(i=0;i<d_m_x;i++) {
                        for(j=0;j<d_m_y;j++) {
                            //dot->out_vec[i] += mult_fixed_32(dot->in_mat[i][j],dot->in_vec[j]);
                            FIXED_MAC(dot->out_vec[i],dot->in_mat[i][j],dot->in_vec[j],dot->iwl,dot->frac);
                        }
                    }
                } else {
                    float tmp_a;
                    float tmp_b;

                    for(i=0;i<d_m_x;i++) {
                        for(j=0;j<d_m_y;j++) {
                            if(dot->in_mat[i][j] >= 0) {
                                tmp_a = 1.0;
                            } else {
                                tmp_a = -1.0;
                            }

                            if(dot->in_vec[j] >= 0) {
                                tmp_b = 1.0;
                            } else {
                                tmp_b = -1.0;
                            }
                            //dot->out_vec[i] += dot->in_mat[i][j]*dot->in_vec[j];
                            dot->out_vec[i] += tmp_a*tmp_b;
                        }
                    }
                }
            }
        }
        if(verbose) {
            printf("dot mat vec fwd\n");
            printf("dim mat x : %d\n", dot->dim_mat_x);
            printf("dim mat y : %d\n", dot->dim_mat_y);
            printf("dim vec   : %d\n", dot->dim_vec);
            if(dot->f_trans) {
                printf("dim out   : %d\n", dot->dim_mat_y);
            } else {
                printf("dim out   : %d\n", dot->dim_mat_x);
            }
    
            printf("in_mat\n");
            for(i=0;i<d_m_x;i++) {
                for(j=0;j<d_m_y;j++) {
                    printf("%.2f ",dot->in_mat[i][j]);
                }
                printf("\n");
            }
            printf("in_vec\n");
            for(j=0;j<dot->dim_vec;j++) {
                printf("%.2f ",dot->in_vec[j]);
            }
            printf("\n");
    
            if(dot->attention_mode==2) {
                printf("in_vec_quant\n");
                for(j=0;j<dot->dim_vec;j++) {
                    printf("%.2f ",FLOAT_QUANT(dot->in_vec[j],dot->iwl,dot->frac));
                }
                printf("\n");
            }
            printf("out_vec\n");
            if(dot->f_trans) {
                for(i=0;i<d_m_y;i++) {
                    printf("%.2f ",dot->out_vec[i]);
                }
            } else {
                for(i=0;i<d_m_x;i++) {
                    printf("%.2f ",dot->out_vec[i]);
                }
            }
            printf("\n");
        }
		*/
    }

    time_e = clock();


// verification_point_start
//    if(dot->f_trans) {
//        float *veri_dev_out_vec;
//
//        veri_dev_out_vec = (float*) malloc(d_m_y*sizeof(float));
//
//        cuda_memcpy_dev_to_host
//        (
//            veri_dev_out_vec,
//            dot->dev_out_vec,
//            d_m_y
//        );
//
//        /*
//        printf("dot_mat_vec_fwd: out_vec\n");
//        for(i=0;i<d_m_y;i++) {
//            printf("%f, ",dot->out_vec[i]);
//        }
//        printf("\n");
//        */
//
//        for(i=0;i<d_m_y;i++) {
//            if((dot->out_vec[i]-veri_dev_out_vec[i]) > TH_ERROR_FLOAT) {
//                printf("dot_mat_vec_fwd:trans true: out_vec\n");
//                printf("error %d : CPU : %f : GPU : %f\n",i,dot->out_vec[i],veri_dev_out_vec[i]);
//            }
//        }
//    } else {
//        float *veri_dev_in_vec;
//        float *veri_dev_in_mat;
//        float *veri_dev_out_vec;
//
//        veri_dev_in_vec = (float*) malloc(d_m_y*sizeof(float));
//        veri_dev_in_mat = (float*) malloc(d_m_x*d_m_y*sizeof(float));
//        veri_dev_out_vec = (float*) malloc(d_m_x*sizeof(float));
//
//        cuda_memcpy_dev_to_host
//        (
//            veri_dev_in_vec,
//            dot->dev_in_vec,
//            d_m_y
//        );
//
//        cuda_memcpy_dev_to_host
//        (
//            veri_dev_in_mat,
//            dot->dev_in_mat,
//            d_m_x*d_m_y
//        );
//
//        cuda_memcpy_dev_to_host
//        (
//            veri_dev_out_vec,
//            dot->dev_out_vec,
//            d_m_x
//        );
//
//
//        for(i=0;i<d_m_y;i++) {
//            if((dot->in_vec[i]-veri_dev_in_vec[i]) > TH_ERROR_FLOAT) {
//                printf("dot_mat_vec_fwd:trans false: in_vec\n");
//                printf("error %d : CPU : %f : GPU : %f\n",i,dot->in_vec[i],veri_dev_in_vec[i]);
//            }
//        }
//
//        for(i=0;i<d_m_x;i++) {
//            for(j=0;j<d_m_y;j++) {
//                if((dot->in_mat[i][j]-veri_dev_in_mat[i*d_m_y+j]) > TH_ERROR_FLOAT) {
//                    printf("dot_mat_vec_fwd:trans false: in_mat\n");
//                    printf("error %d %d: CPU : %f->%f : GPU : %f\n",i,j,dot->in_mat[i][j],FLOAT_QUANT(dot->in_mat[i][j]),veri_dev_in_mat[i*d_m_y+j]);
//                }
//            }
//        }
//
//        for(i=0;i<d_m_x;i++) {
//            if((dot->out_vec[i]-veri_dev_out_vec[i]) > TH_ERROR_FLOAT) {
//                printf("dot_mat_vec_fwd:trans false: out_vec\n");
//                printf("error %d : CPU : %f : GPU : %f\n",i,dot->out_vec[i],veri_dev_out_vec[i]);
//            }
//        }
//
//
//
//    }
//
// verification_point_end

    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double dot_mat_vec_bwd(dot_mat_vec *dot, unsigned int hop, bool verbose) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j;

    if(en_gpu_model) {
        if(dot->attention_mode==1) {
            cuda_dot_mat_vec_bwd
            (
                dot->dev_in_mat,
                dot->dev_in_vec,
                dot->dev_grad_in,
                dot->dev_grad_out_mat,
                dot->dev_grad_out_vec,
                dot->dev_f_overflow,
                dot->dim_mat_x,
                dot->dim_mat_y,
                dot->f_trans,
#ifdef EN_GRAD_QUANT
                dot->f_fixed,
#else
                false,
#endif
                dot->iwl_m,
                dot->frac_m,
                dot->iwl_v,
                dot->frac_v,
				dot->f_mode,
                verbose
            );
        } else if(dot->attention_mode==2) {
            cuda_dot_mat_vec_bwd
            (
                dot->dev_in_mat,
                dot->dev_in_vec,
                dot->dev_grad_in,
                dot->dev_grad_out_mat,
                dot->dev_grad_out_vec,
                dot->dev_f_overflow,
                dot->dim_mat_x,
                dot->dim_mat_y,
                dot->f_trans,
#ifdef EN_GRAD_QUANT
                dot->f_fixed,
#else
                false,
#endif
                dot->iwl_m,
                dot->frac_m,
                dot->iwl_v,
                dot->frac_v,
				dot->f_mode,
                verbose
            );
        } else if(dot->attention_mode==3) {
            cuda_dot_mat_vec_bwd_appx
            (
                dot->dev_in_mat,
                dot->dev_in_vec,
                dot->dev_grad_in,
                dot->dev_grad_out_mat,
                dot->dev_grad_out_vec,
                dot->dev_f_overflow,
                dot->dev_cliff_marker,
                dot->dim_mat_x,
                dot->dim_mat_y,
                dot->f_fixed,
                //false,
                dot->iwl_m,
                dot->frac_m,
				dot->f_mode,
                //NUM_BIT_ATTENTION,
                (1+dot->iwl_m+dot->frac_m),
                dot->f_trans,
                verbose,
                hop
            );
        } else if(dot->attention_mode==4) {
            printf(" not implemented binary att mode yet. use attention mode 2 \n");
        }
    }

    if(en_cpu) {
		printf("NOT YET FIX CPU MODE\n");
		/*
        if(dot->attention_mode==1) {
            for(i=0;i<dot->dim_mat_x;i++) {
                for(j=0;j<dot->dim_mat_y;j++) {
                    if(dot->f_trans) {
                        dot->grad_out_mat[i][j] = dot->in_vec[i]*dot->grad_in[j];
                    } else {
                        dot->grad_out_mat[i][j] = dot->in_vec[j]*dot->grad_in[i];
                    }
                }
            }

            for(i=0;i<dot->dim_vec;i++) {
                dot->grad_out_vec[i] = 0.0;
            }

            for(i=0;i<dot->dim_mat_x;i++) {
                for(j=0;j<dot->dim_mat_y;j++) {
                    if(dot->f_trans) {
                        dot->grad_out_vec[i] += dot->in_mat[i][j]*dot->grad_in[j];
                    } else {
                        dot->grad_out_vec[j] += dot->in_mat[i][j]*dot->grad_in[i];
                    }
                }
            }
        } else if(dot->attention_mode==2) {
            for(i=0;i<dot->dim_mat_x;i++) {
                for(j=0;j<dot->dim_mat_y;j++) {
                    if(dot->f_trans) {
                        dot->grad_out_mat[i][j] = dot->in_vec[i]*dot->grad_in[j];
                    } else {
                        //dot->grad_out_mat[i][j] = mult_fixed_32(dot->in_vec[j],dot->grad_in[i]);
                        //dot->grad_out_mat[i][j] = FIXED2FLOAT(FLOAT2FIXED(FIXED2FLOAT(FLOAT2FIXED(dot->in_vec[j]))*dot->grad_in[i]));
                        //dot->grad_out_mat[i][j] = FLOAT_QUANT(FLOAT_QUANT(dot->in_vec[j])*dot->grad_in[i]);

                        //dot->grad_out_mat[i][j] = dot->in_vec[j]*dot->grad_in[i];

                        // last
                        //dot->grad_out_mat[i][j] = FLOAT_QUANT(dot->in_vec[j])*dot->grad_in[i];
                        dot->grad_out_mat[i][j] = dot->in_vec[j]*dot->grad_in[i];
                        //printf("%f : %f\n",dot->in_vec[j]*dot->grad_in[i],dot->grad_out_mat[i][j]);
                    }
                }
            }

            for(i=0;i<dot->dim_vec;i++) {
                dot->grad_out_vec[i] = 0.0;
            }

            for(i=0;i<dot->dim_mat_x;i++) {
                for(j=0;j<dot->dim_mat_y;j++) {
                    if(dot->f_trans) {
                        dot->grad_out_vec[i] += dot->in_mat[i][j]*dot->grad_in[j];
                    } else {
                        //dot->grad_out_vec[j] += mult_fixed_32(dot->in_mat[i][j],dot->grad_in[i]);

                        // last
                        //dot->grad_out_vec[j] = FLOAT_QUANT(dot->in_mat[i][j])*dot->grad_in[i];
                        dot->grad_out_vec[j] = dot->in_mat[i][j]*dot->grad_in[i];
                    }
                }
            }
        } else if(dot->attention_mode==3) {
            float tmp_a;
            float tmp_b;

            int fixed_in_vec;
            int fixed_in_mat;

            int fixed_in_vec_bit;
            int fixed_in_mat_bit;

            unsigned int ind_hamming;
            int ind_hamming_bit;

            float hamming_weight;

            float sign;

            float sign_mat;
            float sign_vec;

            float sign_deriv;

            for(i=0;i<dot->dim_mat_x;i++) {
                for(j=0;j<dot->dim_mat_y;j++) {
                    if(dot->f_trans) {
                        dot->grad_out_mat[i][j] = dot->in_vec[i]*dot->grad_in[j];
                    } else {
                        // proposed method
                        // hamming distance w
                        tmp_a=0;

                        if( (FLOAT2FIXED(dot->in_vec[j],dot->iwl,dot->frac)&(0x80000000)) == 0x00000000 ) {
                            sign = 1.0;
                        } else {
                            sign = -1.0;
                        };



                        //printf(" %d : %f\n",sign,dot->in_vec[j]);
                        //printf(" %08x\n",FLOAT2FIXED(dot->in_vec[j])&(0x80000000));

                        fixed_in_vec = FLOAT2FIXED(dot->in_vec[j],dot->iwl,dot->frac);
                        fixed_in_mat = FLOAT2FIXED(dot->in_mat[i][j],dot->iwl,dot->frac);

                        if(fixed_in_mat&(0x80000000) == 0x00000000) {
                            sign_mat = 1.0;
                        } else {
                            sign_vec = -1.0;
                        }

                        if(fixed_in_vec&(0x80000000) == 0x00000000) {
                            sign_mat = 1.0;
                        } else {
                            sign_vec = -1.0;
                        }

                        for(ind_hamming=0;ind_hamming<NUM_BIT_ATTENTION;ind_hamming++) {
                            fixed_in_vec_bit = (unsigned int)(fixed_in_vec&(0x80000000>>ind_hamming))>>(31-ind_hamming);
                            fixed_in_mat_bit = (unsigned int)(fixed_in_mat&(0x80000000>>ind_hamming))>>(31-ind_hamming);

                            //printf("fixed in vec : %08x, fixed in vec bit[%d] : fixed_in_vec_bit : %08x : %08x\n", fixed_in_vec, ind_hamming, fixed_in_vec_bit, (unsigned)fixed_in_vec_bit>>(31-ind_hamming));
                            //printf("test shift : %08x\n",(0x80000000>>ind_hamming));

                            ind_hamming_bit = NUM_BIT_ATTENTION-1-ind_hamming;
                            //hamming_weight = pow(2,ind_hamming_bit);
                            //hamming_weight = pow(2,ind_hamming_bit)/pow(2,NUM_BIT_ATTENTION+1);
                            //hamming_weight = pow(2,ind_hamming_bit-NUM_BIT_ATTENTION);
                            //hamming_weight = pow(2,ind_hamming_bit)/pow(2,NUM_BIT_ATTENTION);
                            hamming_weight = pow(2,(int)(ind_hamming_bit-NUM_BIT_ATTENTION-1));
                            //hamming_weight = pow(2,ind_hamming_bit)/dot->dim_mat_y;
                            //hamming_weight = pow(2,ind_hamming_bit)/dot->dim_mat_y/pow(2,NUM_BIT_ATTENTION);
                            //hamming_weight = ind_hamming_bit;

                            //if((fixed_in_vec_bit!=fixed_in_mat_bit)&&(fixed_in_vec_bit!=0x00000000)) {
                            //    sign_deriv = -1.0;
                            //} else {
                            //    sign_deriv = 1.0;
                            //}

                            if(fixed_in_vec_bit != fixed_in_mat_bit) {
                                //tmp_a += ind_hamming_bit*pow(2,-1*ind_hamming_bit);
                                //if(ind_hamming_bit==NUM_BIT_ATTENTION-1) {
                                if(ind_hamming==0) {
                                    //tmp_a += (float)ind_hamming_bit/-2.0/scale_factor/dot->in_vec[j]*sign;

                                    //tmp_a += (float)hamming_weight*(2.0-NUM_BIT_ATTENTION)/(float)(3.0-2.0*NUM_BIT_ATTENTION);
                                    //tmp_a += sign_deriv/4.0;

                                    // last
                                    //tmp_a += sign_deriv*hamming_weight/2.0;

                                    // new
                                    //tmp_a += -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec/dot->in_vec[j];
                                    tmp_a += -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat/2.0;
                                    //printf("tmp_a : %f\n",tmp_a);
                                } else {
                                    //tmp_a += (float)ind_hamming_bit/(float)sign/(float)scale_factor/(float)pow(2,-1*ind_hamming_bit);

                                    //tmp_a += (float)hamming_weight*(1.0+(float)(2.0-NUM_BIT_ATTENTION)/(float)(3.0-2.0*NUM_BIT_ATTENTION)*sign*dot->in_vec[j])/((float)sign*scale_factor*pow(2,ind_hamming_bit));
                                    //tmp_a += (float)hamming_weight*(1.0+sign*dot->in_vec[j])/((float)sign*scale_factor*pow(2,ind_hamming_bit));
                                    //tmp_a += (float)hamming_weight*(dot->in_vec[j])/((float)scale_factor*pow(2,ind_hamming_bit));

                                    //tmp_a += sign_deriv*(sign+dot->in_vec[j])*(pow(2,(int)(-1-BW_IWL)));
                                    //tmp_a += sign_deriv*sign*(fixed_in_vec>>(BW_WL-NUM_BIT_ATTENTION))*(pow(2,(int)(-1-BW_IWL)));

                                    // last
                                    //tmp_a += sign_deriv*hamming_weight*(dot->in_vec[j])*(pow(2,(int)(ind_hamming-BW_IWL)));

                                    // new
                                    tmp_a += sign_mat/pow(2,(int)(dot->iwl+1))*(fixed_in_mat_bit-fixed_in_vec_bit);
                                }
                            }
                        }

                        //printf("tmp_a : %f\n",tmp_a);

                        tmp_b = dot->grad_in[i];

                        dot->grad_out_mat[i][j] = tmp_a*tmp_b;
                        //dot->grad_out_mat[i][j] = dot->in_vec[j]*dot->grad_in[i];
                    }
                }
            }

            for(i=0;i<dot->dim_vec;i++) {
                dot->grad_out_vec[i] = 0.0;
            }

            for(i=0;i<dot->dim_mat_x;i++) {
                for(j=0;j<dot->dim_mat_y;j++) {
                    if(dot->f_trans) {
                        dot->grad_out_vec[i] += dot->in_mat[i][j]*dot->grad_in[j];
                    } else {
                        //tmp_a = dot->in_mat[i][j];
                        // hamming distance w
                        tmp_a=0;

                        if( (FLOAT2FIXED(dot->in_mat[i][j],dot->iwl,dot->frac)&(0x80000000)) == 0x00000000 ) {
                            sign = 1.0;
                        } else {
                            sign = -1.0;
                        };

                        fixed_in_vec = FLOAT2FIXED(dot->in_vec[j],dot->iwl,dot->frac);
                        fixed_in_mat = FLOAT2FIXED(dot->in_mat[i][j],dot->iwl,dot->frac);

                        if(fixed_in_mat&(0x80000000) == 0x00000000) {
                            sign_mat = 1.0;
                        } else {
                            sign_vec = -1.0;
                        }

                        if(fixed_in_vec&(0x80000000) == 0x00000000) {
                            sign_mat = 1.0;
                        } else {
                            sign_vec = -1.0;
                        }


                        for(ind_hamming=0;ind_hamming<NUM_BIT_ATTENTION;ind_hamming++) {
                            //printf("test shift : %08x",(0x80000000>>ind_hamming));
                            //fixed_in_vec_bit = fixed_in_vec&(0x80000000>>ind_hamming);
                            //fixed_in_mat_bit = fixed_in_mat&(0x80000000>>ind_hamming);
                            fixed_in_vec_bit = (unsigned int)(fixed_in_vec&(0x80000000>>ind_hamming))>>(31-ind_hamming);
                            fixed_in_mat_bit = (unsigned int)(fixed_in_mat&(0x80000000>>ind_hamming))>>(31-ind_hamming);

                            ind_hamming_bit = NUM_BIT_ATTENTION-1-ind_hamming;
                            //hamming_weight = pow(2,ind_hamming_bit);
                            //hamming_weight = pow(2,ind_hamming_bit)/pow(2,NUM_BIT_ATTENTION+1);
                            //hamming_weight = pow(2,ind_hamming_bit)/pow(2,NUM_BIT_ATTENTION);
                            hamming_weight = pow(2,(int)(ind_hamming_bit-1-NUM_BIT_ATTENTION));
                            //hamming_weight = pow(2,ind_hamming_bit)/dot->dim_mat_y;
                            //hamming_weight = pow(2,ind_hamming_bit)/dot->dim_mat_y/pow(2,NUM_BIT_ATTENTION);
                            //hamming_weight = ind_hamming_bit;

                            if((fixed_in_vec_bit!=fixed_in_mat_bit)&&(fixed_in_vec_bit==0x00000000)) {
                                sign_deriv = -1.0;
                            } else {
                                sign_deriv = 1.0;
                            }

                            if(fixed_in_vec_bit == fixed_in_mat_bit) {
                                tmp_a += 0.0;
                            } else {
                                //tmp_a += ind_hamming_bit*pow(2,-1*ind_hamming_bit);
                                if(ind_hamming_bit==NUM_BIT_ATTENTION-1) {
                                    //tmp_a += (float)ind_hamming_bit/-2.0/scale_factor/dot->in_mat[i][j]*sign;

                                    //tmp_a += (float)hamming_weight*(2.0-NUM_BIT_ATTENTION)/(float)(3.0-2.0*NUM_BIT_ATTENTION);

                                    //tmp_a += sign_deriv/4.0;

                                    // last
                                    //tmp_a += sign_deriv*hamming_weight/2.0;

                                    // new
                                    //tmp_a += -1.0*(float)(fixed_in_vec_bit-fixed_in_mat_bit)*sign/dot->in_vec[j];
                                    tmp_a += -1.0*(fixed_in_vec_bit-fixed_in_mat_bit)*sign_vec/2.0;
                                } else {
                                    //tmp_a += (float)ind_hamming_bit/(float)sign/(float)scale_factor/(float)pow(2,-1*ind_hamming_bit);

                                    //tmp_a += (float)hamming_weight*(1.0+(float)(2.0-NUM_BIT_ATTENTION)/(float)(3.0-2.0*NUM_BIT_ATTENTION)*sign*dot->in_mat[i][j])/((float)sign*scale_factor*pow(2,ind_hamming_bit));
                                    //tmp_a += (float)hamming_weight*(1.0+sign*dot->in_mat[i][j])/((float)sign*scale_factor*pow(2,ind_hamming_bit));
                                    //tmp_a += sign_deriv*hamming_weight*(dot->in_mat[i][j])/((float)scale_factor*pow(2,ind_hamming_bit));

                                    //tmp_a += sign_deriv*(sign+dot->in_mat[i][j])*pow(2,(int)(-1-BW_IWL));
                                    //tmp_a += sign_deriv*sign*(fixed_in_mat>>(BW_WL-NUM_BIT_ATTENTION))*(pow(2,(int)(-1-BW_IWL)));

                                    // last
                                    //tmp_a += sign_deriv*hamming_weight*(dot->in_mat[i][j])*(pow(2,(int)(ind_hamming-BW_IWL)));

                                    // new
                                    tmp_a += sign_vec/pow(2,(int)(dot->iwl+1))*(fixed_in_vec_bit-fixed_in_mat_bit);
                                }
                            }
                        }

                        //printf("tmp_a : %f\n",tmp_a);
                        //printf("tmp_b : %f\n",tmp_b);

                        tmp_b = dot->grad_in[i];

                        dot->grad_out_vec[j] += tmp_a*tmp_b;
                        //dot->grad_out_vec[j] += dot->in_mat[i][j]*dot->grad_in[i];
                    }
                }
            }

        } else if(dot->attention_mode==4) {
            for(i=0;i<dot->dim_mat_x;i++) {
                for(j=0;j<dot->dim_mat_y;j++) {
                    if(dot->f_trans) {
                        dot->grad_out_mat[i][j] = dot->in_vec[i]*dot->grad_in[j];
                    } else {
                        float tmp_a;
                        float tmp_b;
                        //tmp_b=clip(dot->grad_in[i],-1.0,1.0);

                        tmp_b = dot->grad_in[i];

                        if(dot->in_vec[j] >= 0.0 ) {
                            tmp_a = 1.0;
                        } else {
                            tmp_a = -1.0;
                        }

                        tmp_a=dot->in_vec[j];

                        //dot->grad_out_mat[i][j] = dot->in_vec[j]*dot->grad_in[i];
                        dot->grad_out_mat[i][j] = tmp_a*tmp_b;
                    }
                }
            }

            for(i=0;i<dot->dim_vec;i++) {
                dot->grad_out_vec[i] = 0.0;
            }

            for(i=0;i<dot->dim_mat_x;i++) {
                for(j=0;j<dot->dim_mat_y;j++) {
                    if(dot->f_trans) {
                        dot->grad_out_vec[i] += dot->in_mat[i][j]*dot->grad_in[j];
                    } else {
                        float tmp_a;
                        float tmp_b;
                        //tmp_b=clip(grad_in[i],-1.0,1.0);

                        tmp_b = dot->grad_in[i];

                        if(dot->in_mat[i][j] >= 0.0 ) {
                            tmp_a = 1.0;
                        } else {
                            tmp_a = -1.0;
                        }

                        tmp_a = dot->in_mat[i][j];

                        //dot->grad_out_vec[j] += dot->in_mat[i][j]*dot->grad_in[i];
                        dot->grad_out_vec[j] += tmp_a*tmp_b;
                    }
                }
            }
        }
        if(verbose) {
            printf("dot mat vec bwd\n");
            printf("dim mat x : %d\n", dot->dim_mat_x);
            printf("dim mat y : %d\n", dot->dim_mat_y);
            printf("dim vec   : %d\n", dot->dim_vec);
            if(dot->f_trans) {
                printf("dim out   : %d\n", dot->dim_mat_y);
            } else {
                printf("dim out   : %d\n", dot->dim_mat_x);
            }
    
            printf("in_mat\n");
            for(i=0;i<dot->dim_mat_x;i++) {
                for(j=0;j<dot->dim_mat_y;j++) {
                    printf("%.2f ",dot->in_mat[i][j]);
                }
                printf("\n");
            }
            printf("in_vec\n");
            for(j=0;j<dot->dim_vec;j++) {
                printf("%.2f ",dot->in_vec[j]);
            }
            printf("\n");
            printf("grad_in\n");
            if(dot->f_trans) {
                for(i=0;i<dot->dim_mat_y;i++) {
                    printf("%.2f ",dot->grad_in[i]);
                }
            } else {
                for(i=0;i<dot->dim_mat_x;i++) {
                    printf("%.2f ",dot->grad_in[i]);
                }
            }
            printf("\n");
    
            printf("grad_out_mat\n");
            for(i=0;i<dot->dim_mat_x;i++) {
                for(j=0;j<dot->dim_mat_y;j++) {
                    printf("%.2f ",dot->grad_out_mat[i][j]);
                }
                printf("\n");
            }
            printf("grad_out_vec\n");
            if(dot->f_trans) {
                for(i=0;i<dot->dim_mat_y;i++) {
                    printf("%.2f ",dot->grad_out_vec[i]);
                }
            } else {
                for(i=0;i<dot->dim_mat_x;i++) {
                    printf("%.2f ",dot->grad_out_vec[i]);
                }
            }
            printf("\n");
        }
		*/
    }

    time_e = clock();

    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double dot_mat_vec_destructor(dot_mat_vec *dot) {
    clock_t time_s, time_e;
    time_s = clock();

    free(dot->grad_out_mat[0]);
    free(dot->grad_out_mat);
    /*
    unsigned int i;
    for(i=0;i<dot->dim_mat_x;i++) {
        free(dot->grad_out_mat[i]);
    }
    free(dot->grad_out_mat);
    */
    free(dot->out_vec);

    free(dot->grad_out_vec);

    if(en_gpu_model) {
        cuda_dot_mat_vec_destructor
        (
            dot->dev_out_vec,
            dot->dev_grad_out_vec,
            dot->dev_grad_out_mat,
            dot->dev_f_overflow,
            dot->dev_cliff_marker
        );
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}


////////////////////////////////////////////////////////////////////////////////
// softmax
////////////////////////////////////////////////////////////////////////////////
double softmax_constructor
(
    softmax *sf,
    unsigned int dim_max,
    bool f_exp_plan,
    bool f_shift_based,
	FILE *fp_result
)
{
    clock_t time_s, time_e;
    time_s = clock();

    sf->dim_max = dim_max;
    sf->dim = 0;

    sf->f_exp_plan = f_exp_plan;
    sf->f_shift_based = f_shift_based;

    sf->out_vec = (float*) malloc(dim_max*sizeof(float));
    sf->grad_out = (float*) malloc(dim_max*sizeof(float));

    if(en_gpu_model) {
        cuda_softmax_constructor
        (
            &(sf->dev_out_vec),
            &(sf->dev_grad_out),
            &(sf->dev_max),
            sf->dim_max
        );
    }

    time_e = clock();
    printf(" < softmax_constructor > - dim_max: %d, f_shift_based: %s\n",dim_max,BOOL_PRINTF(f_shift_based));
    fprintf(fp_result," < softmax_constructor > - dim_max: %d, f_shift_based: %s\n",dim_max,BOOL_PRINTF(f_shift_based));
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double softmax_init(softmax *sf) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i;

    for(i=0;i<sf->dim_max;i++) {
        sf->out_vec[i] = 0.0;
        sf->grad_out[i] = 0.0;
    }

    if(en_gpu_model) {
        cuda_softmax_init
        (
            sf->dev_out_vec,
            sf->dev_grad_out,
			sf->dev_max,
            sf->dim_max
        );
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double softmax_in
(
    softmax *sf,
    unsigned int dim,
    float *in_vec,
    float *grad_in,
    float *dev_in_vec,
    float *dev_grad_in
)
{
    clock_t time_s, time_e;
    time_s = clock();

    if(sf->dim_max < dim) {
        printf("*E : softmat_in : exceed max dim\n");
        exit(1);
        return -1;
    }

    sf->dim = dim;
    sf->in_vec = in_vec;

    sf->grad_in = grad_in;

    if(en_gpu_model) {
        sf->dev_in_vec = dev_in_vec;
        sf->dev_grad_in = dev_grad_in;
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double softmax_fwd(softmax *sf, bool verbose) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i;
    float tot=0.0;

    float max = sf->in_vec[0];

    //test
    float min = sf->in_vec[0];
    unsigned int ind_max=0;
    float tmp_taylor;

    if(en_gpu_model) {
        cuda_softmax_fwd
        (
            sf->dev_out_vec,
            sf->dev_in_vec,
            sf->out_vec,
            sf->in_vec,
			sf->dev_max,
            sf->dim,
            sf->f_shift_based,
            verbose
        );
    }

    if(en_cpu) {
        for(i=1;i<sf->dim;i++) {
            if(sf->in_vec[i] > max) {
                max = sf->in_vec[i];
                ind_max = i;
            }
            if(sf->in_vec[i] < min) {
                min = sf->in_vec[i];
            }
        }

        for(i=0;i<sf->dim;i++) {
            if(sf->f_exp_plan) {
                // exp plan
                sf->out_vec[i] = exp_plan(sf->in_vec[i]-max);

                // taylor expansion of EXP at 0
                /*
                tmp_taylor = sf->in_vec[i]-max;
                sf->out_vec[i] = tmp_taylor*tmp_taylor*tmp_taylor*tmp_taylor/24.0 + tmp_taylor*tmp_taylor*tmp_taylor/6.0 + tmp_taylor*tmp_taylor/2.0 + tmp_taylor + 1;
                */
                // linear
                //sf->out_vec[i] = sf->in_vec[i]-min;

                // hard max
                /*
                if(i==ind_max) {
                    sf->out_vec[i] = 1.0;
                } else {
                    sf->out_vec[i] = 0.0;
                }
                */
            } else if(sf->f_shift_based) {
                sf->out_vec[i] = pow(2,sf->in_vec[i]-max+1.0);
            } else {
// need quantization
//#ifdef APPX_ATT
                // exp
                //sf->out_vec[i] = exp(sf->in_vec[i]-max+1.0);

                // exp 2
                sf->out_vec[i] = pow(2,sf->in_vec[i]-max);

//#else
                //sf->out_vec[i] = exp(sf->in_vec[i]-max+1.0);
                // exp
                //sf->out_vec[i] = exp(sf->in_vec[i]-max);
//#endif
            }

            //
            tot += sf->out_vec[i];
        }
        // quantization
        //tot = pow(2,floor(log2f(tot)));
        //tot = pow(2,ceil(log2f(tot)));

        // normalization
        for(i=0;i<sf->dim;i++) {
            sf->out_vec[i] = sf->out_vec[i]/tot;
        }

        if(verbose) {
            printf("softmax_fwd\n");
            printf("in_vec\n");
            for(i=0;i<sf->dim;i++) {
                printf("%.3f ", sf->in_vec[i]);
            }
            printf("\nout_vec\n");
            for(i=0;i<sf->dim;i++) {
                printf("%.3f ", sf->out_vec[i]);
            }
            printf("\n");
        }
    }

    time_e = clock();

    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double softmax_bwd(softmax *sf, bool verbose) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j;


    if(en_gpu_model) {
        cuda_softmax_bwd
        (
            sf->dev_grad_in,
            sf->dev_out_vec,
            sf->dev_grad_out,
            sf->dev_in_vec,
            sf->dim,
            sf->f_shift_based,
            verbose
        );
    }

    if(en_cpu) {
        float sum=0.0;

// need quantization
//#ifdef APPX_ATT
        // linear normalization
        /*
        for(i=0;i<sf->dim;i++) {
            sum += sf->out_vec[i];

            sf->grad_out[i] = 0.0;
        }
        for(i=0;i<sf->dim;i++) {
            for(j=0;j<sf->dim;j++) {
                if(i==j) {
                    //sf->grad_out[i] += (1.0/sum - sf->out_vec[i]*sf->out_vec[i])*sf->grad_in[i];

                    sf->grad_out[i] += (sum - sf->out_vec[i])/(sum*sum)*sf->grad_in[i];

                    // norm -> exp test
                    //sf->grad_out[i] += 0.7*sf->out_vec[i]*(sum - sf->out_vec[i])/(sum*sum)*sf->grad_in[i];
                } else {
                    //sf->grad_out[i] -= (sf->out_vec[i]*sf->out_vec[j])*sf->grad_in[i];

                    sf->grad_out[i] += -1.0*sf->out_vec[i]*sf->out_vec[j]/(sum*sum)*sf->grad_in[i];

                    // norm -> exp test
                    //sf->grad_out[i] += 0.7*sf->out_vec[i]*(sum - sf->out_vec[i])/(sum*sum)*sf->grad_in[i];
                }
            }
        }
        */

        /*
        // exp normalization
        for(i=0;i<sf->dim;i++) {
            sum+= sf->out_vec[i]*sf->grad_in[i];
        }

        for(i=0;i<sf->dim;i++) {
            sf->grad_out[i] = sf->out_vec[i]*(sf->grad_in[i]-sum);
        }
        */
//#else
        for(i=0;i<sf->dim;i++) {
            sum+= sf->out_vec[i]*sf->grad_in[i];
        }

        for(i=0;i<sf->dim;i++) {
            sf->grad_out[i] = sf->out_vec[i]*(sf->grad_in[i]-sum);
        }
//#endif

        if(verbose) {
            printf("softmax_bwd\n");
            printf("out_vec\n");
            for(i=0;i<sf->dim;i++) {
                printf("%2f \n",sf->out_vec[i]);
            }
            printf("\n");
    
            printf("grad\n");
            for(i=0;i<sf->dim;i++) {
                printf("%2f \n",sf->grad_out[i]);
            }
            printf("\n");
        }
    }

    time_e = clock();



    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}


double softmax_destructor(softmax *sf) {
    clock_t time_s, time_e;
    time_s = clock();

    free(sf->out_vec);
    free(sf->grad_out);

    if(en_gpu_model) {
        cuda_softmax_destructor
        (
            sf->dev_out_vec,
            sf->dev_grad_out,
			sf->dev_max
        );
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

////////////////////////////////////////////////////////////////////////////////
// sum - element sum vector (vector + vector)
////////////////////////////////////////////////////////////////////////////////
double sum_vec_constructor
(
    sum_vec *sv,
    unsigned int dim,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac,
	unsigned int f_mode,
	FILE *fp_result
)
{
    clock_t time_s, time_e;
    time_s = clock();

    sv->dim = dim;

    sv->f_fixed = f_fixed;
    sv->iwl = iwl;
    sv->frac = frac;
	sv->f_mode = f_mode;

    sv->out_vec = (float *) malloc(dim*sizeof(float));
    sv->grad_out = (float *) malloc(dim*sizeof(float));


    if(en_gpu_model) {
        cuda_sum_vec_constructor
        (
            &(sv->dev_out_vec),
            &(sv->dev_grad_out),
            sv->dim
        );
    }

    time_e = clock();
    
    printf(" < sum_vec_constructor > - dim: %d, f_fixed: %s, iwl: %d, frac: %d, f_mode: %d\n",dim,BOOL_PRINTF(f_fixed),iwl,frac,f_mode);
    fprintf(fp_result," < sum_vec_constructor > - dim: %d, f_fixed: %s, iwl: %d, frac: %d, f_mode: %d\n",dim,BOOL_PRINTF(f_fixed),iwl,frac,f_mode);

    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double sum_vec_init(sum_vec *sv) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i;

    for(i=0;i<sv->dim;i++) {
        sv->out_vec[i] = 0.0;
        sv->grad_out[i] = 0.0;
    }

    if(en_gpu_model) {
        cuda_sum_vec_init
        (
            sv->dev_out_vec,
            sv->dev_grad_out,
            sv->dim
        );
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double sum_vec_in
(
    sum_vec *sv,
    float *in_vec_a,
    float *in_vec_b,
    float *grad_in,
    float *dev_in_vec_a,
    float *dev_in_vec_b,
    float *dev_grad_in
)
{
    clock_t time_s, time_e;
    time_s = clock();

    sv->in_vec_a = in_vec_a;
    sv->in_vec_b = in_vec_b;

    sv->grad_in = grad_in;

    if(en_gpu_model) {
        sv->dev_in_vec_a = dev_in_vec_a;
        sv->dev_in_vec_b = dev_in_vec_b;
        sv->dev_grad_in = dev_grad_in;
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}


double sum_vec_fwd(sum_vec *sv, bool verbose) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i;

    if(en_gpu_model) {
        cuda_sum_vec_fwd
        (
            sv->dev_in_vec_a,
            sv->dev_in_vec_b,
            sv->dev_out_vec,
            sv->dim,
            sv->f_fixed,
            sv->iwl,
            sv->frac,
			sv->f_mode,
            verbose
        );
    }

    if(en_cpu) {
        if(sv->f_fixed) {
            for(i=0;i<sv->dim;i++) {
                FIXED_ADD(sv->out_vec[i],sv->in_vec_a[i],sv->in_vec_b[i],sv->iwl,sv->frac);
            }
        } else {
            for(i=0;i<sv->dim;i++) {
                sv->out_vec[i] = sv->in_vec_a[i] + sv->in_vec_b[i];
            }
        }

        if(verbose) {
            printf("sum_vec_fwd\n");
            printf("vector a\n");
            for(i=0;i<sv->dim;i++) {
                printf("%.2f ",sv->in_vec_a[i]);
            }
            printf("\nvector b\n");
            for(i=0;i<sv->dim;i++) {
                printf("%.2f ",sv->in_vec_b[i]);
            }
            printf("\noutput\n");
            for(i=0;i<sv->dim;i++) {
                printf("%.2f ",sv->out_vec[i]);
            }
            printf("\n");
        }
    }

    time_e = clock();

    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double sum_vec_bwd(sum_vec *sv) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i;

    if(en_gpu_model) {
        cuda_sum_vec_bwd
        (
            sv->dev_grad_out,
            sv->dev_grad_in,
            sv->grad_in,
            sv->grad_out,
            sv->dim
        );
    }

    if(en_cpu) {
        for(i=0;i<sv->dim;i++) {
            sv->grad_out[i] = sv->grad_in[i];
        }
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}


double sum_vec_destructor(sum_vec *sv) {
    clock_t time_s, time_e;
    time_s = clock();

    free(sv->out_vec);
    free(sv->grad_out);

    if(en_gpu_model) {
        cuda_sum_vec_destructor
        (
            sv->dev_out_vec,
            sv->dev_grad_out
        );
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}


////////////////////////////////////////////////////////////////////////////////
// dense - fully connected
////////////////////////////////////////////////////////////////////////////////
double dense_constructor
(
    dense *ds,
    unsigned int dim_in,
    unsigned int dim_out,
    bool en_max_grad_l2_norm,
    float max_grad_l2_norm,
    char *activation,
    bool f_fixed,
    unsigned int iwl_in,
    unsigned int frac_in,
    unsigned int iwl_w,
    unsigned int frac_w,
	unsigned int f_mode,
	FILE *fp_result
)
{
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j;

    ds->dim_in = dim_in;
    ds->dim_out = dim_out;

    strcpy(ds->activation,activation);

    if(!strcmp(ds->activation,"NULL")) {
        //printf("NULL\n");
    } else if(!strcmp(ds->activation,"SIGMOID")) {
        //printf("SIGMOID\n");
    } else if(!strcmp(ds->activation,"RELU")) {
    //    printf("RELU\n");
    } else {
        printf("*E : dense_constructor : not supported activation function : %s\n",ds->activation);
        exit(1);
    }

    ds->en_max_grad_l2_norm = en_max_grad_l2_norm;
    ds->max_grad_l2_norm = max_grad_l2_norm;

    ds->f_fixed = f_fixed;
    ds->iwl_in = iwl_in;
    ds->frac_in = frac_in;
    ds->iwl_w = iwl_w;
    ds->frac_w = frac_w;
	ds->f_mode = f_mode;

    //ds->in_vec_stored = (float *) malloc(dim_in*sizeof(float));
    ds->out_vec = (float *) malloc(dim_out*sizeof(float));

    ds->w_mat = (float **) malloc(dim_out*sizeof(float*));
    ds->w_mat[0] = (float *) malloc(dim_out*dim_in*sizeof(float));
    for(i=1;i<dim_out;i++) {
        ds->w_mat[i] = ds->w_mat[i-1] + dim_in;
    }

    ds->w_mat_del = (float **) malloc(dim_out*sizeof(float*));
    ds->w_mat_del[0] = (float *) malloc(dim_out*dim_in*sizeof(float));
    for(i=1;i<dim_out;i++) {
        ds->w_mat_del[i] = ds->w_mat_del[i-1] + dim_in;
    }

    ds->w_mat_momentum = (float **) malloc(dim_out*sizeof(float*));
    ds->w_mat_momentum[0] = (float *) malloc(dim_out*dim_in*sizeof(float));
    for(i=1;i<dim_out;i++) {
        ds->w_mat_momentum[i] = ds->w_mat_momentum[i-1] + dim_in;
    }

    ds->bias = (float *) malloc(dim_out*sizeof(float));
    ds->bias_del = (float *) malloc(dim_out*sizeof(float));

    ds->grad_out = (float *) malloc(dim_in*sizeof(float));

    ds->grad_accum = (float **) malloc(dim_out*sizeof(float*));
    ds->grad_accum[0] = (float *) malloc(dim_out*dim_in*sizeof(float));

    for(i=1;i<dim_out;i++) {
        ds->grad_accum[i] = ds->grad_accum[i-1] + dim_in;
    }


    // adam
    ds->adam_m = (float **) malloc(dim_out*sizeof(float*));
    ds->adam_m[0] = (float *) malloc(dim_out*dim_in*sizeof(float));
    for(i=1;i<dim_out;i++) {
        ds->adam_m[i] = ds->adam_m[i-1] + dim_in;
    }

    ds->adam_v = (float **) malloc(dim_out*sizeof(float*));
    ds->adam_v[0] = (float *) malloc(dim_out*dim_in*sizeof(float));
    for(i=1;i<dim_out;i++) {
        ds->adam_v[i] = ds->adam_v[i-1] + dim_in;
    }

    // fixed point
    ds->f_overflow = (float *) malloc(dim_out*sizeof(bool));

    if(en_gpu_model) {
        cuda_dense_constructor
        (
            &(ds->dev_w_mat),
            &(ds->dev_w_mat_del),
			&(ds->dev_w_mat_best),
            &(ds->dev_bias),
            &(ds->dev_bias_del),
            &(ds->dev_out_vec),
            &(ds->dev_grad_out),
            &(ds->dev_grad_l2_norm),
            &(ds->dev_grad_bias_l2_norm),
            &(ds->dev_f_overflow),
            ds->dim_in,
            ds->dim_out
        );
    }

    time_e = clock();

    printf(" < dense constructor > - dim_out: %d, dim_in: %d, act: %s, f_fixed: %s, iwl_in: %d, frac_in: %d, iwl_w: %d, frac_w: %d, f_mode: %d\n",dim_out,dim_in,activation,BOOL_PRINTF(f_fixed),iwl_in,frac_in,iwl_w,frac_w,f_mode);
    fprintf(fp_result," < dense constructor > - dim_out: %d, dim_in: %d, act: %s, f_fixed: %s, iwl_in: %d, frac_in: %d, iwl_w: %d, frac_w: %d, f_mode: %d\n",dim_out,dim_in,activation,BOOL_PRINTF(f_fixed),iwl_in,frac_in,iwl_w,frac_w,f_mode);
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double dense_init(dense *ds) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j;

    for(i=0;i<ds->dim_out;i++) {
        ds->out_vec[i] = 0.0;
    }

    for(i=0;i<ds->dim_in;i++) {
        ds->grad_out[i] = 0.0;
    }

    for(i=0;i<ds->dim_out;i++) {
        for(j=0;j<ds->dim_in;j++) {
            ds->w_mat_del[i][j] = 0.0;
            ds->grad_accum[i][j] = 0.0;
        }
    }

    // weight
    for(i=0;i<ds->dim_out;i++) {
        for(j=0;j<ds->dim_in;j++) {
            //ds->w_mat[i][j] = 0.0;
            //ds->w_mat[i][j] = 0.1*i;
            //ds->w_mat[i][j] = rand()/(double)RAND_MAX*2-1;

            ds->w_mat[i][j] = gaussian_random(0.0, 0.1);

            if(ds->f_fixed) {
                //ds->w_mat[i][j] = FLOAT_QUANT(ds->w_mat[i][j],ds->iwl,ds->frac);
                //printf("dense> weight init : NOT QUANT\n");
            }
        }
    }

    // momentum
    /*
    for(i=0;i<ds->dim_out;i++) {
        for(j=0;j<ds->dim_in;j++) {
            ds->w_mat_momentum[i][j] = 0.0;
        }
    }
    */

    /*
    // bias
    for(i=0;i<ds->dim_out;i++) {
        ds->bias[i] = gaussian_random(0.0, 0.1);
    }
    */

    // adam
    /*
    for(i=0;i<ds->dim_out;i++) {
        for(j=0;j<ds->dim_in;j++) {
            ds->adam_m[i][j] = 0.0;
        }
    }

    for(i=0;i<ds->dim_out;i++) {
        for(j=0;j<ds->dim_in;j++) {
            ds->adam_v[i][j] = 0.00000001;
        }
    }

    ds->adam_beta_1 = 0.9;
    ds->adam_beta_2 = 0.999;
    */

    if(en_gpu_model) {
        cuda_dense_init
        (
            ds->dev_out_vec,
            ds->dev_grad_out,
            ds->dev_w_mat_del,
            ds->dev_w_mat,
            ds->dev_bias,
            ds->dev_bias_del,
            ds->w_mat[0],
            ds->bias,
            ds->dev_f_overflow,
            ds->dim_in,
            ds->dim_out
        );
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double dense_in
(
    dense *ds,
    float *in_vec,
    float *grad_in,
    float *dev_in_vec,
    float *dev_grad_in
)
{
    clock_t time_s, time_e;
    time_s = clock();

    ds->in_vec = in_vec;

    ds->grad_in = grad_in;

    if(en_gpu_model) {
        ds->dev_in_vec = dev_in_vec;
        ds->dev_grad_in = dev_grad_in;
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double dense_fwd(dense *ds, bool verbose) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j;

    if(en_gpu_model) {
        cuda_dense_fwd
        (
            ds->dev_w_mat,
            ds->dev_bias,
            ds->dev_in_vec,
            ds->dev_out_vec,
            ds->dev_f_overflow,
            ds->dim_in,
            ds->dim_out,
            ds->activation,
            ds->f_fixed,
            ds->iwl_in,
            ds->frac_in,
            ds->iwl_w,
            ds->frac_w,
			ds->f_mode,
            verbose
        );
    }

    if(en_cpu) {
		printf("NOT FIXED YET CPU MODE\n");
		/*
        for(i=0;i<ds->dim_out;i++) {
            ds->out_vec[i] = 0.0;
        }

        if(ds->f_fixed) {
            for(i=0;i<ds->dim_out;i++) {
                for(j=0;j<ds->dim_in;j++) {
                    //ds->out_vec[i] += FLOAT_QUANT(ds->w_mat[i][j])*FLOAT_QUANT(ds->in_vec[j]);
                    FIXED_MAC(ds->out_vec[i],ds->w_mat[i][j],ds->in_vec[j],ds->iwl,ds->frac);
                }
                ds->out_vec[i] = FLOAT_QUANT(ds->out_vec[i],ds->iwl,ds->frac);
            }

            // activation output
            if(!strcmp(ds->activation,"SIGMOID")) {
                for(i=0;i<ds->dim_out;i++) {
                    ds->out_vec[i] = sigmoid(ds->out_vec[i]);

                    ds->out_vec[i] = FLOAT_QUANT(ds->out_vec[i],ds->iwl,ds->frac);
                }
            } else if(!strcmp(ds->activation,"RELU")) {
                for(i=0;i<ds->dim_out;i++) {
                    if(ds->out_vec[i]>0) {
                        ds->out_vec[i] = ds->out_vec[i];
                    } else {
                        ds->out_vec[i] = 0.0;
                    }
                    ds->out_vec[i] = FLOAT_QUANT(ds->out_vec[i],ds->iwl,ds->frac);
                }
            }
        } else {
            for(i=0;i<ds->dim_out;i++) {
                for(j=0;j<ds->dim_in;j++) {
                    ds->out_vec[i] += ds->w_mat[i][j]*ds->in_vec[j];
                }
            }

            // activation output
            if(!strcmp(ds->activation,"SIGMOID")) {
                for(i=0;i<ds->dim_out;i++) {
                    ds->out_vec[i] = sigmoid(ds->out_vec[i]);
                }
            } else if(!strcmp(ds->activation,"RELU")) {
                for(i=0;i<ds->dim_out;i++) {
                    if(ds->out_vec[i]>0) {
                        ds->out_vec[i] = ds->out_vec[i];
                    } else {
                        ds->out_vec[i] = 0.0;
                    }
                }
            }
        }
        if(verbose) {
            printf("dense_fwd\n");
            printf("input\n");
            for(i=0;i<ds->dim_in;i++) {
                printf("%.2f ",ds->in_vec[i]);
            }
            printf("\nweight\n");
            for(i=0;i<ds->dim_out;i++) {
                for(j=0;j<ds->dim_in;j++) {
                    printf("%.2f ",ds->w_mat[i][j]);
                }
                printf("\n");
            }
            printf("output\n");
            for(i=0;i<ds->dim_out;i++) {
                printf("%.2f ",ds->out_vec[i]);
            }
            printf("\n");
        }
		*/
    }

    time_e = clock();

#ifdef VERIFICATION_GPU
// verification_point_start
        float *veri_dev_w_mat;
        float *veri_dev_in_vec;
        float *veri_dev_out_vec;

        veri_dev_w_mat = (float*) malloc(ds->dim_out*ds->dim_in*sizeof(float));
        veri_dev_in_vec = (float*) malloc(ds->dim_in*sizeof(float));
        veri_dev_out_vec = (float*) malloc(ds->dim_out*sizeof(float));

        cuda_memcpy_dev_to_host
        (
            veri_dev_w_mat,
            ds->dev_w_mat,
            ds->dim_out*ds->dim_in
        );

        cuda_memcpy_dev_to_host
        (
            veri_dev_in_vec,
            ds->dev_in_vec,
            ds->dim_in
        );

        cuda_memcpy_dev_to_host
        (
            veri_dev_out_vec,
            ds->dev_out_vec,
            ds->dim_out
        );


        for(i=0;i<ds->dim_out;i++) {
            for(j=0;j<ds->dim_in;j++) {
                if((ds->w_mat[i][j]-veri_dev_w_mat[i*ds->dim_in+j]) > TH_ERROR_FLOAT) {
                    printf("dense_fwd: w_mat : %dx%d\n",ds->dim_out,ds->dim_in);
                    printf("error %d %d: CPU : %f : GPU : %f\n",i,j,ds->w_mat[i][j],veri_dev_w_mat[i*ds->dim_in+j]);
                }
            }
        }

        for(i=0;i<ds->dim_in;i++) {
            if((ds->in_vec[i]-veri_dev_in_vec[i]) > TH_ERROR_FLOAT) {
                printf("dense_fwd: in_vec\n");
                printf("error %d: CPU : %f : GPU : %f\n",i,ds->in_vec[i],veri_dev_in_vec[i]);
            }
        }

        for(i=0;i<ds->dim_out;i++) {
            if((ds->out_vec[i]-veri_dev_out_vec[i]) > TH_ERROR_FLOAT) {
                printf("dense_fwd: out_vec\n");
                printf("error %d: CPU : %f : GPU : %f\n",i,ds->out_vec[i],veri_dev_out_vec[i]);
                printf("in_vec\n");
                for(j=0;j<ds->dim_in;j++) {
                    if((ds->in_vec[j]-veri_dev_in_vec[j]) > TH_ERROR_FLOAT) {
                        printf("%d: %f : %f\n",j,ds->in_vec[j],veri_dev_in_vec[j]);
                    }
                }
            }
        }
// verification_point_end
#endif

    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double dense_bwd(dense *ds, bool verbose) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j;

    float deriv;

    if(en_gpu_model) {
        cuda_dense_bwd
        (
            ds->dev_w_mat,
            ds->dev_w_mat_del,
            ds->dev_bias,
            ds->dev_bias_del,
            ds->dev_in_vec,
            ds->dev_out_vec,
            ds->dev_grad_in,
            ds->dev_grad_out,
            ds->dev_f_overflow,
            ds->dim_in,
            ds->dim_out,
            ds->activation,
#ifdef EN_GRAD_QUANT
            ds->f_fixed,
#else
            false,
#endif
            ds->iwl_in,
            ds->frac_in,
            ds->iwl_w,
            ds->frac_w,
			ds->f_mode,
            verbose
        );
    }

    if(en_cpu) {
		printf("NOT YET FIX CPU MODE\n");
		/*
        if(ds->f_fixed) {
            for(i=0;i<ds->dim_out;i++) {
                for(j=0;j<ds->dim_in;j++) {
                    //ds->w_mat_del[i][j] += FLOAT_QUANT(ds->in_vec[j])*FLOAT_QUANT(ds->grad_in[i]);
                    FIXED_MAC(ds->w_mat_del[i][j],ds->in_vec[j],ds->grad_in[i],ds->iwl,ds->frac);
                }
                ds->w_mat_del[i][j] = FLOAT_QUANT(ds->w_mat_del[i][j],ds->iwl,ds->frac);
            }

            //for(i=0;i<ds->dim_out;i++) {
            //    ds->bias_del[i] += ds->grad_in[i];
            //}

            for(i=0;i<ds->dim_in;i++) {
                ds->grad_out[i] = 0.0;
            }

            for(i=0;i<ds->dim_in;i++) {
                for(j=0;j<ds->dim_out;j++) {
                    if(!strcmp(ds->activation,"SIGMOID")) {
                        deriv = FLOAT_QUANT(sigmoid_deriv(ds->out_vec[j]),ds->iwl,ds->frac);
                    } else if(!strcmp(ds->activation,"RELU")) {
                        if(ds->out_vec[j]>0) {
                            deriv = 1.0;
                        } else {
                            deriv = 0.0;
                        }
                    } else {
                        deriv = 1.0;
                    }
                    ds->grad_out[i] += FLOAT_QUANT(ds->w_mat[j][i],ds->iwl,ds->frac)*FLOAT_QUANT(ds->grad_in[j],ds->iwl,ds->frac)*FLOAT_QUANT(deriv,ds->iwl,ds->frac);
                }
                ds->grad_out[i] = FLOAT_QUANT(ds->grad_out[i],ds->iwl,ds->frac);
            }
        } else {
            for(i=0;i<ds->dim_out;i++) {
                for(j=0;j<ds->dim_in;j++) {
                    ds->w_mat_del[i][j] += ds->in_vec[j]*ds->grad_in[i];
                }
            }

            //for(i=0;i<ds->dim_out;i++) {
            //    ds->bias_del[i] += ds->grad_in[i];
            //}

            for(i=0;i<ds->dim_in;i++) {
                ds->grad_out[i] = 0.0;
            }

            for(i=0;i<ds->dim_in;i++) {
                for(j=0;j<ds->dim_out;j++) {
                    if(!strcmp(ds->activation,"SIGMOID")) {
                        deriv = sigmoid_deriv(ds->out_vec[j]);
                    } else if(!strcmp(ds->activation,"RELU")) {
                        if(ds->out_vec[j]>0) {
                            deriv = 1.0;
                        } else {
                            deriv = 0.0;
                        }
                    } else {
                        deriv = 1.0;
                    }

                    ds->grad_out[i] += ds->w_mat[j][i]*ds->grad_in[j]*deriv;
                }
            }
        }
        if(verbose) {
        printf("dense_bwd\n");
            printf("in_vec\n");
            for(j=0;j<ds->dim_in;j++) {
                printf("%.2f ",ds->in_vec[j]);
            }
            printf("\n");
            printf("grad\n");
            for(j=0;j<ds->dim_in;j++) {
                printf("%.2f ",ds->grad_out[j]);
            }
            printf("\n");
    
            printf("w_mat_del\n");
            for(i=0;i<ds->dim_out;i++) {
                for(j=0;j<ds->dim_in;j++) {
                    printf("%.2f ",ds->w_mat_del[i][j]);
                }
                printf("\n");
            }
    
            printf("bias_del\n");
            for(i=0;i<ds->dim_out;i++) {
                printf("%.2f ",ds->bias_del[i]);
            }
            printf("\n");
        }
		*/
    }

    time_e = clock();

#ifdef VERIFICATION_GPU
// verification_point_start
        float *veri_dev_w_mat_del;
        float *veri_dev_in_vec;
        float *veri_dev_grad_in;

        veri_dev_w_mat_del = (float*) malloc(ds->dim_out*ds->dim_in*sizeof(float));
        veri_dev_in_vec = (float*) malloc(ds->dim_in*sizeof(float));
        veri_dev_grad_in = (float*) malloc(ds->dim_out*sizeof(float));

        cuda_memcpy_dev_to_host
        (
            veri_dev_w_mat_del,
            ds->dev_w_mat_del,
            ds->dim_out*ds->dim_in
        );

        cuda_memcpy_dev_to_host
        (
            veri_dev_in_vec,
            ds->dev_in_vec,
            ds->dim_in
        );

        cuda_memcpy_dev_to_host
        (
            veri_dev_grad_in,
            ds->dev_grad_in,
            ds->dim_out
        );

        for(i=0;i<ds->dim_out;i++) {
            for(j=0;j<ds->dim_in;j++) {
                //printf("%d %d: CPU : %f : GPU : %f\n",i,j,ds->w_mat_del[i][j],veri_dev_w_mat_del[i*ds->dim_in+j]);
                if((ds->w_mat_del[i][j]-veri_dev_w_mat_del[i*ds->dim_in+j]) > TH_ERROR_FLOAT) {
                    printf("dense_bwd: w_mat_del : %dx%d\n",ds->dim_out,ds->dim_in);
                    printf("error %d %d: CPU : %f : GPU : %f\n",i,j,ds->w_mat_del[i][j],veri_dev_w_mat_del[i*ds->dim_in+j]);
                    printf(" in_vec: CPU:GPU : %f:%f, grad_in: %f:%f\n",ds->in_vec[j],veri_dev_in_vec[j],ds->grad_in[i],veri_dev_grad_in[i]);
                }
            }
        }
// verification_point_end
#endif

    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}


double dense_w_up(dense *ds, float lr, unsigned int m, float lambda, bool verbose) {
    clock_t time_s, time_e, time;
    time_s = clock();
    unsigned int i, j;

    if(en_gpu_model) {
        cuda_dense_w_up
        (
            ds->dev_w_mat,
            ds->dev_w_mat_del,
            ds->dev_bias,
            ds->dev_bias_del,
            ds->dev_grad_l2_norm,
            ds->dev_grad_bias_l2_norm,
            ds->dim_in,
            ds->dim_out,
            m,
            &lr,
            &lambda,
            &(ds->max_grad_l2_norm),
            //ds->f_fixed
            false,
            ds->iwl_w,
            ds->frac_w,
			ds->f_mode,
            verbose
        );
    }

    if(en_cpu) {
		printf("NOY YET FIXED CPU MODE\n");
		/*
        ds->grad_l2_norm = 0.0;

        for(i=0;i<ds->dim_out;i++) {
            for(j=0;j<ds->dim_in;j++) {
                ds->grad_l2_norm += ds->w_mat_del[i][j]*ds->w_mat_del[i][j];
            }
        }

        ds->grad_l2_norm = sqrt(ds->grad_l2_norm);

        if(ds->en_max_grad_l2_norm) {
            if(ds->grad_l2_norm > ds->max_grad_l2_norm) {
                for(i=0;i<ds->dim_out;i++) {
                    for(j=0;j<ds->dim_in;j++) {
                        ds->w_mat_del[i][j] = ds->w_mat_del[i][j]*ds->max_grad_l2_norm/ds->grad_l2_norm;

                        if(ds->f_fixed) {
                            ds->w_mat_del[i][j] = FLOAT_QUANT(ds->w_mat_del[i][j],ds->iwl,ds->frac);
                        }
                    }
                }
            }
        }

        // sgd
        for(i=0;i<ds->dim_out;i++) {
            for(j=0;j<ds->dim_in;j++) {
                if(ds->f_fixed) {
                    ds->w_mat[i][j] += FLOAT_QUANT(lr/m*ds->w_mat_del[i][j],ds->iwl,ds->frac) + FLOAT_QUANT(lr*lambda*ds->w_mat[i][j],ds->iwl,ds->frac);
                    ds->w_mat[i][j] = FLOAT_QUANT(ds->w_mat[i][j],ds->iwl,ds->frac);
                } else {
                    ds->w_mat[i][j] += lr/m*ds->w_mat_del[i][j] + lr*lambda*ds->w_mat[i][j];
                }
            }
        }
		*/

        // bias
        /*
        ds->grad_bias_l2_norm = 0.0;
        for(i=0;i<ds->dim_out;i++) {
            ds->grad_bias_l2_norm += ds->bias[i]*ds->bias[i];
        }

        ds->grad_bias_l2_norm = sqrt(ds->grad_bias_l2_norm);

        if(ds->en_max_grad_l2_norm) {
            if(ds->grad_bias_l2_norm > ds->max_grad_l2_norm) {
                for(i=0;i<ds->dim_out;i++) {
                    ds->bias[i] = ds->bias_del[i]*ds->max_grad_l2_norm/ds->grad_bias_l2_norm;
                }
            }
        }

        for(i=0;i<ds->dim_out;i++) {
            ds->bias[i] += lr/m*ds->bias_del[i];
        }
        */

        /*
        // sgd w/ adaMax
        float max_in_1;
        float max_in_2;

        for(i=0;i<ds->dim_out;i++) {
            for(j=0;j<ds->dim_in;j++) {
                ds->adam_m[i][j] = ds->adam_beta_1*ds->adam_m[i][j] + (1.0-ds->adam_beta_1)*ds->w_mat_del[i][j];
                //ds->adam_v[i][j] = ds->adam_beta_2*ds->adam_v[i][j] + (1.0-ds->adam_beta_2)*(ds->w_mat_del[i][j])*(ds->w_mat_del[i][j]);

                max_in_1 = ds->adam_beta_2*ds->adam_v[i][j];

                // abs
                if(ds->w_mat_del[i][j] < 0 ) {
                    max_in_2 = -1.0*ds->w_mat_del[i][j];
                } else {
                    max_in_2 = ds->w_mat_del[i][j];
                }


                if(max_in_1 >= max_in_2) {
                    ds->adam_v[i][j] = max_in_1;
                } else {
                    ds->adam_v[i][j] = max_in_2;
                }

                //printf("%d %d\n",i,j);
                //printf("max 1 : %f\n",max_in_1);
                //printf("max 2 : %f\n",max_in_2);

                //printf("ds->adam_v : %f\n",ds->adam_v[i][j]);
            }
        }

        for(i=0;i<ds->dim_out;i++) {
            for(j=0;j<ds->dim_in;j++) {

                //printf("ds->adam_v : %f\n",ds->adam_v[i][j]);
                ds->w_mat[i][j] += lr/(1.0-ds->adam_beta_1)*ds->adam_m[i][j]/ds->adam_v[i][j];
            }
        }

        */

        /*
        // sgd w/ momentum
        for(i=0;i<ds->dim_out;i++) {
            for(j=0;j<ds->dim_in;j++) {
                ds->w_mat_momentum[i][j] = 0.9*ds->w_mat_momentum[i][j] + lr/m*ds->w_mat_del[i][j];
                ds->w_mat[i][j] += ds->w_mat_momentum[i][j] + lr*lambda*ds->w_mat[i][j];
                //ds->w_mat[i][j] += ds->w_mat_momentum[i][j];
            }
        }
        */
        if(verbose) {
            printf("dense_w_up - (dim_outxdim_in): %dx%d\n",ds->dim_out,ds->dim_in);
            printf("w_mat_del\n");
            for(i=0;i<ds->dim_out;i++) {
                for(j=0;j<ds->dim_in;j++) {
                    printf("%.2f ",ds->w_mat_del[i][j]);
                }
                printf("\n");
            }
            printf("\n");
            printf("w_mat\n");
            for(i=0;i<ds->dim_out;i++) {
                for(j=0;j<ds->dim_in;j++) {
                    printf("%.2f ",ds->w_mat[i][j]);
                }
                printf("\n");
            }
            printf("\n");
    
            printf("bias_del\n");
            for(i=0;i<ds->dim_out;i++) {
                printf("%.2f ",ds->bias_del[i]);
            }
            printf("\n");
            printf("bias\n");
            for(i=0;i<ds->dim_out;i++) {
                printf("%.2f ",ds->bias[i]);
            }
            printf("\n");
        }
    }

    /*
    // rmsprop
    float momentum_rms = 0.9;
    float grad_tmp;

    for(i=0;i<ds->dim_out;i++) {
        for(j=0;j<ds->dim_in;j++) {
            ds->grad_accum[i][j] = momentum_rms*ds->grad_accum[i][j] + (1.0-momentum_rms)*ds->grad[i][j]*ds->grad[i][j];
            //printf("%f\n",ds->grad_accum[i][j]);
            ds->w_mat[i][j] += lr/m/(sqrt(ds->grad_accum[i][j]))*ds->grad[i][j]+lr*lambda*ds->w_mat[i][j];
        }
    }
    */
/*

    // fixed point print test
        printf("fixed point test - dense_w_up\n");

        printf("w_mat\n");
        for(i=0;i<ds->dim_out;i++) {
            for(j=0;j<ds->dim_in;j++) {
                printf("%.4f ",ds->w_mat[i][j]);
            }
            printf("\n");
        }
        printf("\n");

        for(i=0;i<ds->dim_out;i++) {
            for(j=0;j<ds->dim_in;j++) {
                printf("%08x ",FLOAT2FIXED(ds->w_mat[i][j]));
            }
            printf("\n");
        }
        printf("\n");
*/

    time_e = clock();

    time = time_e-time_s;



    time_s = clock();

    if(en_cpu) {
        for(i=0;i<ds->dim_out;i++) {
            for(j=0;j<ds->dim_in;j++) {
                ds->w_mat_del[i][j] = 0.0;
            }
        }

        for(i=0;i<ds->dim_out;i++) {
            ds->bias_del[i] = 0.0;
        }
    }

    time_e = clock();

    return (double)(time_e-time_s+time)/(double)CLOCKS_PER_SEC;
}

double dense_destructor(dense *ds) {
    clock_t time_s, time_e;
    time_s = clock();

    //free(ds->in_vec_stored);
    free(ds->w_mat[0]);
    free(ds->w_mat);
    free(ds->w_mat_del[0]);
    free(ds->w_mat_del);
    free(ds->out_vec);
    free(ds->f_overflow);


    if(en_gpu_model) {
        cuda_dense_destructor(
            ds->dev_w_mat,
            ds->dev_w_mat_del,
            ds->dev_w_mat_best,
            ds->dev_out_vec,
            ds->dev_grad_out,
            ds->dev_grad_l2_norm,
            ds->dev_grad_bias_l2_norm,
            ds->dev_f_overflow
        );
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}


////////////////////////////////////////////////////////////////////////////////
// dense_mat - fully connected
////////////////////////////////////////////////////////////////////////////////
double dense_mat_constructor
(
    dense_mat *ds_m,
    unsigned int dim_len_max,
    unsigned int dim_in,
    unsigned int dim_out,
    bool en_max_grad_l2_norm,
    float max_grad_l2_norm,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac,
	unsigned int f_mode,
	FILE *fp_result
)
{
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j;

    ds_m->dim_len_max = dim_len_max;
    ds_m->dim_in = dim_in;
    ds_m->dim_out = dim_out;
    ds_m->en_max_grad_l2_norm = en_max_grad_l2_norm;
    ds_m->max_grad_l2_norm = max_grad_l2_norm;

    ds_m->dim_len = 0;

    ds_m->f_fixed = f_fixed;
    ds_m->iwl = iwl;
    ds_m->frac = frac;
	ds_m->f_mode = f_mode;

    ds_m->out_mat = (float **) malloc(dim_len_max*sizeof(float*));
    ds_m->out_mat[0] = (float *) malloc(dim_len_max*dim_out*sizeof(float));
    for(i=1;i<dim_len_max;i++) {
        ds_m->out_mat[i] = ds_m->out_mat[i-1] + dim_out;
    }

    ds_m->w_mat = (float **) malloc(dim_out*sizeof(float*));
    ds_m->w_mat[0] = (float *) malloc(dim_out*dim_in*sizeof(float));
    for(i=1;i<dim_out;i++) {
        ds_m->w_mat[i] = ds_m->w_mat[i-1] + dim_in;
    }

    ds_m->w_mat_del = (float **) malloc(dim_out*sizeof(float*));
    ds_m->w_mat_del[0] = (float *) malloc(dim_out*dim_in*sizeof(float));
    for(i=1;i<dim_out;i++) {
        ds_m->w_mat_del[i] = ds_m->w_mat_del[i-1] + dim_in;
    }

    ds_m->bias = (float *) malloc(dim_out*sizeof(float));
    ds_m->bias_del = (float *) malloc(dim_out*sizeof(float));

    ds_m->grad_out = (float **) malloc(dim_len_max*sizeof(float*));
    ds_m->grad_out[0] = (float *) malloc(dim_len_max*dim_in*sizeof(float));
    for(i=1;i<dim_len_max;i++) {
        ds_m->grad_out[i] = ds_m->grad_out[i-1] + dim_in;
    }

    if(en_gpu_model) {
        cuda_dense_mat_constructor
        (
            &(ds_m->dev_w_mat),
            &(ds_m->dev_w_mat_del),
            &(ds_m->dev_w_mat_best),
            &(ds_m->dev_bias),
            &(ds_m->dev_bias_del),
            &(ds_m->dev_out_mat),
            &(ds_m->dev_grad_out),
            &(ds_m->dev_grad_l2_norm),
            &(ds_m->dev_grad_bias_l2_norm),
            &(ds_m->dev_f_overflow),
            ds_m->dim_in,
            ds_m->dim_out,
            ds_m->dim_len_max
        );
    }

    time_e = clock();

    printf(" < dense_mat_constructor > - dim_len_max: %d, dim_out: %d, dim_in: %d, f_fixed: %s, iwl: %d, frac: %d, f_mode: %d\n",dim_len_max,dim_out,dim_in,BOOL_PRINTF(f_fixed),iwl,frac,f_mode);
    fprintf(fp_result," < dense_mat_constructor > - dim_len_max: %d, dim_out: %d, dim_in: %d, f_fixed: %s, iwl: %d, frac: %d, f_mode: %d\n",dim_len_max,dim_out,dim_in,BOOL_PRINTF(f_fixed),iwl,frac,f_mode);
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double dense_mat_init(dense_mat *ds_m) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j;

    for(i=0;i<ds_m->dim_len_max;i++) {
        for(j=0;j<ds_m->dim_out;j++) {
            ds_m->out_mat[i][j] = 0.0;
        }
    }

    for(i=0;i<ds_m->dim_out;i++) {
        for(j=0;j<ds_m->dim_in;j++) {
            ds_m->w_mat_del[i][j] = 0.0;
        }
    }

    for(i=0;i<ds_m->dim_len_max;i++) {
        for(j=0;j<ds_m->dim_in;j++) {
            ds_m->grad_out[i][j] = 0.0;
        }
    }

    // weight
    for(i=0;i<ds_m->dim_out;i++) {
        for(j=0;j<ds_m->dim_in;j++) {
            //ds->w_mat[i][j] = 0.0;
            //ds->w_mat[i][j] = 0.1*i;
            //ds->w_mat[i][j] = rand()/(double)RAND_MAX*2-1;

            ds_m->w_mat[i][j] = gaussian_random(0.0, 0.1);

            if(ds_m->f_fixed) {
                //ds_m->w_mat[i][j] = FLOAT_QUANT(ds_m->w_mat[i][j],ds_m->iwl,ds_m->frac);
                //printf("dense_mat> weight init - NOT QUANT\n");
            }
        }
    }

    /*
    // bias
    for(i=0;i<ds_m->dim_out;i++) {
        ds_m->bias[i] = gaussian_random(0.0, 0.1);
    }
    */

    if(en_gpu_model) {
        cuda_dense_mat_init
        (
            ds_m->dev_out_mat,
            ds_m->dev_grad_out,
            ds_m->dev_w_mat,
            ds_m->dev_w_mat_del,
            ds_m->dev_bias,
            ds_m->dev_bias_del,
            ds_m->w_mat[0],
            ds_m->bias,
            ds_m->dev_f_overflow,
            ds_m->dim_in,
            ds_m->dim_out,
            ds_m->dim_len_max
        );
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double dense_mat_in
(
    dense_mat *ds_m,
    unsigned int dim_len,
    float **in_mat,
    float **grad_in,
    float *dev_in_mat,
    float *dev_grad_in
)
{
    clock_t time_s, time_e;
    time_s = clock();

    if(ds_m->dim_len_max < dim_len) {
        printf("*E : dense_mat_in : exceed max dim\n");
        exit(1);
    }

    ds_m->dim_len = dim_len;

    ds_m->in_mat = in_mat;

    ds_m->grad_in = grad_in;

    if(en_gpu_model) {
        ds_m->dev_in_mat = dev_in_mat;
        ds_m->dev_grad_in = dev_grad_in;
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double dense_mat_fwd(dense_mat *ds_m, bool verbose) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j, k;

    if(en_gpu_model) {
        cuda_dense_mat_fwd
        (
            ds_m->dev_w_mat,
            ds_m->dev_bias,
            ds_m->dev_in_mat,
            ds_m->dev_out_mat,
            ds_m->dev_f_overflow,
            ds_m->dim_in,
            ds_m->dim_out,
            ds_m->dim_len,
            ds_m->f_fixed,
            ds_m->iwl,
            ds_m->frac,
			ds_m->f_mode,
            verbose
        );
    }

    if(en_cpu) {
        for(i=0;i<ds_m->dim_len;i++) {
            for(j=0;j<ds_m->dim_out;j++) {
                ds_m->out_mat[i][j] = 0.0;
            }
        }

        if(ds_m->f_fixed) {
            for(i=0;i<ds_m->dim_len;i++) {
                for(j=0;j<ds_m->dim_out;j++) {
                    for(k=0;k<ds_m->dim_in;k++) {
                        //ds_m->out_mat[i][j] += FLOAT_QUANT(ds_m->w_mat[j][k])*FLOAT_QUANT(ds_m->in_mat[i][k]);
                        FIXED_MAC(ds_m->out_mat[i][j],ds_m->w_mat[j][k],ds_m->in_mat[i][k],ds_m->iwl,ds_m->frac);
                    }
                    ds_m->out_mat[i][j] = FLOAT_QUANT(ds_m->out_mat[i][j],ds_m->iwl,ds_m->frac);
                }
            }
        } else {
            for(i=0;i<ds_m->dim_len;i++) {
                for(j=0;j<ds_m->dim_out;j++) {
                    for(k=0;k<ds_m->dim_in;k++) {
                        ds_m->out_mat[i][j] += ds_m->w_mat[j][k] * ds_m->in_mat[i][k];
                    }
                }
            }
        }

        if(verbose) {
            printf("dense_mat_fwd\n");
            printf("input\n");
            for(i=0;i<ds_m->dim_len;i++) {
                for(j=0;j<ds_m->dim_in;j++) {
                    printf("%.2f ",ds_m->in_mat[i][j]);
                }
                printf("\n");
            }
            printf("\nweight\n");
            for(i=0;i<ds_m->dim_out;i++) {
                for(j=0;j<ds_m->dim_in;j++) {
                    printf("%.2f ",ds_m->w_mat[i][j]);
                }
                printf("\n");
            }
            printf("output\n");
            for(i=0;i<ds_m->dim_len;i++) {
                for(j=0;j<ds_m->dim_out;j++) {
                    printf("%.2f ",ds_m->out_mat[i][j]);
                }
                printf("\n");
            }
        }
    }
    time_e = clock();

#ifdef VERIFICATION_GPU
// verification_point_start
        float *veri_dev_out_mat;

        veri_dev_out_mat = (float*) malloc(ds_m->dim_len*ds_m->dim_out*sizeof(float));

        cuda_memcpy_dev_to_host
        (
            veri_dev_out_mat,
            ds_m->dev_out_mat,
            ds_m->dim_len*ds_m->dim_out
        );

        for(i=0;i<ds_m->dim_len;i++) {
            for(j=0;j<ds_m->dim_out;j++) {
                if((ds_m->out_mat[i][j]-veri_dev_out_mat[i*ds_m->dim_out+j]) > TH_ERROR_FLOAT) {
                    printf("dense_mat_fwd: out_mat\n");
                    printf("error %d %d: CPU : %f : GPU : %f\n",i,j,ds_m->out_mat[i][j],veri_dev_out_mat[i*ds_m->dim_out+j]);
                }
            }
        }
// verification_point_end
#endif

    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double dense_mat_bwd(dense_mat *ds_m, bool verbose) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j, k;

    if(en_gpu_model) {
        cuda_dense_mat_bwd
        (
            ds_m->dev_in_mat,
            ds_m->dev_w_mat,
            ds_m->dev_w_mat_del,
            ds_m->dev_bias,
            ds_m->dev_bias_del,
            ds_m->dev_grad_in,
            ds_m->dev_grad_out,
            ds_m->dev_f_overflow,
            ds_m->dim_in,
            ds_m->dim_out,
            ds_m->dim_len,
#ifdef EN_GRAD_QUANT
            ds_m->f_fixed,
#else
            false,
#endif
            ds_m->iwl,
            ds_m->frac,
			ds_m->f_mode,
            verbose
        );
    }

    if(en_cpu) {
        if(ds_m->f_fixed) {
            for(i=0;i<ds_m->dim_len;i++) {
                for(j=0;j<ds_m->dim_out;j++) {
                    for(k=0;k<ds_m->dim_in;k++) {
                        //ds_m->w_mat_del[j][k] += FLOAT_QUANT(ds_m->grad_in[i][j])*FLOAT_QUANT(ds_m->in_mat[i][k]);
                        FIXED_MAC(ds_m->w_mat_del[j][k],ds_m->grad_in[i][j],ds_m->in_mat[i][k],ds_m->iwl,ds_m->frac);
                    }
                    ds_m->w_mat_del[j][k] = FLOAT_QUANT(ds_m->w_mat_del[j][k],ds_m->iwl,ds_m->frac);
                }
            }

            for(i=0;i<ds_m->dim_len;i++) {
                for(j=0;j<ds_m->dim_in;j++) {
                    ds_m->grad_out[i][j] = 0.0;
                }
            }

            for(i=0;i<ds_m->dim_len;i++) {
                for(j=0;j<ds_m->dim_in;j++) {
                    for(k=0;k<ds_m->dim_out;k++) {
                        //ds_m->grad_out[i][j] += FLOAT_QUANT(ds_m->grad_in[i][k])*FLOAT_QUANT(ds_m->w_mat[k][j]);
                        FIXED_MAC(ds_m->grad_out[i][j],ds_m->grad_in[i][k],ds_m->w_mat[k][j],ds_m->iwl,ds_m->frac);
                    }
                    ds_m->grad_out[i][j] = FLOAT_QUANT(ds_m->grad_out[i][j],ds_m->iwl,ds_m->frac);
                }
            }
        } else {
            for(i=0;i<ds_m->dim_len;i++) {
                for(j=0;j<ds_m->dim_out;j++) {
                    for(k=0;k<ds_m->dim_in;k++) {
                        ds_m->w_mat_del[j][k] += ds_m->grad_in[i][j]*ds_m->in_mat[i][k];
                    }
                }

                // bias
                /*
                for(j=0;j<ds_m->dim_out;j++) {
                    ds_m->bias_del[j] += ds_m->grad_in[i][j];
                }
                */
            }

            for(i=0;i<ds_m->dim_len;i++) {
                for(j=0;j<ds_m->dim_in;j++) {
                    ds_m->grad_out[i][j] = 0.0;
                }
            }

            for(i=0;i<ds_m->dim_len;i++) {
                for(j=0;j<ds_m->dim_in;j++) {
                    for(k=0;k<ds_m->dim_out;k++) {
                        ds_m->grad_out[i][j] += ds_m->grad_in[i][k]*ds_m->w_mat[k][j];
                    }
                }
            }
        }

        if(verbose) {
        printf("dense_bwd\n");
            printf("in_mat\n");
            for(i=0;i<ds_m->dim_len;i++) {
                for(j=0;j<ds_m->dim_in;j++) {
                    printf("%.2f ",ds_m->in_mat[i][j]);
                }
                printf("\n");
            }
            printf("\n");
            printf("grad\n");
            for(i=0;i<ds_m->dim_len;i++) {
                for(j=0;j<ds_m->dim_in;j++) {
                    printf("%.2f ",ds_m->grad_out[i][j]);
                }
                printf("\n");
            }
            printf("\n");
            printf("w_mat_del\n");
            for(i=0;i<ds_m->dim_out;i++) {
                for(j=0;j<ds_m->dim_in;j++) {
                    printf("%.2f ",ds_m->w_mat_del[i][j]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    time_e = clock();


    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}


double dense_mat_w_up(dense_mat *ds_m, float lr, unsigned int m, float lambda, bool verbose) {
    clock_t time_s, time_e, time;
    time_s = clock();

    unsigned int i, j;

    if(en_gpu_model) {
        cuda_dense_mat_w_up
        (
            ds_m->dev_w_mat,
            ds_m->dev_w_mat_del,
            ds_m->dev_bias,
            ds_m->dev_bias_del,
            ds_m->dev_grad_l2_norm,
            ds_m->dev_grad_bias_l2_norm,
            ds_m->w_mat[0],
            ds_m->w_mat_del[0],
            ds_m->dim_in,
            ds_m->dim_out,
            m,
            &lr,
            &lambda,
            &(ds_m->max_grad_l2_norm),
            //ds_m->f_fixed
            false,
            ds_m->iwl,
            ds_m->frac,
			ds_m->f_mode,
            verbose
        );
    }

    if(en_cpu) {
        ds_m->grad_l2_norm = 0.0;

        for(i=0;i<ds_m->dim_out;i++) {
            for(j=0;j<ds_m->dim_in;j++) {
                ds_m->grad_l2_norm += ds_m->w_mat_del[i][j]*ds_m->w_mat_del[i][j];
            }
        }

        ds_m->grad_l2_norm = sqrt(ds_m->grad_l2_norm);

        if(ds_m->en_max_grad_l2_norm) {
            if(ds_m->grad_l2_norm > ds_m->max_grad_l2_norm) {
                for(i=0;i<ds_m->dim_out;i++) {
                    for(j=0;j<ds_m->dim_in;j++) {
                        ds_m->w_mat_del[i][j] = ds_m->w_mat_del[i][j]*ds_m->max_grad_l2_norm/ds_m->grad_l2_norm;

                        if(ds_m->f_fixed) {
                            ds_m->w_mat_del[i][j] = FLOAT_QUANT(ds_m->w_mat_del[i][j],ds_m->iwl,ds_m->frac);
                        }
                    }
                }
            }
        }

        // sgd
        if(ds_m->f_fixed) {
            for(i=0;i<ds_m->dim_out;i++) {
                for(j=0;j<ds_m->dim_in;j++) {
                    //ds_m->w_mat[i][j] += FLOAT_QUANT(lr/m*ds_m->w_mat_del[i][j]) + FLOAT_QUANT(lr*lambda*ds_m->w_mat[i][j]);
                    FIXED_MAC(ds_m->w_mat[i][j],lr/m*ds_m->w_mat_del[i][j],lr*lambda*ds_m->w_mat[i][j],ds_m->iwl,ds_m->frac);
                }
                ds_m->w_mat[i][j] = FLOAT_QUANT(ds_m->w_mat[i][j],ds_m->iwl,ds_m->frac);
            }
        } else {
            for(i=0;i<ds_m->dim_out;i++) {
                for(j=0;j<ds_m->dim_in;j++) {
                    ds_m->w_mat[i][j] += lr/m*ds_m->w_mat_del[i][j] + lr*lambda*ds_m->w_mat[i][j];
                }
            }
        }

        // bias
        /*
        ds_m->grad_bias_l2_norm = 0.0;
        for(i=0;i<ds_m->dim_out;i++) {
            ds_m->grad_bias_l2_norm += ds_m->bias[i]*ds_m->bias[i];
        }

        ds_m->grad_bias_l2_norm = sqrt(ds_m->grad_bias_l2_norm);

        if(ds_m->en_max_grad_l2_norm) {
            if(ds_m->grad_bias_l2_norm > ds_m->max_grad_l2_norm) {
                for(i=0;i<ds_m->dim_out;i++) {
                    ds_m->bias[i] = ds_m->bias_del[i]*ds_m->max_grad_l2_norm/ds_m->grad_bias_l2_norm;
                }
            }
        }

        for(i=0;i<ds_m->dim_out;i++) {
            ds_m->bias[i] += lr/m*ds_m->bias_del[i];
        }
        */

        if(verbose) {
            printf("dense_mat_w_up\n");
            printf("w_mat_del\n");
            for(i=0;i<ds_m->dim_out;i++) {
                for(j=0;j<ds_m->dim_in;j++) {
                    printf("%.2f ",ds_m->w_mat_del[i][j]);
                }
                printf("\n");
            }
            printf("\n");
            printf("w_mat\n");
            for(i=0;i<ds_m->dim_out;i++) {
                for(j=0;j<ds_m->dim_in;j++) {
                    printf("%.2f ",ds_m->w_mat[i][j]);
                }
                printf("\n");
            }
            printf("\n");
        }

    }

    time_e = clock();
    time = time_e-time_s;

#ifdef VERIFICATION_GPU
// verification_point_start
//        float *veri_dev_w_mat;
//
//        veri_dev_w_mat = (float*) malloc(ds_m->dim_out*ds_m->dim_in*sizeof(float));
//
//        cuda_memcpy_dev_to_host
//        (
//            veri_dev_w_mat,
//            ds_m->dev_w_mat,
//            ds_m->dim_out*ds_m->dim_in
//        );
//
//        for(i=0;i<ds_m->dim_out;i++) {
//            for(j=0;j<ds_m->dim_in;j++) {
//                if((ds_m->w_mat[i][j]-veri_dev_w_mat[i*ds_m->dim_in+j]) > TH_ERROR_FLOAT) {
//                    printf("dense_mat_w_up: w_mat\n");
//                    printf("error %d %d: CPU : %f : GPU : %f\n",i,j,ds_m->w_mat[i][j],veri_dev_w_mat[i*ds_m->dim_in+j]);
//                }
//            }
//        }
// verification_point_end
#endif


    time_s = clock();
    for(i=0;i<ds_m->dim_out;i++) {
        for(j=0;j<ds_m->dim_in;j++) {
            ds_m->w_mat_del[i][j] = 0.0;
        }
    }


    time_e = clock();

    return (double)(time_e-time_s+time)/(double)CLOCKS_PER_SEC;
}

double dense_mat_destructor(dense_mat *ds_m) {
    clock_t time_s, time_e;
    time_s = clock();

    free(ds_m->w_mat[0]);
    free(ds_m->w_mat);
    free(ds_m->w_mat_del[0]);
    free(ds_m->w_mat_del);
    free(ds_m->out_mat[0]);
    free(ds_m->out_mat);

    if(en_gpu_model) {
        cuda_dense_mat_destructor
        (
            ds_m->dev_w_mat,
            ds_m->dev_w_mat_del,
            ds_m->dev_w_mat_best,
            ds_m->dev_out_mat,
            ds_m->dev_grad_out,
            ds_m->dev_grad_l2_norm,
            ds_m->dev_grad_bias_l2_norm,
            ds_m->dev_f_overflow
        );
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}




////////////////////////////////////////////////////////////////////////////////
// cost function - cross entropy
////////////////////////////////////////////////////////////////////////////////
double cross_entropy_constructor
(
    cross_entropy *ce,
    unsigned int k,
	FILE *fp_result
)
{
    clock_t time_s, time_e;
    time_s = clock();

    ce->k = k;

    ce->grad_out = (float *) malloc(k*sizeof(float));

    if(en_gpu_model) {
        cuda_cross_entropy_constructor
        (
            &(ce->dev_cost_train),
            &(ce->dev_cost_valid),
            &(ce->dev_cost_test),
            &(ce->dev_m_cnt_train),
            &(ce->dev_m_cnt_valid),
            &(ce->dev_m_cnt_test),
            &(ce->dev_pred_i),
            &(ce->dev_grad_out),
            ce->k
        );
    }

    time_e = clock();
    printf(" < cross_entropy_constructor > - dim: %d\n",k);
    fprintf(fp_result," < cross_entropy_constructor > - dim: %d\n",k);
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double cross_entropy_init(cross_entropy *ce) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i;

    for(i=0;i<ce->k;i++) {
        ce->grad_out[i] = 0.0;
    }

    if(en_gpu_model) {
        cuda_cross_entropy_init
        (
		 	ce->dev_cost_train,
			ce->dev_cost_valid,
			ce->dev_cost_test,
			ce->dev_m_cnt_train,
			ce->dev_m_cnt_valid,
			ce->dev_m_cnt_test,
            ce->dev_grad_out,
            ce->k
        );
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double cross_entropy_in
(
    cross_entropy *ce,
    float *h,
    float *y,
    float *dev_h,
    float *dev_y
)
{
    clock_t time_s, time_e;
    time_s = clock();

    ce->h = h;
    ce->y = y;

    if(en_gpu_model) {
        ce->dev_h = dev_h;
        ce->dev_y = dev_y;
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double cross_entropy_run(cross_entropy *ce, unsigned int mode) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i;


    // should add weight decay term

    if(en_gpu_model) {
        cuda_cross_entropy_run
        (
            ce->dev_cost_train,
            ce->dev_cost_valid,
            ce->dev_cost_test,
			ce->dev_m_cnt_train,
			ce->dev_m_cnt_valid,
			ce->dev_m_cnt_test,
			ce->dev_pred_i,
            &(ce->cost),
            ce->dev_h,
            ce->dev_y,
            ce->h,
            ce->y,
            ce->dev_grad_out,
            ce->grad_out,
            ce->k,
			mode
        );
    }

    if(en_cpu) {
        for(i=0;i<ce->k;i++) {
            if(ce->y[i] == 1.0) {
                //ce->cost = -exp(ce->h[i])/tot;
                //ce->cost = -log((ce->h[i]));
                ce->cost = -(ce->h[i]);
            }
        }

        for(i=0;i<ce->k;i++) {
            if(ce->y[i] == 1.0) {
                ce->grad_out[i] = 1.0 - (ce->h[i]);
                //ce->grad[i] = ce->h[i] - 1.0;     // ref - facebook
            } else {
                ce->grad_out[i] = -(ce->h[i]);
                //ce->grad[i] = ce->h[i];           // ref - facebook
            }
        }
    }

// verification_point_start
/*
        float *veri_dev_h;
        float *veri_dev_grad_out;

        veri_dev_h = (float*) malloc(ce->k*sizeof(float));
        veri_dev_grad_out = (float*) malloc(ce->k*sizeof(float));

        cuda_memcpy_dev_to_host
        (
            veri_dev_h,
            ce->dev_h,
            ce->k
        );

        cuda_memcpy_dev_to_host
        (
            veri_dev_grad_out,
            ce->dev_grad_out,
            ce->k
        );

        for(i=0;i<ce->k;i++) {
            if((ce->h[i]-veri_dev_h[i]) > TH_ERROR_FLOAT*1000) {
                printf("cross_entropy_run: h\n");
                printf("error %d: CPU : %f : GPU : %f\n",i,ce->h[i],veri_dev_h[i]);
            }
        }

        for(i=0;i<ce->k;i++) {
            if((ce->grad_out[i]-veri_dev_grad_out[i]) > TH_ERROR_FLOAT*1000) {
                printf("cross_entropy_run: grad_out\n");
                printf("error %d: CPU : %f : GPU : %f\n",i,ce->grad_out[i],veri_dev_grad_out[i]);
            }
        }
*/
// verification_point_end


    /*
    printf("corss_entropy_run\n");
    printf("h\n");
    for(i=0;i<ce->k;i++) {
        printf("%.3f ",ce->h[i]);
    }
    printf("\n");
    printf("grad_out\n");
    for(i=0;i<ce->k;i++) {
        printf("%.3f ",ce->grad_out[i]);
    }
    printf("\n");
    printf("cost : %f\n",ce->cost);
    */

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

// load cost GPU to HOST
double cross_entropy_cost_load(cross_entropy *ce, float *cost_train, float *cost_valid, float *cost_test) {
    clock_t time_s, time_e;
    time_s = clock();

	cuda_cross_entropy_cost_load
	(
		ce->dev_cost_train,
		ce->dev_cost_valid,
		ce->dev_cost_test,
		cost_train,
		cost_valid,
		cost_test
	);

    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

// load match count GPU to HOST
double cross_entropy_m_cnt_load(cross_entropy *ce, unsigned int *m_cnt_train, unsigned int *m_cnt_valid, unsigned int *m_cnt_test) {
    clock_t time_s, time_e;
    time_s = clock();

	cuda_cross_entropy_m_cnt_load
	(
		ce->dev_m_cnt_train,
		ce->dev_m_cnt_valid,
		ce->dev_m_cnt_test,
		m_cnt_train,
		m_cnt_valid,
		m_cnt_test
	);

    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double cross_entropy_destructor(cross_entropy *ce) {
    clock_t time_s, time_e;
    time_s = clock();

    free(ce->grad_out);

    if(en_gpu_model) {
        cuda_cross_entropy_destructor
        (
            ce->dev_cost_train,
            ce->dev_cost_valid,
            ce->dev_cost_test,
			ce->dev_m_cnt_train,
			ce->dev_m_cnt_valid,
			ce->dev_m_cnt_test,
			ce->dev_pred_i,
            ce->dev_grad_out
        );
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}


////////////////////////////////////////////////////////////////////////////////
// maxout
////////////////////////////////////////////////////////////////////////////////
double maxout_constructor(maxout *mxout, unsigned int dim_in_max, unsigned int dim_hd_max, unsigned int dim_out_max) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j;

    mxout->dim_in_max = dim_in_max;
    mxout->dim_hd_max = dim_hd_max;
    mxout->dim_out_max = dim_out_max;

    mxout->dim_in = 0;
    mxout->dim_hd = 0;
    mxout->dim_out = 0;

    mxout->w_mat = (float ***) malloc(mxout->dim_out_max*sizeof(float**));

    for(i=0;i<mxout->dim_out_max;i++) {
        mxout->w_mat[i] = (float **) malloc(mxout->dim_hd_max*sizeof(float*));

        for(j=0;j<mxout->dim_hd_max;j++) {
            mxout->w_mat[i][j] = (float *) malloc(mxout->dim_in_max*sizeof(float));
        }
    }

    mxout->w_mat_del = (float ***) malloc(mxout->dim_out_max*sizeof(float**));

    for(i=0;i<mxout->dim_out_max;i++) {
        mxout->w_mat_del[i] = (float **) malloc(mxout->dim_hd_max*sizeof(float*));

        for(j=0;j<mxout->dim_hd_max;j++) {
            mxout->w_mat_del[i][j] = (float *) malloc(mxout->dim_in_max*sizeof(float));
        }
    }

    mxout->b_mat = (float **) malloc(mxout->dim_out_max*sizeof(float*));
    for(i=0;i<mxout->dim_out_max;i++) {
        mxout->b_mat[i] = (float *) malloc(mxout->dim_hd_max*sizeof(float));
    }

    mxout->b_mat_del = (float **) malloc(mxout->dim_out_max*sizeof(float*));
    for(i=0;i<mxout->dim_out_max;i++) {
        mxout->b_mat_del[i] = (float *) malloc(mxout->dim_hd_max*sizeof(float));
    }

    mxout->ind_max = (unsigned int *) malloc(mxout->dim_out_max*sizeof(unsigned int));

    mxout->out_vec = (float *) malloc(mxout->dim_out_max*sizeof(float));

    mxout->grad = (float *) malloc(mxout->dim_in_max*sizeof(float));


    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double maxout_init(maxout *mxout) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j, k;

    for(i=0;i<mxout->dim_out_max;i++) {
        for(j=0;j<mxout->dim_hd_max;j++) {
            for(k=0;k<mxout->dim_in_max;k++) {
                mxout->w_mat[i][j][k] = gaussian_random(0.0, 0.1);
                mxout->w_mat_del[i][j][k] = 0.0;
            }
        }
    }

    for(i=0;i<mxout->dim_out_max;i++) {
        for(j=0;j<mxout->dim_hd_max;j++) {
            mxout->b_mat[i][j] = gaussian_random(0.0, 0.1);
            mxout->b_mat_del[i][j] = 0.0;
        }
    }


    for(i=0;i<mxout->dim_out_max;i++) {
        mxout->ind_max[i] = 0;
        mxout->out_vec[i] = 0.0;
    }

    for(i=0;i<mxout->dim_in_max;i++) {
        mxout->grad[i] = 0.0;
    }

    /*
    for(i=0;i<mxout->dim_out_max;i++) {
        for(j=0;j<mxout->dim_hd_max;j++) {
            for(k=0;k<mxout->dim_in_max;k++) {
                printf("%2f ", mxout->w_mat[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }*/

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double maxout_in(maxout *mxout, unsigned int dim_in, unsigned int dim_hd, unsigned int dim_out, float *in_vec) {
    clock_t time_s, time_e;
    time_s = clock();

    mxout->dim_in = dim_in;
    mxout->dim_hd = dim_hd;
    mxout->dim_out = dim_out;

    mxout->in_vec = in_vec;


    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double maxout_fwd(maxout *mxout) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j, k;

    float max;
    float tmp;

    for(i=0;i<mxout->dim_out;i++) {
        max = 0.0;
        for(j=0;j<mxout->dim_hd;j++) {
            tmp = 0.0;
            for(k=0;k<mxout->dim_in;k++) {
                tmp += mxout->w_mat[i][j][k]*mxout->in_vec[k];
            }

            tmp += mxout->b_mat[i][j];

            if((max < tmp)||(j==0)) {
                mxout->ind_max[i] = j;
                max = tmp;
            }
        }

        mxout->out_vec[i] = max;
    }

    mxout->sq_sum = 0.0;
    for(i=0;i<mxout->dim_out;i++) {
        for(j=0;j<mxout->dim_hd;j++) {
            for(k=0;k<mxout->dim_out;k++) {
                mxout->sq_sum += mxout->w_mat[i][j][k]*mxout->w_mat[i][j][k];
            }
        }
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double maxout_bwd(maxout *mxout, float *grad) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j;

    for(i=0;i<mxout->dim_out;i++) {
        for(j=0;j<mxout->dim_in;j++) {
            mxout->w_mat_del[i][mxout->ind_max[i]][j] += mxout->in_vec[j]*grad[i];
            //mxout->grad[j] += mxout->w_mat[i][mxout->ind_max[i]][j]*grad[i];
            mxout->grad[j] += grad[i]/mxout->w_mat[i][mxout->ind_max[i]][j];
        }
        mxout->b_mat_del[i][mxout->ind_max[i]] = grad[i];
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double maxout_w_up(maxout *mxout, float lr, unsigned int m, float lambda, bool verbose) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j, k;

    for(i=0;i<mxout->dim_out;i++) {
        for(j=0;j<mxout->dim_hd;j++) {
            for(k=0;k<mxout->dim_in;k++) {
                mxout->w_mat[i][j][k] += lr/m*mxout->w_mat_del[i][j][k] + lr*lambda*mxout->w_mat[i][j][k];
            }
            mxout->b_mat[i][j] += lr/m*mxout->b_mat_del[i][j];
        }
    }

    for(i=0;i<mxout->dim_out;i++) {
        for(j=0;j<mxout->dim_hd;j++) {
            for(k=0;k<mxout->dim_in;k++) {
                mxout->w_mat_del[i][j][k] = 0.0;
            }
            mxout->b_mat_del[i][j] = 0.0;
        }
        mxout->grad[i] = 0.0;
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double maxout_destructor(maxout *mxout) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j;


    for(i=0;i<mxout->dim_out;i++) {
        for(j=0;j<mxout->dim_hd;j++) {
            free(mxout->w_mat[i][j]);
            free(mxout->w_mat_del[i][j]);
        }
    }

    for(i=0;i<mxout->dim_out;i++) {
        free(mxout->w_mat[i]);
        free(mxout->w_mat_del[i]);

        free(mxout->b_mat[i]);
    }

    free(mxout->w_mat);
    free(mxout->w_mat_del);

    free(mxout->b_mat);

    free(mxout->ind_max);
    free(mxout->out_vec);
    free(mxout->grad);

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}


////////////////////////////////////////////////////////////////////////////////
// cost function - SE - Squared Error
////////////////////////////////////////////////////////////////////////////////
double se_constructor(se *se, unsigned int dim_x_max) {
    clock_t time_s, time_e;
    time_s = clock();

    se->dim_x_max = dim_x_max;
    se->dim_x = 0;

    se->grad = (float *) malloc(dim_x_max*sizeof(float));

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double se_in(se *se, unsigned int dim_x, float *h_vec, float *y_vec) {
    clock_t time_s, time_e;
    time_s = clock();

    if( (se->dim_x_max < dim_x) ) {
        printf("*E : se_in : exceed max dim\n");
        exit(1);
        return -1;
    }

    se->dim_x = dim_x;
    se->h_vec = h_vec;
    se->y_vec = y_vec;

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double se_run(se *se) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j;

    se->cost = 0.0;

    for(i=0;i<se->dim_x;i++) {
        se->grad[i] = se->h_vec[i]-se->y_vec[i];
        se->cost += se->grad[i]*se->grad[i]/2.0;
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double se_destructor(se *se) {
    clock_t time_s, time_e;
    time_s = clock();

    free(se->grad);

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

////////////////////////////////////////////////////////////////////////////////
// mult_e_vec - element-wise multiply vector
////////////////////////////////////////////////////////////////////////////////
double mult_e_vec_constructor
(
    mult_e_vec *mev,
    unsigned int dim_max,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac
)
{
    clock_t time_s, time_e;
    time_s = clock();

    mev->dim_max = dim_max;
    mev->dim = 0;

    mev->f_fixed = f_fixed;
    mev->iwl = iwl;
    mev->frac = frac;

    mev->out_vec = (float *) malloc(dim_max*sizeof(float));
    mev->grad_out_a = (float *) malloc(dim_max*sizeof(float));
    mev->grad_out_b = (float *) malloc(dim_max*sizeof(float));

    if(en_gpu_model) {
        cuda_mult_e_vec_constructor
        (
            &(mev->dev_out_vec),
            &(mev->dev_grad_out_a),
            &(mev->dev_grad_out_b),
            mev->dim_max
        );
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double mult_e_vec_init(mult_e_vec *mev) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i;

    for(i=0;i<mev->dim_max;i++) {
        mev->out_vec[i] = 0.0;
        mev->grad_out_a[i] = 0.0;
        mev->grad_out_b[i] = 0.0;
    }

    if(en_gpu_model) {
        cuda_mult_e_vec_init
        (
            mev->dev_out_vec,
            mev->dev_grad_out_a,
            mev->dev_grad_out_b,
            mev->dim_max
        );
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double mult_e_vec_in
(
    mult_e_vec *mev,
    unsigned int dim,
    float *in_vec_a,
    float *in_vec_b,
    float *grad_in,
    float *dev_in_vec_a,
    float *dev_in_vec_b,
    float *dev_grad_in
)
{
    clock_t time_s, time_e;
    time_s = clock();

    if(mev->dim_max < dim) {
        printf("*E : mult_e_vec_in : exceed max dim\n");
        exit(1);
        return -1;
    }

    mev->dim = dim;

    mev->in_vec_a = in_vec_a;
    mev->in_vec_b = in_vec_b;

    mev->grad_in = grad_in;

    if(en_gpu_model) {
        mev->dev_in_vec_a = dev_in_vec_a;
        mev->dev_in_vec_b = dev_in_vec_b;
        mev->dev_grad_in = dev_grad_in;
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}


double mult_e_vec_fwd(mult_e_vec *mev, bool verbose) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i;

    if(en_gpu_model) {
        cuda_mult_e_vec_fwd
        (
            mev->dev_in_vec_a,
            mev->dev_in_vec_b,
            mev->dev_out_vec,
            mev->in_vec_a,
            mev->in_vec_b,
            mev->out_vec,
            mev->dim_max
        );
    } else {
        if(mev->f_fixed) {
            for(i=0;i<mev->dim;i++) {
                //mev->out_vec[i] = mult_fixed_32(mev->in_vec_a[i],mev->in_vec_b[i]);
                FIXED_MUL(mev->out_vec[i],mev->in_vec_a[i],mev->in_vec_b[i],mev->iwl,mev->frac);
            }
        } else {
            for(i=0;i<mev->dim;i++) {
                mev->out_vec[i] = mev->in_vec_a[i] * mev->in_vec_b[i];
            }
        }
    }


    time_e = clock();

    if(verbose) {
        printf("mult_e_vec_fwd\n");
        printf("vector a\n");
        for(i=0;i<mev->dim;i++) {
            printf("%.2f ",mev->in_vec_a[i]);
        }
        printf("\nvector b\n");
        for(i=0;i<mev->dim;i++) {
            printf("%.2f ",mev->in_vec_b[i]);
        }
        printf("\noutput\n");
        for(i=0;i<mev->dim;i++) {
            printf("%.2f ",mev->out_vec[i]);
        }
        printf("\n");
    }

    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double mult_e_vec_bwd(mult_e_vec *mev) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i;

    if(en_gpu_model) {
        cuda_mult_e_vec_bwd
        (
            mev->dev_in_vec_a,
            mev->dev_in_vec_b,
            mev->dev_grad_out_a,
            mev->dev_grad_out_b,
            mev->dev_grad_in,
            mev->grad_in,
            mev->grad_out_a,
            mev->grad_out_b,
            mev->dim_max
        );
        /*
        for(i=0;i<mev->dim_a;i++) {
            mev->grad_out[i] = mev->grad_in[i];
        }
        */

    } else {
        for(i=0;i<mev->dim;i++) {
            mev->grad_out_a[i] = mev->grad_in[i]*mev->in_vec_b[i];
            mev->grad_out_b[i] = mev->grad_in[i]*mev->in_vec_a[i];
        }
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}


double mult_e_vec_destructor(mult_e_vec *mev) {
    clock_t time_s, time_e;
    time_s = clock();

    free(mev->out_vec);
    free(mev->grad_out_a);
    free(mev->grad_out_b);

    if(en_gpu_model) {
        cuda_mult_e_vec_destructor
        (

        );
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}


////////////////////////////////////////////////////////////////////////////////
// mult_e_mat - element-wise multiply matrix
////////////////////////////////////////////////////////////////////////////////
double mult_e_mat_constructor
(
    mult_e_mat *mem,
    unsigned int dim_row_max,
    unsigned int dim_col_max,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac
)
{
    clock_t time_s, time_e;
    unsigned int i;

    time_s = clock();

    mem->dim_row_max = dim_row_max;
    mem->dim_col_max = dim_col_max;
    mem->dim_row = 0;
    mem->dim_col = 0;

    mem->f_fixed = f_fixed;
    mem->iwl = iwl;
    mem->frac = frac;

    mem->out_mat = (float **) malloc(dim_row_max*sizeof(float*));
    mem->out_mat[0] = (float *) malloc(dim_row_max*dim_col_max*sizeof(float));
    for(i=1;i<dim_row_max;i++) {
        mem->out_mat[i] = mem->out_mat[i-1]+dim_col_max;
    }
    mem->grad_out_a = (float **) malloc(dim_row_max*sizeof(float*));
    mem->grad_out_a[0] = (float *) malloc(dim_row_max*dim_col_max*sizeof(float));
    for(i=1;i<dim_row_max;i++) {
        mem->grad_out_a[i] = mem->grad_out_a[i-1]+dim_col_max;
    }
    mem->grad_out_b = (float **) malloc(dim_row_max*sizeof(float*));
    mem->grad_out_b[0] = (float *) malloc(dim_row_max*dim_col_max*sizeof(float));
    for(i=1;i<dim_row_max;i++) {
        mem->grad_out_b[0] = mem->grad_out_b[i-1]+dim_col_max;
    }

    if(en_gpu_model) {
        cuda_mult_e_mat_constructor
        (
            &(mem->dev_out_mat),
            &(mem->dev_grad_out_a),
            &(mem->dev_grad_out_b),
            mem->dim_row_max,
            mem->dim_col_max
        );
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double mult_e_mat_init(mult_e_mat *mem) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j;

    for(i=0;i<mem->dim_row_max;i++) {
        for(j=0;j<mem->dim_col_max;j++) {
            mem->out_mat[i][j] = 0.0;
            mem->grad_out_a[i][j] = 0.0;
            mem->grad_out_b[i][j] = 0.0;
        }
    }

    if(en_gpu_model) {
        cuda_mult_e_mat_init
        (
            mem->dev_out_mat,
            mem->dev_grad_out_a,
            mem->dev_grad_out_b,
            mem->dim_row_max,
            mem->dim_col_max
        );
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double mult_e_mat_in
(
    mult_e_mat *mem,
    unsigned int dim_row,
    unsigned int dim_col,
    float **in_mat_a,
    float **in_mat_b,
    float **grad_in,
    float **dev_in_mat_a,
    float **dev_in_mat_b,
    float **dev_grad_in
)
{
    clock_t time_s, time_e;
    time_s = clock();

    if((mem->dim_row_max < dim_row)||(mem->dim_col_max < dim_col)) {
        printf("*E : mult_e_mat_in : exceed max dim\n");
        exit(1);
        return -1;
    }

    mem->dim_row = dim_row;
    mem->dim_col = dim_col;

    mem->in_mat_a = in_mat_a;
    mem->in_mat_b = in_mat_b;

    mem->grad_in = grad_in;

    if(en_gpu_model) {
        mem->dev_in_mat_a = dev_in_mat_a;
        mem->dev_in_mat_b = dev_in_mat_b;
        mem->dev_grad_in = dev_grad_in;
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}


double mult_e_mat_fwd(mult_e_mat *mem, bool verbose) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j;

    if(en_gpu_model) {
        cuda_mult_e_mat_fwd
        (
            mem->dev_in_mat_a,
            mem->dev_in_mat_b,
            mem->dev_out_mat,
            mem->in_mat_a,
            mem->in_mat_b,
            mem->out_mat,
            mem->dim_row_max,
            mem->dim_col_max
        );
    } else {
        if(mem->f_fixed) {
            for(i=0;i<mem->dim_row;i++) {
                for(j=0;j<mem->dim_col;j++) {
                    //mem->out_mat[i][j] = mult_fixed_32(mem->in_mat_a[i][j],mem->in_mat_b[i][j]);
                    FIXED_MUL(mem->out_mat[i][j],mem->in_mat_a[i][j],mem->in_mat_b[i][j],mem->iwl,mem->frac);
                }
            }
        } else {
            for(i=0;i<mem->dim_row;i++) {
                for(j=0;j<mem->dim_col;j++) {
                    mem->out_mat[i][j] = mem->in_mat_a[i][j] * mem->in_mat_b[i][j];
                }
            }
        }
    }


    time_e = clock();

    if(verbose) {
        printf("mult_e_mat_fwd\n");
        printf("mattor a\n");
        for(i=0;i<mem->dim_row;i++) {
            for(j=0;j<mem->dim_col;j++) {
                printf("%.2f ",mem->in_mat_a[i][j]);
            }
            printf("\n");
        }
        printf("\nmattor b\n");
        for(i=0;i<mem->dim_row;i++) {
            for(j=0;j<mem->dim_col;j++) {
                printf("%.2f ",mem->in_mat_b[i][j]);
            }
            printf("\n");
        }
        printf("\noutput\n");
        for(i=0;i<mem->dim_row;i++) {
            for(j=0;j<mem->dim_col;j++) {
                printf("%.2f ",mem->out_mat[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double mult_e_mat_bwd(mult_e_mat *mem) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i, j;

    if(en_gpu_model) {
        cuda_mult_e_mat_bwd
        (
            mem->dev_in_mat_a,
            mem->dev_in_mat_b,
            mem->dev_grad_out_a,
            mem->dev_grad_out_b,
            mem->dev_grad_in,
            mem->grad_in,
            mem->grad_out_a,
            mem->grad_out_b,
            mem->dim_row_max,
            mem->dim_col_max
        );
        /*
        for(i=0;i<mem->dim_row;i++) {
            mem->grad_out[i] = mem->grad_in[i];
        }
        */

    } else {
        for(i=0;i<mem->dim_row;i++) {
            for(j=0;j<mem->dim_col;j++) {
                mem->grad_out_a[i][j] = mem->grad_in[i][j]*mem->in_mat_b[i][j];
                mem->grad_out_b[i][j] = mem->grad_in[i][j]*mem->in_mat_a[i][j];
            }
        }
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}


double mult_e_mat_destructor(mult_e_mat *mem) {
    clock_t time_s, time_e;
    time_s = clock();

    free(mem->out_mat);
    free(mem->grad_out_a);
    free(mem->grad_out_b);

    if(en_gpu_model) {
        cuda_mult_e_mat_destructor
        (

        );
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}


////////////////////////////////////////////////////////////////////////////////
// activation
////////////////////////////////////////////////////////////////////////////////
double activation_constructor
(
    activation *act,
    unsigned int dim,
    char *type_act,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac,
	unsigned int f_mode,
	FILE *fp_result
)
{
    clock_t time_s, time_e;
    unsigned int i;

    time_s = clock();

    act->dim = dim;
    strcpy(act->type_act,type_act);

    if( (strcmp(act->type_act,"NULL"))
        && (strcmp(act->type_act,"SIGMOID"))
        && (strcmp(act->type_act,"RELU")) ) {
        printf("*E : activation_constructor : not supported activation function : %s\n",act->type_act);
        exit(1);
    }

    act->f_fixed = f_fixed;
    act->iwl = iwl;
    act->frac = frac;
	act->f_mode = f_mode;

    act->out = (float*) malloc(dim*sizeof(float));
    act->grad_out = (float*) malloc(dim*sizeof(float));

    if(en_gpu_model) {
        cuda_activation_constructor
        (
            &(act->dev_out),
            &(act->dev_grad_out),
            act->dim
        );
    }

    time_e = clock();
    printf(" < activation constructor > - dim: %d, type_act: %s, f_fixed: %s, iwl: %d, frac: %d, f_mode: %d\n",dim,type_act,BOOL_PRINTF(f_fixed),iwl,frac,f_mode);
    fprintf(fp_result," < activation constructor > - dim: %d, type_act: %s, f_fixed: %s, iwl: %d, frac: %d, f_mode: %d\n",dim,type_act,BOOL_PRINTF(f_fixed),iwl,frac,f_mode);
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double activation_init(activation *act) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i;

    for(i=0;i<act->dim;i++) {
        act->out[i] = 0.0;
        act->grad_out[i] = 0.0;
    }

    if(en_gpu_model) {
        cuda_activation_init
        (
            act->dev_out,
            act->dev_grad_out,
            act->dim
        );
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double activation_in
(
    activation *act,
    float *in,
    float *grad_in,
    float *dev_in,
    float *dev_grad_in
)
{
    clock_t time_s, time_e;
    time_s = clock();

    act->in = in;
    act->grad_in = grad_in;

    if(en_gpu_model) {
        act->dev_in = dev_in;
        act->dev_grad_in = dev_grad_in;
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}


double activation_fwd(activation *act, bool verbose) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i;

    if(en_gpu_model) {
        cuda_activation_fwd
        (
            act->dev_in,
            act->dev_out,
            act->type_act,
            act->dim,
            act->f_fixed,
            act->iwl,
            act->frac,
			act->f_mode
        );
    }

    if(en_cpu) {
        if(act->f_fixed) {
            printf("not implemented fixed operation\n");
        } else {
            if(!strcmp(act->type_act,"NULL")) {
                for(i=0;i<act->dim;i++) {
                    act->out[i] = act->in[i];
                }
            } else if (!strcmp(act->type_act,"SIGMOID")) {
                for(i=0;i<act->dim;i++) {
                    act->out[i] = sigmoid(act->in[i]);
                }
            } else if (!strcmp(act->type_act,"RELU")) {
                for(i=0;i<act->dim;i++) {
                    act->out[i] = relu(act->in[i]);
                }
            }
        }
    }

    time_e = clock();

    if(verbose) {
        printf("activation_fwd: %s\n",act->type_act);
        printf("input\n");
        for(i=0;i<act->dim;i++) {
            printf("%2f ",act->in[i]);
        }
        printf("\n");

        printf("output\n");
        for(i=0;i<act->dim;i++) {
            printf("%2f ",act->out[i]);
        }
        printf("\n");
    }

    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double activation_bwd(activation *act, bool verbose) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i;

    if(en_gpu_model) {
        cuda_activation_bwd
        (
            act->dev_out,
            act->dev_grad_in,
            act->dev_grad_out,
            act->type_act,
            act->dim,
#ifdef EN_GRAD_QUANT
            act->f_fixed,
#else
			false,
#endif
            act->iwl,
            act->frac,
			act->f_mode
        );
    }

    if(en_cpu) {
        if(act->f_fixed) {
            printf("not implemented fixed operation\n");
        } else {
            if(!strcmp(act->type_act,"NULL")) {
                for(i=0;i<act->dim;i++) {
                    act->grad_out[i] = act->grad_in[i];
                }
            } else if (!strcmp(act->type_act,"SIGMOID")) {
                for(i=0;i<act->dim;i++) {
                    act->grad_out[i] = act->grad_in[i]*sigmoid_deriv(act->out[i]);
                }
            } else if (!strcmp(act->type_act,"RELU")) {
                for(i=0;i<act->dim;i++) {
                    act->grad_out[i] = act->grad_in[i]*relu_deriv(act->out[i]);
                }
            }
        }
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}


double activation_destructor(activation *act) {
    clock_t time_s, time_e;
    time_s = clock();

    free(act->out);
    free(act->grad_out);

    if(en_gpu_model) {
        cuda_activation_destructor
        (
            act->dev_out,
            act->dev_grad_out
        );
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

////////////////////////////////////////////////////////////////////////////////
// scale - learnable scale layer
////////////////////////////////////////////////////////////////////////////////
double scale_constructor
(
    scale *sc,
    unsigned int dim_max,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac,
	unsigned int f_mode,
	FILE *fp_result
)
{
    clock_t time_s, time_e;
    unsigned int i;

    time_s = clock();

    sc->dim_max = dim_max;

    sc->f_fixed = f_fixed;
    sc->iwl = iwl;
    sc->frac = frac;
	sc->f_mode = f_mode;

    sc->w = (float*) malloc(sizeof(float));
    sc->w_del = (float*) malloc(sizeof(float));

    sc->out = (float*) malloc(dim_max*sizeof(float));
    sc->grad_out = (float*) malloc(dim_max*sizeof(float));


    if(en_gpu_model) {
        cuda_scale_constructor
        (
            &(sc->dev_w),
            &(sc->dev_w_del),
            &(sc->dev_w_best),
            &(sc->dev_out),
            &(sc->dev_grad_out),
            sc->dim_max
        );
    }

    time_e = clock();

    printf(" < scale constructor > - dim_max: %d, f_fixed: %s, iwl: %d, frac: %d, f_mode: %d\n",dim_max,BOOL_PRINTF(f_fixed),iwl,frac,f_mode);
    fprintf(fp_result," < scale constructor > - dim_max: %d, f_fixed: %s, iwl: %d, frac: %d, f_mode: %d\n",dim_max,BOOL_PRINTF(f_fixed),iwl,frac,f_mode);
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double scale_init(scale *sc) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i;

    *(sc->w) = gaussian_random(0.0, 0.1);
    //*(sc->w) = 0.1;
    *(sc->w_del) = 0.0;

    for(i=0;i<sc->dim_max;i++) {
        sc->out[i] = 0.0;
        sc->grad_out[i] = 0.0;
    }

    if(en_gpu_model) {
        cuda_scale_init
        (
            sc->dev_w,
            sc->dev_w_del,
            sc->dev_out,
            sc->dev_grad_out,
            sc->w,
            sc->dim_max
        );
    }

    if(en_cpu) {
        printf("scale_init: not implemented yet\n");
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double scale_in
(
    scale *sc,
    unsigned int dim,
    float *in,
    float *grad_in,
    float *dev_in,
    float *dev_grad_in
)
{
    clock_t time_s, time_e;
    time_s = clock();

    sc->dim = dim;

    sc->in = in;
    sc->grad_in = grad_in;

    if(en_gpu_model) {
        sc->dev_in = dev_in;
        sc->dev_grad_in = dev_grad_in;
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}


double scale_fwd(scale *sc, bool verbose) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i;

    if(en_gpu_model) {
        cuda_scale_fwd
        (
            sc->dev_in,
            sc->dev_w,
            sc->dev_out,
            sc->dim,
            //sc->f_fixed,
			false,
            sc->iwl,
            sc->frac,
			sc->f_mode,
            verbose
        );
    }

    if(en_cpu) {
        if(sc->f_fixed) {
            printf("scale_fwd: not implemented fixed operation\n");
        } else {
            printf("scale_fwd: not implemented cpu fwd path\n");
        }
    }

    time_e = clock();

    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double scale_bwd(scale *sc, bool verbose) {
    clock_t time_s, time_e;
    time_s = clock();

    unsigned int i;

    if(en_gpu_model) {
        cuda_scale_bwd
        (
            sc->dev_in,
            sc->dev_grad_in,
            sc->dev_w,
            sc->dev_w_del,
            sc->dev_grad_out,
            sc->dim,
#ifdef EN_GRAD_QUANT
            sc->f_fixed,
#else
			false,
#endif
            sc->iwl,
            sc->frac,
			sc->f_mode,
            verbose
        );
    }

    if(en_cpu) {
        if(sc->f_fixed) {
            printf("scale_bwd: not implemented fixed operation\n");
        } else {
            printf("scale_bwd: not implemented scale bwd\n");
        }
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}

double scale_w_up(scale *sc, float lr, unsigned int m, float lambda, bool verbose) {
    clock_t time_s, time_e, time;
    time_s = clock();
    unsigned int i, j;

    if(en_gpu_model) {
        cuda_scale_w_up
        (
            sc->dev_w,
            sc->dev_w_del,
            //sc->dev_grad_l2_norm,
            sc->dim,
            m,
            &lr,
            &lambda,
            //&(sc->max_grad_l2_norm),
            //sc->f_fixed
            false,
            sc->iwl,
            sc->frac,
			sc->f_mode,
            verbose
        );
    }

    if(en_cpu) {
        printf("scale_w_up: not implemented yet\n");
    }

    time_e = clock();

    return (double)(time_e-time_s+time)/(double)CLOCKS_PER_SEC;
}

double scale_destructor(scale *sc) {
    clock_t time_s, time_e;
    time_s = clock();

    free(sc->out);
    free(sc->grad_out);

    if(en_gpu_model) {
        cuda_scale_destructor
        (
            sc->dev_w,
            sc->dev_w_del,
            sc->dev_w_best,
            sc->dev_out,
            sc->dev_grad_out
        );
    }

    time_e = clock();
    return (double)(time_e-time_s)/(double)CLOCKS_PER_SEC;
}
