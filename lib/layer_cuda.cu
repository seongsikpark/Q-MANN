extern "C" {
#include "layer_cuda.h"
}
////////////////////////////////////////////////////////////
// mat_vec_product kernel
////////////////////////////////////////////////////////////
//
// mat(dim_out x dim_in) * in_vec(dim_in) = out_vec(dim_out)
// dim_out = num_block
// dim_in = num_thread
//
////////////////////////////////////////////////////////////
__global__ void _cuda_printf(float *vec, unsigned int size) {
    unsigned int i;

    for(i=0;i<size;i++) {
        printf("%f \n",vec[i]);
    }
}

__global__ void _cuda_printf_vec(float *vec, unsigned int size) {
    unsigned int i;

    for(i=0;i<size;i++) {
        if(i==size-1) {
            printf("%f\n",vec[i]);
        } else {
            printf("%f, ",vec[i]);
        }
    }
}

__global__ void _cuda_printf_mat(float *mat, unsigned int row, unsigned int col) {
    unsigned int i, j;

    //printf("_cuda_printf_mat: %dx%d",row,col);

    for(i=0;i<row;i++) {
        for(j=0;j<col;j++) {
            if(j==col-1) {
                printf("%f\n",mat[i*col+j]);
            } else {
                printf("%f, ",mat[i*col+j]);
            }
        }
    }
}

__global__ void _cuda_mat_vec_product(float *mat, float *vec, float *out, float *f_overflow, bool f_fixed, unsigned int iwl_m, unsigned int frac_m, unsigned int iwl_v, unsigned int frac_v, unsigned int f_mode) {
    __shared__ float temp[CUDA_MAX_NUM_THREAD];

    if(f_fixed) {
        CUDA_FIXED_MUL(temp[threadIdx.x],mat[threadIdx.x + blockIdx.x*blockDim.x],vec[threadIdx.x],iwl_m,frac_m,iwl_v,frac_v,f_mode);

        __syncthreads();

        if(threadIdx.x==0) {
            float sum = 0;

            for(int i=0;i<blockDim.x;i++) {
                //CUDA_FIXED_ADD(sum,sum,temp[i],iwl,frac);
                sum += temp[i];
            }
            out[blockIdx.x]=CUDA_FLOAT_QUANT(sum,iwl_m,frac_m,f_mode);

            //f_overflow[blockIdx.x]=CUDA_FIXED_OVERFLOW_F(sum,iwl_v,frac_v);
            //printf("%f\n",f_overflow[blockIdx.x]);
        }
    } else {
        temp[threadIdx.x] = mat[threadIdx.x + blockIdx.x*blockDim.x]*vec[threadIdx.x];

        __syncthreads();

        if(threadIdx.x==0) {
            float sum = 0;

            for(int i=0;i<blockDim.x;i++) {
                sum += temp[i];
            }
            out[blockIdx.x]=sum;
        }
    }
}

__global__ void _cuda_mat_trans_vec_product(float *mat, float *vec, float *out) {
    __shared__ float temp[CUDA_MAX_NUM_THREAD];

    unsigned ind_mat = blockIdx.x + threadIdx.x*gridDim.x;
    unsigned ind_vec = threadIdx.x;

    temp[threadIdx.x] = mat[ind_mat]*vec[ind_vec];

    __syncthreads();

    if(threadIdx.x==0) {
        float sum = 0;

        for(int i=0;i<blockDim.x;i++) {
            sum += temp[i];
        }
        out[blockIdx.x]=sum;
    }
}

__global__ void _cuda_mat_mat_trans_product(float *mat_a, float *mat_b, float *out, unsigned int dim_out_col, bool fwd_path, float *f_overflow, bool f_fixed, unsigned int iwl_m, unsigned int frac_m, unsigned int iwl_v, unsigned int frac_v, unsigned int iwl_out, unsigned int frac_out, unsigned int f_mode) {
    __shared__ float temp[CUDA_MAX_NUM_THREAD];

    unsigned int ind_out_r;
    unsigned int ind_out_c;
    unsigned int ind_mat_a;
    unsigned int ind_mat_b;

    ind_out_r = blockIdx.x/dim_out_col;
    ind_out_c = blockIdx.x%dim_out_col;

    ind_mat_a = threadIdx.x + ind_out_r*blockDim.x;
    ind_mat_b = threadIdx.x + ind_out_c*blockDim.x;

    if(f_fixed) {
        CUDA_FIXED_MUL(temp[threadIdx.x],mat_a[ind_mat_a],mat_b[ind_mat_b],iwl_m,frac_m,iwl_v,frac_v,f_mode);
        //temp[threadIdx.x] = mat_a[ind_mat_a]*mat_b[ind_mat_b];

        __syncthreads();

        if(threadIdx.x==0) {
            float sum = 0;

            for(int i=0;i<blockDim.x;i++) {
                //CUDA_FIXED_ADD(sum,sum,temp[i],iwl,frac);
                sum += temp[i];
                //printf("%d: %f\n",i,temp[i]);
            }
            //printf("sum[%d]: %f\n",blockIdx.x,sum);

            out[blockIdx.x]=CUDA_FLOAT_QUANT(sum,iwl_out,frac_out,f_mode);
			/*
            if(fwd_path) {
                f_overflow[blockIdx.x]=CUDA_FIXED_OVERFLOW_F(sum,iwl_v,frac_v);
            }
			*/
        }
    } else {
        temp[threadIdx.x] = mat_a[ind_mat_a]*mat_b[ind_mat_b];

        __syncthreads();

        if(threadIdx.x==0) {
            float sum = 0;

            for(int i=0;i<blockDim.x;i++) {
                sum += temp[i];
            }
            out[blockIdx.x]=sum;
            /*
            if(out[blockIdx.x]!=out[blockIdx.x]) {
                printf("_cuda_mat_mat_trans_product: out[%d] NaN\n",blockIdx.x);
                printf("<<<%d,%d>>> dim_out_col : %d\n",gridDim.x,blockDim.x,dim_out_col);

                for(int i=0;i<blockDim.x;i++) {
                    printf("temp[%d]: %f\n",i,temp[i]);
                }

                printf("mat_a\n");
                _cuda_printf_mat(mat_a,gridDim.x/dim_out_col,blockDim.x);

                printf("mat_b\n");
                _cuda_printf_mat(mat_b,dim_out_col,blockDim.x);
            }
            */
        }
    }
}

__device__ void _cuda_bin2gray(unsigned int bin, unsigned int *gray, int idx_bit_low, int idx_bit_high) {
    int i;

    *gray = bin&(0x00000001<<(idx_bit_high));

    //printf("gray: %08x\n",gray[idx]);

    for(i=idx_bit_high-1;i>=idx_bit_low;i--) {
        *gray = *gray|(((((bin)>>(i+1))&(0x00000001))^(((bin)>>(i))&(0x00000001)))<<i);

        //printf("%d:%08x, %d:%08x -> %08x\n",i+1,(bin>>(i+1))&(0x00000001), i,(bin>>(i))&(0x00000001), ((bin>>(i+1))&(0x00000001))^((bin>>(i))&(0x00000001)));
        //printf("gray: %08x\n",gray);

    }
}

__global__ void _cuda_gray2bin(unsigned int *gray, unsigned int *bin, int idx_bit_low, int idx_bit_high) {
    int i;
    unsigned int bin_last;

	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    //bin = gray&(0x00000001<<(idx_bit_high));
    bin_last = (gray[idx]>>idx_bit_high)&0x00000001;
    bin[idx] = bin_last<<idx_bit_high;

    //printf("gray: %08x\n",gray);
    //


    for(i=idx_bit_high-1;i>=idx_bit_low;i--) {
        if(((gray[idx]>>(i))&0x00000001)==0x00000001) {
            bin_last = (!bin_last)&0x00000001;
        }

        bin[idx] = bin[idx]|(bin_last<<i);
        //printf("%d:%08x, %d:%08x -> %08x\n",i+1,(bin>>(i+1))&(0x00000001), i,(bin>>(i))&(0x00000001), ((bin>>(i+1))&(0x00000001))^((bin>>(i))&(0x00000001)));
    
        //printf("gray: %08x\n",gray);

    }
}

// weighted hamming similarity
__device__ void _cuda_hamming_similarity(int in_a, int in_b, unsigned int num_bit, float *similarity, bool f_weighted) {
    int i;

    float tmp_sim=0.0;

    int sign_a;
    int sign_b;

    //printf("test : %f\n",powf(2,-2));

    //new_last
    /*
    for(i=0;i<num_bit;i++) {
        if( (in_a&(0x80000000>>i)) == (in_b&(0x80000000>>i)) ) {
            //tmp_sim += powf(2,(int)(-i-1));
            tmp_sim += powf(2,(int)(-i));

            //tmp_sim += 32-i;

            //tmp_sim += powf(2,(int)32-i);

            //printf("hit: %d : %f : %f\n",num_bit-i,powf(2,num_bit-i),tmp_sim);
        }
    }
    *similarity = tmp_sim;

    if((in_a&0x80000000)==0x00000000) {
        sign_a = 1.0;
    } else {
        sign_a = -1.0;
    }

    if((in_b&0x80000000)==0x00000000) {
        sign_b = 1.0;
    } else {
        sign_b = -1.0;
    }

    if(sign_a!=sign_b) {
        *similarity = 0.0;
    }
    */

	if(f_weighted) {

        if((in_a&0x80000000)==0x00000000) {
            sign_a = 1;
        } else {
            sign_a = -1;
        }
    
        if((in_b&0x80000000)==0x00000000) {
            sign_b = 1;
        } else {
            sign_b = -1;
        }

        //printf("a : %d %f b : %d %f\n",in_a, sign_a, in_b, sign_b);

        for(i=1;i<num_bit;i++) {
            if((in_a&(0x80000000>>i)) == (in_b&(0x80000000>>i))){
            //if(((in_a&(0x80000000>>i)) == (in_b&(0x80000000>>i))) && (in_a&(0x80000000>>i))==(0x80000000>>i) ){
				// last test
				if((in_a&(0x80000000>>i))==(0x80000000>>i)) { 
                	//tmp_sim += powf(2,(int)(-i-1.0*HAMMING_WEIGHT_PARA));
                	tmp_sim += powf(2,(int)(-i));
				} else {
               		//tmp_sim += powf(2,(int)(-i-1.0*HAMMING_WEIGHT_PARA));
               		tmp_sim += powf(2,(int)(-i));
               		//tmp_sim += 0.0;
				}
            }
        }

        if(sign_a==sign_b) {
            *similarity = tmp_sim;
        } else {
            *similarity = -1.0*tmp_sim;
        }
	} else {
	    for(i=1;i<num_bit;i++) {
            if( (in_a&(0x80000000>>i)) == (in_b&(0x80000000>>i)) ) {
				tmp_sim += 1.0;
            }
        }
        *similarity = tmp_sim;
	}

    //*similarity = tmp_sim;

    /*
    if(sign_a==sign_b) {
        *similarity = tmp_sim;
    } else {
        *similarity = 0.0;
    }
    */

    /*
    if(*similarity < 0.3 && *similarity > -0.3) {
        printf("a:%08x : %08x : %d : %f\n",in_a,in_b,num_bit,*similarity);
    }
    */

    //if(*similarity!=30&&*similarity!=0) {
    //    printf("b:%08x : %08x : %d : %d\n",in_a,in_b,num_bit,*similarity);
    //}
    //printf("%08x : %08x : %d : %d\n",CUDA_FLOAT2FIXED(1.0),CUDA_FLOAT2FIXED(0.75),num_bit,*similarity);
}

// binarization
__global__ void _cuda_binarization(float *in_vec, float *out_vec) {
    unsigned int idx;

    idx = blockIdx.x*blockDim.x+threadIdx.x;

    if(in_vec[idx] >= 0.0) {
        out_vec[idx] = 1.0;
    } else {
        out_vec[idx] = -1.0;
    }
    //__syncthreads();

    //printf("%f\n",vec[idx]);
}

// quantization
__global__ void _cuda_quantization(float *vec,unsigned int iwl,unsigned int frac, unsigned int f_mode) {
    unsigned int idx;

    idx = blockIdx.x*blockDim.x+threadIdx.x;

    //vec[idx] = FIXED2FLOAT(CUDA_FLOAT2FIXED(vec[idx]));
    vec[idx] = CUDA_FLOAT_QUANT(vec[idx],iwl,frac,f_mode);
    //printf("%f\n",vec[idx]);
}

__global__ void _cuda_approximate_attention(float *in_mat, float *in_vec, float *out, unsigned int iwl, unsigned int frac, unsigned int f_mode, unsigned int num_bit_attention, float* cliff_marker) {
    __shared__ float temp[CUDA_MAX_NUM_THREAD];

    float similarity;
	//float similarity_gray;

    unsigned int ind_mat;
    unsigned int ind_vec;

	int fixed_in_mat;
	int fixed_in_vec;

	//unsigned int gray_in_mat;
	//unsigned int gray_in_vec;

    int sign_in_mat;
    int sign_in_vec;

	int abs_fixed_in_mat;
	int abs_fixed_in_vec;

	int abs_fixed_min;

    ind_mat = threadIdx.x + blockIdx.x*blockDim.x;
    ind_vec = threadIdx.x;

    //temp[threadIdx.x] = mat_a[ind_mat_a]*mat_b[ind_mat_b];

	fixed_in_mat = CUDA_FLOAT2FIXED(in_mat[ind_mat],iwl,frac,f_mode);
	fixed_in_vec = CUDA_FLOAT2FIXED(in_vec[ind_vec],iwl,frac,f_mode);

    // test 170219

    if((fixed_in_mat&0x80000000)==0x00000000) {
        sign_in_mat = 0x00000000;
    } else {
        sign_in_mat = 0x80000000;
    }

    if((fixed_in_vec&0x80000000)==0x00000000) {
        sign_in_vec = 0x00000000;
    } else {
        sign_in_vec = 0x80000000;
    }

	abs_fixed_in_mat = fixed_in_mat&0x7FFFFFFF;
	abs_fixed_in_vec = fixed_in_vec&0x7FFFFFFF;

	if(abs_fixed_in_mat >= abs_fixed_in_vec) {
		abs_fixed_min = abs_fixed_in_vec;
	} else {
		abs_fixed_min = abs_fixed_in_mat;
	}

	if(sign_in_mat==sign_in_vec) {
		fixed_in_mat = sign_in_mat|(abs_fixed_in_mat-abs_fixed_min);
		fixed_in_vec = sign_in_vec|(abs_fixed_in_vec-abs_fixed_min);
	} else {
		if(abs_fixed_in_mat >= abs_fixed_in_vec) {
			fixed_in_mat = sign_in_mat|(abs_fixed_in_mat+abs_fixed_min);
			fixed_in_vec = sign_in_vec|0x00000000;
		} else {
			fixed_in_mat = sign_in_mat|0x00000000;
			fixed_in_vec = sign_in_vec|(abs_fixed_in_vec+abs_fixed_min);
		}
	}

	// binary representation - fixed point representation
    //_cuda_hamming_similarity(CUDA_FLOAT2FIXED(in_mat[ind_mat],iwl,frac),CUDA_FLOAT2FIXED(in_vec[ind_vec],iwl,frac),num_bit_attention,&similarity,true);
    _cuda_hamming_similarity(fixed_in_mat,fixed_in_vec,num_bit_attention,&similarity,true);


	// gray code
    /*
	_cuda_bin2gray((unsigned int)(fixed_in_mat&0x7FFFFFFF),&gray_in_mat,30-num_bit_attention+2,30);
	_cuda_bin2gray((unsigned int)(fixed_in_vec&0x7FFFFFFF),&gray_in_vec,30-num_bit_attention+2,30);
    _cuda_hamming_similarity(gray_in_mat,gray_in_vec,num_bit_attention,&similarity_gray,false);
    */

    //printf("%08x, %08x, %f\n",CUDA_FLOAT2FIXED(in_mat[ind_mat]),CUDA_FLOAT2FIXED(in_vec[ind_vec]),similarity);
	

	// new try - gray code
	/*
	if(similarity_gray==6 && similarity>=0.0) {
    	similarity = 1.0;
	} else if(similarity_gray==6 && similarity<0.0) {
    	similarity = -1.0;
	}
	*/
	
	/*
	if(similarity >= 0.0) {
		if(-0.5*fabsf(in_mat[ind_mat]-in_vec[ind_vec])+0.7>similarity) {
			similarity = -0.5*fabsf(in_mat[ind_mat]-in_vec[ind_vec])+0.7;
		}
	} else {
		if(0.5*fabsf(in_mat[ind_mat]-in_vec[ind_vec])-1.3>similarity) {
			similarity = 0.5*fabsf(in_mat[ind_mat]-in_vec[ind_vec])+1.3;
		}
	}
	*/

    /*
	float diff;
    float x = FIXED_MAX_FLOAT(iwl,frac);
    float y = powf(2,-1.0*HAMMING_WEIGHT_PARA);
    float th = y*0.3;
    float a = y/x;
	diff = (in_mat[ind_mat]-in_vec[ind_vec]);
    */

    //cliff_marker[ind_mat]=0.0;

    /*
	if(diff>=0.0) {
		if((a*diff-y<similarity)&&(-a*diff+y-th>similarity)){
			//similarity=-0.5*diff+0.7;
			similarity=-a*diff+y;
            //similarity = 0.0;
            cliff_marker[ind_mat]=1.0;
		} else if (a*diff-y-th>similarity){
			//similarity=0.5*diff-1.3;
			similarity=a*diff-y;
            //similarity = 0.0;
            cliff_marker[ind_mat]=1.0;
		}
	} else {
		if((-a*diff-y<similarity)&&(a*diff+y-th>similarity)){
			//similarity=0.5*diff+0.7;
			similarity=a*diff+y;
            //similarity = 0.0;
            cliff_marker[ind_mat]=1.0;
		} else if (-a*diff-y-th>similarity){
			//similarity=-0.5*diff-1.3;
			similarity=-a*diff-y;
            //similarity = 0.0;
            cliff_marker[ind_mat]=1.0;
		}
	}
    */

    // cliff marker 
    /*
    if(cliff_marker[ind_mat]!=0.0) {
        similarity=0.0;
    }
    */
	
	// last
    //temp[threadIdx.x] = similarity;

	// const scale test
    //temp[threadIdx.x] = similarity/blockDim.x;
    //temp[threadIdx.x] = similarity*0.5;
    //temp[threadIdx.x] = similarity*0.25;
    //temp[threadIdx.x] = similarity*0.125;

    //temp[threadIdx.x] = similarity*ATTENTION_CONST_SCALE;
    temp[threadIdx.x] = similarity*powf(2,(int)ATTENTION_CONST_SCALE);


	//printf("%f, %f : %08x, %08x : %08x, %08x : %f, %f : %f, %f\n",in_mat[ind_mat],in_vec[ind_vec],fixed_in_mat,fixed_in_vec,gray_in_mat,gray_in_vec,in_mat[ind_mat]-in_vec[ind_vec],in_mat[ind_mat]*in_vec[ind_vec],similarity,similarity_gray);
	//printf("%f, %f : %08x, %08x : %f, %f : %f\n",in_mat[ind_mat],in_vec[ind_vec],fixed_in_mat,fixed_in_vec,in_mat[ind_mat]-in_vec[ind_vec],in_mat[ind_mat]*in_vec[ind_vec],similarity);

    temp[threadIdx.x] = CUDA_FLOAT_QUANT(temp[threadIdx.x],iwl,frac,f_mode);

    __syncthreads();

    if(threadIdx.x==0) {
        float sum = 0;

        for(int i=0;i<blockDim.x;i++) {
            //CUDA_FIXED_ADD(sum,sum,temp[i],iwl,frac);
            sum += temp[i];
        }

        out[blockIdx.x]=CUDA_FLOAT_QUANT(sum,iwl,frac,f_mode);

        /*
        if(out[blockIdx.x]==INFINITY || out[blockIdx.x]==-INFINITY || out[blockIdx.x]!=out[blockIdx.x]) {
            printf("_cuda_approximate_attention: out[%d] %f\n",blockIdx.x, out[blockIdx.x]);
        }
        */
        //printf("sim : %d : %f\n",blockIdx.x,out[blockIdx.x]);
    }
}


// (row x col) = (col_in x row) * (col_in x col)
// <<<row*col,col_in>>>
extern "C"
__global__ void _cuda_mat_trans_mat_product(float *mat_a, float *mat_b, float *out, unsigned int dim_out_row, int dim_out_col, bool fwd_path, float *f_overflow, bool f_fixed, unsigned int iwl, unsigned int frac, unsigned int iwl_out, unsigned int frac_out, unsigned int f_mode) {
    __shared__ float temp[CUDA_MAX_NUM_THREAD];

    unsigned int ind_mat_a;
    unsigned int ind_mat_b;
    unsigned int ind_out_r;
    unsigned int ind_out_c;

    ind_out_r = blockIdx.x / dim_out_col;
    ind_out_c = blockIdx.x % dim_out_col;

    ind_mat_a = ind_out_r + dim_out_row*threadIdx.x;
    ind_mat_b = ind_out_c + dim_out_col*threadIdx.x;

    if(f_fixed) {
        CUDA_FIXED_MUL(temp[threadIdx.x],mat_a[ind_mat_a],mat_b[ind_mat_b],iwl,frac,iwl,frac,f_mode);

        __syncthreads();

        if(threadIdx.x==0) {
            float sum = 0;

            for(int i=0;i<blockDim.x;i++) {
                //CUDA_FIXED_ADD(sum,sum,temp[i],iwl,frac);
                sum += temp[i];
            }
            out[blockIdx.x]=CUDA_FLOAT_QUANT(sum,iwl_out,frac_out,f_mode);
			/*
            if(fwd_path) {
                f_overflow[blockIdx.x]=CUDA_FIXED_OVERFLOW_F(sum,iwl,frac);
            }
			*/
        }
    } else {
        temp[threadIdx.x] = mat_a[ind_mat_a]*mat_b[ind_mat_b];

        __syncthreads();

        if(threadIdx.x==0) {
            float sum = 0;

            for(int i=0;i<blockDim.x;i++) {
                sum += temp[i];
            }
            out[blockIdx.x]=sum;

            /*
            if(out[blockIdx.x]!=out[blockIdx.x]) {
                printf("_cuda_mat_trans_mat_product: out[%d] NaN\n",blockIdx.x);
                printf("<<<%d,%d>>> dim_out_col : %d\n",gridDim.x,blockDim.x,dim_out_col);

                for(int i=0;i<blockDim.x;i++) {
                    printf("temp[%d]: %f\n",i,temp[i]);
                }

                printf("mat_a\n");
                for(int i=0;i<blockDim.x;i++) {
                    printf("%d: %f\n",i,mat_a[ind_out_r + dim_out_row*i]);
                }

                printf("mat_b\n");
                for(int i=0;i<blockDim.x;i++) {
                    printf("%d: %f\n",i,mat_b[ind_out_c + dim_out_col*i]);
                }
            }
            */

            /*
            if(out[blockIdx.x]==INFINITY || out[blockIdx.x]==-INFINITY) {
                printf("_cuda_mat_trans_mat_product: out[%d] inf %f\n",blockIdx.x,out[blockIdx.x]);

                for(int i=0;i<blockDim.x;i++) {
                    printf("temp[%d]: %f\n",i,temp[i]);
                }

                printf("mat_a\n");
                for(int i=0;i<blockDim.x;i++) {
                    printf("%d: %f\n",i,mat_a[ind_out_r + dim_out_row*i]);
                }

                printf("mat_b\n");
                for(int i=0;i<blockDim.x;i++) {
                    printf("%d: %f\n",i,mat_b[ind_out_c + dim_out_col*i]);
                }
            }
            */
        }
    }
}

__global__ void _cuda_mat_trans_mat_product_accum(float *mat_a, float *mat_b, float *out, unsigned int dim_out_row, int dim_out_col, bool f_fixed, unsigned int iwl, unsigned int frac, unsigned int f_mode) {
    __shared__ float temp[CUDA_MAX_NUM_THREAD];

    unsigned int ind_mat_a;
    unsigned int ind_mat_b;
    unsigned int ind_out_r;
    unsigned int ind_out_c;

    ind_out_r = blockIdx.x / dim_out_col;
    ind_out_c = blockIdx.x % dim_out_col;

    ind_mat_a = ind_out_r + dim_out_row*threadIdx.x;
    ind_mat_b = ind_out_c + dim_out_col*threadIdx.x;

    //printf("blk : %d, th : %d, out_r : %d, out_c : %d, mat_a : %d, mat_b : %d\n",blockIdx.x,threadIdx.x,ind_out_r,ind_out_c,ind_mat_a,ind_mat_b);

    //if(f_fixed) {
    if(false) {
        CUDA_FIXED_MUL(temp[threadIdx.x],mat_a[ind_mat_a],mat_b[ind_mat_b],iwl,frac,iwl,frac,f_mode);

        __syncthreads();

        if(threadIdx.x==0) {
            float sum = 0.0;

            for(int i=0;i<blockDim.x;i++) {
                sum += temp[i];
                //CUDA_FIXED_ADD(sum,sum,temp[i],iwl,frac);
            }
            CUDA_FIXED_ADD(out[blockIdx.x],out[blockIdx.x],sum,iwl,frac,iwl,frac,f_mode);    // accum
        }
    } else {
        temp[threadIdx.x] = mat_a[ind_mat_a]*mat_b[ind_mat_b];

        __syncthreads();

        if(threadIdx.x==0) {
            float sum = 0.0;

            for(int i=0;i<blockDim.x;i++) {
                sum += temp[i];
            }
            out[blockIdx.x]+=sum;       // accum

            //printf("%d : %f ",blockIdx.x,out[blockIdx.x]);
            /*
            if(out[blockIdx.x]==INFINITY || out[blockIdx.x]==-INFINITY || out[blockIdx.x]!=out[blockIdx.x]) {
                printf("_cuda_mat_trans_mat_product_accum: out[%d] %f\n",blockIdx.x,out[blockIdx.x]);
            }
            */
        }
    }

}


// (row x col) = (row x col_in) * (col_in x col)
// <<<row*col,col_in>>>
__global__ void _cuda_mat_mat_product(float *mat_a, float *mat_b, float *out, int dim_out_col, bool f_fixed, unsigned int iwl, unsigned int frac, unsigned int iwl_out, unsigned int frac_out, unsigned int f_mode) {
    __shared__ float temp[CUDA_MAX_NUM_THREAD];

    unsigned int ind_mat_a;
    unsigned int ind_mat_b;
    unsigned int ind_out_r;
    unsigned int ind_out_c;

    ind_out_r = blockIdx.x / dim_out_col;
    ind_out_c = blockIdx.x % dim_out_col;

    ind_mat_a = ind_out_r*blockDim.x + threadIdx.x;
    ind_mat_b = ind_out_c + dim_out_col*threadIdx.x;

    if(f_fixed) {
        CUDA_FIXED_MUL(temp[threadIdx.x],mat_a[ind_mat_a],mat_b[ind_mat_b],iwl,frac,iwl,frac,f_mode);

        __syncthreads();

        if(threadIdx.x==0) {
            float sum = 0;

            for(int i=0;i<blockDim.x;i++) {
                sum += temp[i];
                //CUDA_FIXED_ADD(sum,sum,temp[i],iwl,frac);
            }
            out[blockIdx.x]=CUDA_FLOAT_QUANT(sum,iwl_out,frac_out,f_mode);
        }
    } else {
        temp[threadIdx.x] = mat_a[ind_mat_a]*mat_b[ind_mat_b];

        __syncthreads();

        if(threadIdx.x==0) {
            float sum = 0;

            for(int i=0;i<blockDim.x;i++) {
                sum += temp[i];
            }
            out[blockIdx.x]=sum;
        }
    }

}

// approximate attention - back propagation
// <<<row,col>>> (in mat)
__global__ void _cuda_backprop_grad_out_mat(float *in_mat, float *in_vec, float *grad_in_vec, float *grad_out_mat, unsigned int iwl, unsigned int frac, unsigned int f_mode, unsigned int num_bit_attention, unsigned int hop, float *cliff_marker) {

    unsigned int ind_mat;
    unsigned int ind_vec;
    unsigned int ind_grad_in;

    int fixed_in_mat;
    int fixed_in_vec;

	int abs_fixed_in_mat;
	int abs_fixed_in_vec;

	int abs_fixed_min;

    int fixed_in_mat_bit;
    int fixed_in_vec_bit;

    //unsigned int ind_hamming;
    //unsigned int ind_hamming_bit;

    //float hamming_weight;

    float sign_mat;
    float sign_vec;

	int sign_mat_bit;
	int sign_vec_bit;

    //float sign_deriv;

    int i;

    float tmp_a = 0.0;
    float tmp_b = 0.0;

    //bool f_sign_diff = false;


    ind_mat = blockIdx.x*blockDim.x+threadIdx.x;
    ind_vec = threadIdx.x;
    ind_grad_in = blockIdx.x;

    fixed_in_mat = CUDA_FLOAT2FIXED(in_mat[ind_mat],iwl,frac,f_mode);
    fixed_in_vec = CUDA_FLOAT2FIXED(in_vec[ind_vec],iwl,frac,f_mode);

    if(fixed_in_mat>=0.0) {
        sign_mat = 1.0;
		sign_mat_bit = 0x00000000;
    } else {
        sign_mat = -1.0;
		sign_mat_bit = 0x80000000;
    }

    if(fixed_in_vec>=0.0) {
        sign_vec = 1.0;
		sign_vec_bit = 0x00000000;
    } else {
        sign_vec = -1.0;
		sign_vec_bit = 0x80000000;
    }

    /*
    if(sign_mat!=sign_vec) {
        f_sign_diff = true;
    }
    */

    /*
    printf("in_mat : %f %08x %f\n",in_mat[ind_mat],fixed_in_mat,sign_mat);
    printf("in_vec : %f %08x %f\n",in_vec[ind_vec],fixed_in_vec,sign_vec);
    */


	abs_fixed_in_mat = fixed_in_mat&0x7FFFFFFF;
	abs_fixed_in_vec = fixed_in_vec&0x7FFFFFFF;

	if(abs_fixed_in_mat >= abs_fixed_in_vec) {
		abs_fixed_min = abs_fixed_in_vec;
	} else {
		abs_fixed_min = abs_fixed_in_mat;
	}

	if(sign_mat==sign_vec) {
		fixed_in_mat = sign_mat_bit|(abs_fixed_in_mat-abs_fixed_min);
		fixed_in_vec = sign_vec_bit|(abs_fixed_in_vec-abs_fixed_min);
	} else {
		if(abs_fixed_in_mat >= abs_fixed_in_vec) {
			fixed_in_mat = sign_mat_bit|(abs_fixed_in_mat+abs_fixed_min);
			fixed_in_vec = sign_vec_bit|0x00000000;
		} else {
			fixed_in_mat = sign_mat_bit|0x00000000;
			fixed_in_vec = sign_vec_bit|(abs_fixed_in_vec+abs_fixed_min);
		}
	}

    for(i=0;i<num_bit_attention;i++) {
    //for(i=0;i<=iwl;i++) {
        fixed_in_vec_bit = (int)((fixed_in_vec&(0x80000000>>i))>>(31-i)&0x00000001);
        fixed_in_mat_bit = (int)((fixed_in_mat&(0x80000000>>i))>>(31-i)&0x00000001);

        /*
        if((fixed_in_vec_bit-fixed_in_mat_bit < -1) || (fixed_in_vec_bit-fixed_in_mat_bit > 1)) {
            printf("%d\n",fixed_in_vec_bit-fixed_in_mat_bit);
        }
        */

        /*
        if(fixed_in_vec!=fixed_in_vec) {
            printf("_cuda_backprop_grad_out_mat: fixed_in_vec NaN\n");
        }

        if(fixed_in_mat!=fixed_in_mat) {
            printf("_cuda_backprop_grad_out_mat: fixed_in_mat NaN\n");
        }
        */

        //printf("%d: %08x, %08x -> %08x, %08x -> %d, %d\n",31-i,fixed_in_vec,fixed_in_mat,fixed_in_vec_bit,fixed_in_mat_bit,fixed_in_vec_bit,fixed_in_mat_bit);


        //printf("%d - %d = %d\n",fixed_in_mat_bit,fixed_in_vec_bit,(float)(fixed_in_mat_bit-fixed_in_vec_bit));

        //ind_hamming_bit = num_bit_attention-1-i;
        //hamming_weight = powf(2,(int)(ind_hamming_bit-num_bit_attention-1));
        /*
        if((fixed_in_vec_bit!=fixed_in_mat_bit)&&(fixed_in_vec_bit!=0x00000000)) {
            sign_deriv = -1.0;
        } else {
            sign_deriv = 1.0;
        }
        */

        // new_new
        /*
        float similarity;
        if(i==0) {
            _cuda_hamming_similarity_w(fixed_in_mat,fixed_in_vec,num_bit_attention,&similarity);
            //tmp_a += -2.0*sign_mat*similarity;
            tmp_a += -0.02*sign_mat*similarity;
        }
        */

        //if(fixed_in_vec_bit != fixed_in_mat_bit) {
            if(i==0) {
                //f_sign_diff = true;
                // last
                //tmp_a += sign_deriv*hamming_weight/2.0;

                // new - test 170216
                /*
                if(in_mat[ind_mat]==0.0) {
                    tmp_a += (fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat/2.0;
                } else {
                    tmp_a += (fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat/2.0/(in_mat[ind_mat])*0.1;
                }
                */

                //tmp_a += (fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat/4.0/(in_mat[ind_mat]+0.001);
                //tmp_a += (fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat/4.0/(in_mat[ind_mat]+1.5);

                // new_last
                //tmp_a += (fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat/4.0;
                //tmp_a += (fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat/8.0;
                //tmp_a += (fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat/2.0;
                // new_last 170216
                //tmp_a += (fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat/powf(2,(int(2+1.0*HAMMING_WEIGHT_PARA)));

                // last 170222
                //tmp_a += (fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat/powf(2,(int(-1)));
                //tmp_a += (fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat/powf(2,(int(1)));

                // 170306
                //tmp_a += (fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat/powf(2,(int(2)));
                if(fixed_in_mat_bit!=fixed_in_vec_bit) {
                    //tmp_a += (fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat/8.0;
                    //tmp_a += (fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat/4.0;
                    tmp_a += (fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat*powf(2,(int)ATTENTION_CONST_SCALE);
                }

                // test_170306
                //tmp_a += (fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat/powf(2,(int(1)));
                //tmp_a += (fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat/powf(2,(int(3)));
                //tmp_a += (fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat;
                //tmp_a += (fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat/powf(2,(int(HAMMING_WEIGHT_PARA)));
                //tmp_a += 2.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat;
                //tmp_a += 2.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat;
                //tmp_a += (fixed_in_mat_bit-fixed_in_vec_bit)*sign_mat/powf(2,(int)(BW_IWL+1));
                //tmp_a += -1.0*(float)(fixed_in_mat_bit-fixed_in_vec_bit)*sign/4.0/in_vec[ind_vec];

                // test 170222
                /*
                _cuda_hamming_similarity(fixed_in_mat,fixed_in_vec,num_bit_attention,&similarity,true);
                if(in_mat[ind_mat]==0.0) {
                    tmp_a += 0.0;
                    //tmp_a += sign_vec*sign_mat*similarity;
                } else{
                    // original
                    //tmp_a += sign_vec*sign_mat*similarity/in_mat[ind_mat];
                    tmp_a += sign_vec*sign_mat*similarity;
                    // test scale
                    //tmp_a += sign_vec*sign_mat*similarity/in_mat[ind_mat]*0.1;
                    //tmp_a += sign_vec*sign_mat*similarity*0.1;
                    //tmp_a += sign_vec*sign_mat*similarity;
                }
                */

                //printf("b1 : %f\n",(fixed_in_mat_bit-fixed_in_vec_bit)*sign/4.0/in_mat[ind_mat]);
            } else {
                // last
                //tmp_a += sign_deriv*hamming_weight*(in_vec[ind_vec])*(powf(2,(int)(i-BW_IWL)));

                // new_last
                //tmp_a += -1.0*sign_mat/powf(2,(int)(BW_IWL+1))*(fixed_in_mat_bit-fixed_in_vec_bit);

                // new_new
                //tmp_a += -1.0*sign_mat*sign_vec*sign_mat/powf(2,(int)(BW_IWL+1))*(fixed_in_mat_bit-fixed_in_vec_bit);
                /*
                if(sign_mat > 0.0) {
                    tmp_a += -1.0*sign_mat*sign_vec/powf(2,(int)(BW_IWL+1))*(fixed_in_mat_bit-fixed_in_vec_bit);
                } else {
                    tmp_a += 1.0*sign_mat*sign_vec/powf(2,(int)(BW_IWL+1))*(fixed_in_mat_bit-fixed_in_vec_bit);
                }
                */
                //tmp_a += -4.0*sign_vec/powf(2,(int)(BW_IWL+1+1.0*HAMMING_WEIGHT_PARA))*(fixed_in_mat_bit-fixed_in_vec_bit);
                //tmp_a += -2.0*sign_vec/powf(2,(int)(BW_IWL+1+1.0*HAMMING_WEIGHT_PARA))*(fixed_in_mat_bit-fixed_in_vec_bit);

                // test_170306
                //tmp_a += -1.0*sign_vec/powf(2,(int)(iwl+1+1.0*HAMMING_WEIGHT_PARA))*(fixed_in_mat_bit-fixed_in_vec_bit);

                // 170306
                //tmp_a += -2.0*sign_vec/powf(2,(int)(iwl+1+1.0*HAMMING_WEIGHT_PARA))*(fixed_in_mat_bit-fixed_in_vec_bit);
                if(fixed_in_mat_bit!=fixed_in_vec_bit) {
                    //tmp_a += -1.0*sign_vec/powf(2,(int)(iwl+1))*(fixed_in_mat_bit-fixed_in_vec_bit)/2.0;
                    //tmp_a += -1.0*sign_vec/powf(2,(int)(num_bit_attention+iwl+1))*(fixed_in_mat_bit-fixed_in_vec_bit)*ATTENTION_CONST_SCALE;
                    //tmp_a += -1.0*sign_vec*powf(2,(int)(num_bit_attention-iwl-1+ATTENTION_CONST_SCALE-2-iwl-1))*(fixed_in_mat_bit-fixed_in_vec_bit);
                    //tmp_a += -1.0*sign_vec*powf(2,(int)ATTENTION_CONST_SCALE-iwl-1+3)*(fixed_in_mat_bit-fixed_in_vec_bit);
                    tmp_a += -1.0*sign_vec*powf(2,(int)ATTENTION_CONST_SCALE)*(fixed_in_mat_bit-fixed_in_vec_bit);
                    //tmp_a += -1.0*sign_vec*(fixed_in_mat_bit-fixed_in_vec_bit)/num_bit_attention;
                    //tmp_a += -2.0*sign_vec/powf(2,(int)(iwl+1))*(fixed_in_mat_bit-fixed_in_vec_bit);
                }

                // test_170306
                //tmp_a += -4.0*sign_vec/powf(2,(int)(iwl+1+1.0*HAMMING_WEIGHT_PARA))*(fixed_in_mat_bit-fixed_in_vec_bit);

                //tmp_a += -1.0*sign_mat/powf(2,(int)(BW_IWL+1))*(fixed_in_mat_bit-fixed_in_vec_bit);

                //printf("b : %f\n",-1.0*sign/powf(2,(int)(BW_IWL+1))*(fixed_in_mat_bit-fixed_in_vec_bit));
                //printf("sign  : %f\n",sign);
                //printf("powf  : %f\n",powf(2,(int)(BW_IWL+1)));
                //printf("  : %d\n",(fixed_in_mat_bit-fixed_in_vec_bit));
            }
        //}

        //printf("tmp_a mat : %f\n",tmp_a);
    }

    /*
    if(f_sign_diff) {
        tmp_a = 0.0;
    } else {
        tmp_a = tmp_a;
    }
    */

    //tmp_a = in_vec[ind_vec];

    tmp_b = grad_in_vec[ind_grad_in];
    //tmp_b = 1.0;

    //tmp_a = ceil(tmp_a);

    /*
    if(tmp_a==INFINITY || tmp_a==-INFINITY || tmp_a!=tmp_a) {
        printf("_cuda_backprop_grad_out_mat: tmp_a %f\n",tmp_a);
    }

    if(tmp_b==INFINITY || tmp_b==-INFINITY || tmp_b!=tmp_b) {
        printf("_cuda_backprop_grad_out_mat: tmp_b %f\n",tmp_b);
    }
    */

    // last
    grad_out_mat[ind_mat] = tmp_a*tmp_b;

	// test_170424
    /*
	if(in_mat[ind_mat]>=in_vec[ind_vec]) {
    	grad_out_mat[ind_mat] = tmp_b*-0.125*sign_mat*sign_vec*1.0/powf(2,(int)(num_bit_attention-1));
	} else {
    	grad_out_mat[ind_mat] = tmp_b*0.125*sign_mat*sign_vec*1.0/powf(2,(int)(num_bit_attention-1));
	}
    */

	// test_170422
	/*
	if(sign_mat > 0.0 ){
	//if(abs_fixed_in_mat >= abs_fixed_in_vec) {
    	grad_out_mat[ind_mat] = tmp_b;
	} else {
    	grad_out_mat[ind_mat] = tmp_b*-0.1;
	}
	*/

    // test_170316
    //grad_out_mat[ind_mat] = tmp_a*tmp_b*2.0;

    //grad_out_mat[ind_mat] = tmp_a*tmp_b*0.25;

    // test_170306
    //grad_out_mat[ind_mat] = grad_out_mat[ind_mat]*hop;

    // hop test
    //grad_out_mat[ind_mat] = tmp_a*tmp_b*hop;

    // cliff marker test
	/*
    if(cliff_marker[ind_mat]==0.0) {
        grad_out_mat[ind_mat] = tmp_a*tmp_b;
    } else {
        //grad_out_mat[ind_mat] = tmp_a*tmp_b/2.0;
        grad_out_mat[ind_mat] = 0.0;
    }
	*/

    // test
    //grad_out_mat[ind_mat] = grad_out_mat[ind_mat]*2.0;

    //printf("[%d]: tmp_a: %f, grad_in_vec: %f, grad_out_mat: %f\n",ind_grad_in,tmp_a,grad_in_vec[ind_grad_in],grad_out_mat[ind_mat]);


}


// approximate attention - back propagation
// <<<col,row>>> (in mat)
__global__ void _cuda_backprop_grad_out_vec(float *in_mat, float *in_vec, float *grad_in_vec, float *grad_out_vec, unsigned int iwl, unsigned int frac, unsigned int f_mode, unsigned int num_bit_attention, unsigned int hop, float *cliff_marker) {
    __shared__ float temp[CUDA_MAX_NUM_THREAD];

    unsigned int ind_mat;
    unsigned int ind_vec;
    unsigned int ind_grad_in;

    int fixed_in_mat;
    int fixed_in_vec;

	int abs_fixed_in_mat;
	int abs_fixed_in_vec;

	int abs_fixed_min;

    int fixed_in_mat_bit;
    int fixed_in_vec_bit;

    //unsigned int ind_hamming;
    //unsigned int ind_hamming_bit;

    //float hamming_weight;

    float sign_mat;
    float sign_vec;

	int sign_mat_bit;
	int sign_vec_bit;

    //float sign_deriv;

    int i;

	float grad_appx = 0.0;
    float tmp_a = 0.0;
    float tmp_b = 0.0;

    //bool f_sign_diff = false;


    ind_mat = gridDim.x*threadIdx.x+blockIdx.x;
    ind_vec = blockIdx.x;
    ind_grad_in = threadIdx.x;

    fixed_in_mat = CUDA_FLOAT2FIXED(in_mat[ind_mat],iwl,frac,f_mode);
    fixed_in_vec = CUDA_FLOAT2FIXED(in_vec[ind_vec],iwl,frac,f_mode);

    if(fixed_in_mat>=0.0) {
        sign_mat = 1.0;
		sign_mat_bit = 0x00000000;
    } else {
        sign_mat = -1.0;
		sign_mat_bit = 0x80000000;
    }

    if(fixed_in_vec>=0.0) {
        sign_vec = 1.0;
		sign_vec_bit = 0x00000000;
    } else {
        sign_vec = -1.0;
		sign_vec_bit = 0x80000000;
    }

    /*
    if(sign_mat!=sign_vec) {
        f_sign_diff = true;
    }
    */

    /*
    printf("in_mat : %f %08x %f\n",in_mat[ind_mat],fixed_in_mat,sign_mat);
    printf("in_vec : %f %08x %f\n",in_vec[ind_vec],fixed_in_vec,sign_vec);
    */

	abs_fixed_in_mat = fixed_in_mat&0x7FFFFFFF;
	abs_fixed_in_vec = fixed_in_vec&0x7FFFFFFF;

	if(abs_fixed_in_mat >= abs_fixed_in_vec) {
		abs_fixed_min = abs_fixed_in_vec;
	} else {
		abs_fixed_min = abs_fixed_in_mat;
	}

	if(sign_mat==sign_vec) {
		fixed_in_mat = sign_mat_bit|(abs_fixed_in_mat-abs_fixed_min);
		fixed_in_vec = sign_vec_bit|(abs_fixed_in_vec-abs_fixed_min);
	} else {
		if(abs_fixed_in_mat >= abs_fixed_in_vec) {
			fixed_in_mat = sign_mat_bit|(abs_fixed_in_mat+abs_fixed_min);
			fixed_in_vec = sign_vec_bit|0x00000000;
		} else {
			fixed_in_mat = sign_mat_bit|0x00000000;
			fixed_in_vec = sign_vec_bit|(abs_fixed_in_vec+abs_fixed_min);
		}
	}

	/*
    unsigned int ind_mat;
    unsigned int ind_vec;
    unsigned int ind_grad_in;

    int fixed_in_mat;
    int fixed_in_vec;

    int fixed_in_mat_bit;
    int fixed_in_vec_bit;

    unsigned int ind_hamming;
    unsigned int ind_hamming_bit;

    float hamming_weight;

    float sign_mat;
    float sign_vec;

    float sign_deriv;

    int i;

    float tmp_a = 0.0;
    float tmp_b = 0.0;

    bool f_sign_diff=false;

    float similarity;

    ind_mat = gridDim.x*threadIdx.x+blockIdx.x;
    ind_vec = blockIdx.x;
    ind_grad_in = threadIdx.x;

    fixed_in_mat = CUDA_FLOAT2FIXED(in_mat[ind_mat],iwl,frac);
    fixed_in_vec = CUDA_FLOAT2FIXED(in_vec[ind_vec],iwl,frac);


    if(fixed_in_mat>=0.0) {
        sign_mat = 1.0;
    } else {
        sign_mat = -1.0;
    }

    if(fixed_in_vec>=0.0) {
        sign_vec = 1.0;
    } else {
        sign_vec = -1.0;
    }

    if(sign_mat!=sign_vec) {
        f_sign_diff = true;
    }

	*/

    for(i=0;i<num_bit_attention;i++) {
    //for(i=0;i<=iwl;i++) {
        fixed_in_vec_bit = (int)((fixed_in_vec&(0x80000000>>i))>>(31-i)&0x00000001);
        fixed_in_mat_bit = (int)((fixed_in_mat&(0x80000000>>i))>>(31-i)&0x00000001);

        if((fixed_in_vec_bit-fixed_in_mat_bit < -1) || (fixed_in_vec_bit-fixed_in_mat_bit > 1)) {
            printf("%d\n",fixed_in_vec_bit-fixed_in_mat_bit);
        }

        /*
        if(fixed_in_vec!=fixed_in_vec) {
            printf("_cuda_backprop_grad_out_vec: fixed_in_vec NaN\n");
        }

        if(fixed_in_mat!=fixed_in_mat) {
            printf("_cuda_backprop_grad_out_vec: fixed_in_mat NaN\n");
        }
        */

        //ind_hamming_bit = num_bit_attention-1-i;
        //hamming_weight = powf(2,(int)(ind_hamming_bit-num_bit_attention-1));
        /*
        if((fixed_in_vec_bit!=fixed_in_mat_bit)&&(fixed_in_vec_bit!=0x00000000)) {
            sign_deriv = -1.0;
        } else {
            sign_deriv = 1.0;
        }
        */

        // new_new
        /*
        float similarity;
        //if(i==0) {
            //_cuda_hamming_similarity(fixed_in_mat,fixed_in_vec,num_bit_attention,&similarity,true);
            //tmp_a += -2.0*sign_vec*similarity;
            //tmp_a += -0.2*sign_vec*similarity;
            //tmp_a += sign_vec*sign_mat*similarity;
        //}
        */

        //if(fixed_in_vec_bit != fixed_in_mat_bit) {
            if(i==0) {
                //f_sign_diff = true;
                // last
                //tmp_a += sign_deriv*hamming_weight/2.0;

                // new - test 170216
                /*
                if(in_vec[ind_vec]==0.0) {
                    tmp_a += -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec/2.0;
                } else {
                    tmp_a += -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec/2.0/(in_vec[ind_vec])*0.1;
                }
                */

                //tmp_a += -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec/4.0/(in_vec[ind_vec]);
                //tmp_a += -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec/4.0/(in_vec[ind_vec]+0.001);
                //tmp_a += -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec/4.0/(in_vec[ind_vec]+1.5);

                // new_last
                //tmp_a += -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec/4.0;
                //tmp_a += -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec/8.0;
                //tmp_a += -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec/2.0;
                // new_last_170216
                //tmp_a += -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec/(powf(2,int(2+1.0*HAMMING_WEIGHT_PARA)));
                // last 170222
                //tmp_a = -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec/(powf(2,int(-1)));
                //tmp_a = -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec/(powf(2,int(1)));

                // 170306
                //tmp_a = -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec/(powf(2,int(2)));
                if(fixed_in_mat_bit!=fixed_in_vec_bit) {
                    //tmp_a = -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec/8.0;
                    //tmp_a = -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec/4.0;
                    tmp_a = -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec*powf(2,(int)ATTENTION_CONST_SCALE);
                }

                // test_170306
                //tmp_a = -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec/(powf(2,int(1)));
                //tmp_a = -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec/(powf(2,int(3)));
                //tmp_a += -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec;
                //tmp_a += -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec/(powf(2,int(HAMMING_WEIGHT_PARA)));
                //tmp_a += -2.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec;
                //tmp_a += -2.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec;
                //tmp_a += -1.0*(fixed_in_mat_bit-fixed_in_vec_bit)*sign_vec/powf(2,(int)BW_IWL+1);
                //tmp_a += -1.0*(float)(fixed_in_vec_bit-fixed_in_mat_bit)*sign/4.0/in_mat[ind_mat];

                // test 170222
                /*
                _cuda_hamming_similarity(fixed_in_mat,fixed_in_vec,num_bit_attention,&similarity,true);
                if(in_vec[ind_vec]==0.0) {
                    tmp_a = 0.0;
                    //tmp_a = sign_vec*sign_mat*similarity;
                } else{
                    // original
                    //tmp_a = sign_vec*sign_mat*similarity/in_vec[ind_vec];
                    tmp_a = sign_vec*sign_mat*similarity;
                    // test scale
                    //tmp_a = sign_vec*sign_mat*similarity/in_vec[ind_vec]*0.1;
                    //tmp_a = sign_vec*sign_mat*similarity*0.1;
                    //tmp_a = sign_vec*sign_mat*similarity;
                }
                */
            } else {
                // last
                //tmp_a += sign_deriv*hamming_weight*(in_mat[ind_mat])*(powf(2,(int)(i-BW_IWL)));

                // new_last
                //tmp_a += sign_vec/powf(2,(int)(BW_IWL+1))*(fixed_in_mat_bit-fixed_in_vec_bit);
                //tmp_a += sign_vec/powf(2,(int)(BW_IWL+1))*(fixed_in_mat_bit-fixed_in_vec_bit)*blockDim.x;

                // new_new
                //tmp_a += sign_mat*sign_vec*sign_vec/powf(2,(int)(BW_IWL+1))*(fixed_in_mat_bit-fixed_in_vec_bit);
                /*
                if(sign_vec > 0.0) {
                    tmp_a += 1.0*sign_mat*sign_vec/powf(2,(int)(BW_IWL+1))*(fixed_in_mat_bit-fixed_in_vec_bit);
                } else {
                    tmp_a += -1.0*sign_mat*sign_vec/powf(2,(int)(BW_IWL+1))*(fixed_in_mat_bit-fixed_in_vec_bit);
                }
                */
                //tmp_a += 4.0*sign_mat/powf(2,(int)(BW_IWL+1+1.0*HAMMING_WEIGHT_PARA))*(fixed_in_mat_bit-fixed_in_vec_bit);
                //tmp_a += 2.0*sign_mat/powf(2,(int)(BW_IWL+1+1.0*HAMMING_WEIGHT_PARA))*(fixed_in_mat_bit-fixed_in_vec_bit);

                // test_170306
                //tmp_a += 1.0*sign_mat/powf(2,(int)(iwl+1+HAMMING_WEIGHT_PARA))*(fixed_in_mat_bit-fixed_in_vec_bit);

                // 170306
                //tmp_a = 2.0*sign_mat/powf(2,(int)(iwl+1+1.0*HAMMING_WEIGHT_PARA))*(fixed_in_mat_bit-fixed_in_vec_bit);
                //tmp_a = 2.0*sign_mat/powf(2,(int)(iwl+1))*(fixed_in_mat_bit-fixed_in_vec_bit);
                if(fixed_in_mat_bit!=fixed_in_vec_bit) {
                    //tmp_a = 1.0*sign_mat/powf(2,(int)(iwl+1))*(fixed_in_mat_bit-fixed_in_vec_bit)/2.0;
                    //tmp_a = 1.0*sign_mat/powf(2,(int)(num_bit_attention+iwl+1))*(fixed_in_mat_bit-fixed_in_vec_bit)*ATTENTION_CONST_SCALE;
                    //tmp_a = 1.0*sign_mat*powf(2,(int)(num_bit_attention-iwl-1+ATTENTION_CONST_SCALE-2-iwl-1))*(fixed_in_mat_bit-fixed_in_vec_bit);
                    //tmp_a = 1.0*sign_mat*powf(2,(int)ATTENTION_CONST_SCALE-iwl-1+3)*(fixed_in_mat_bit-fixed_in_vec_bit);
                    tmp_a = 1.0*sign_mat*powf(2,(int)ATTENTION_CONST_SCALE)*(fixed_in_mat_bit-fixed_in_vec_bit);
                    //tmp_a = 1.0*sign_mat*(fixed_in_mat_bit-fixed_in_vec_bit)/num_bit_attention;
                    //tmp_a = 2.0*sign_mat/powf(2,(int)(iwl+1))*(fixed_in_mat_bit-fixed_in_vec_bit);
                }

                // test_170306
                //tmp_a = 4.0*sign_mat/powf(2,(int)(iwl+1+1.0*HAMMING_WEIGHT_PARA))*(fixed_in_mat_bit-fixed_in_vec_bit);

                //tmp_a += sign_vec/powf(2,(int)(BW_IWL+1))*(fixed_in_mat_bit-fixed_in_vec_bit);
            }
		grad_appx += tmp_a;
        //printf("%d, %08x, %08x, %d, %d, %f, %f\n",ind_hamming_bit,fixed_in_mat,fixed_in_vec,fixed_in_mat_bit,fixed_in_vec_bit,tmp_a, grad_appx);
    }

    /*
    if(f_sign_diff) {
        tmp_a = 0.0;
    } else {
        tmp_a = tmp_a;
    }
    */

    //tmp_a *= blockDim.x;

    //tmp_a = in_mat[ind_mat];
    tmp_b = grad_in_vec[ind_grad_in];


    //tmp_b = 1.0;
    //tmp_a = ceil(tmp_a);

    /*
    if(tmp_a==INFINITY || tmp_a==-INFINITY || tmp_a!=tmp_a) {
        printf("_cuda_backprop_grad_out_vec: tmp_a %f\n",tmp_a);
    }

    if(tmp_b==INFINITY || tmp_b==-INFINITY || tmp_b!=tmp_b) {
        printf("_cuda_backprop_grad_out_vec: tmp_b %f\n",tmp_b);
    }
    */

    //
    temp[threadIdx.x] = grad_appx*tmp_b;

	// test_170424
    /*
	if(in_mat[ind_mat]>=in_vec[ind_vec]) {
    	temp[threadIdx.x] = tmp_b*0.125*sign_mat*sign_vec*1.0/powf(2,(int)(num_bit_attention-1));
	} else {
    	temp[threadIdx.x] = tmp_b*-0.125*sign_mat*sign_vec*1.0/powf(2,(int)(num_bit_attention-1));
	}
    */

	/*
	// test_170422
	if(sign_vec > 0.0 ){
	//if(abs_fixed_in_mat >= abs_fixed_in_vec) {
    	temp[threadIdx.x] = tmp_b*-1.0;
	} else {
    	temp[threadIdx.x] = tmp_b;
	}
	*/

    // test
    //temp[threadIdx.x] = tmp_a*tmp_b*2.0;

    // cliff marker test
	/*
    if(cliff_marker[ind_mat]==1.0) {
        //temp[threadIdx.x] = temp[threadIdx.x]/2.0;
        temp[threadIdx.x] = 0.0;
    }
	*/

    __syncthreads();

    if(threadIdx.x==0) {
        float sum = 0.0;

        for(int i=0;i<blockDim.x;i++) {
            sum += temp[i];
        }
        // last
        grad_out_vec[blockIdx.x]=sum;

        // test_170316
        //grad_out_vec[blockIdx.x]=sum*2.0;


        // test_170310
        //grad_out_vec[blockIdx.x]=sum*0.25;

        // hop test
        //grad_out_vec[blockIdx.x]=sum*hop;

        //test_170306
        //grad_out_vec[blockIdx.x] = grad_out_vec[blockIdx.x]*hop;
    }

    //printf("grad_out vec : %f\n",grad_out_vec[blockIdx.x]);
}


__global__ void _cuda_mat_mat_product_accum(float *mat_a, float *mat_b, float *out, int dim_out_col, bool f_fixed, unsigned int iwl_m, unsigned int frac_m, unsigned int iwl_v, unsigned int frac_v, unsigned int f_mode) {
    __shared__ float temp[CUDA_MAX_NUM_THREAD];

    unsigned int ind_mat_a;
    unsigned int ind_mat_b;
    unsigned int ind_out_r;
    unsigned int ind_out_c;

    ind_out_r = blockIdx.x / dim_out_col;
    ind_out_c = blockIdx.x % dim_out_col;

    ind_mat_a = ind_out_r*blockDim.x + threadIdx.x;
    ind_mat_b = ind_out_c + dim_out_col*threadIdx.x;

    if(f_fixed) {
        CUDA_FIXED_MUL(temp[threadIdx.x],mat_a[ind_mat_a],mat_b[ind_mat_b],iwl_m,frac_m,iwl_v,frac_v,f_mode);

        __syncthreads();

        if(threadIdx.x==0) {
            float sum = 0;

            for(int i=0;i<blockDim.x;i++) {
                //CUDA_FIXED_ADD(sum,sum,temp[i],iwl,frac);
                sum += temp[i];
            }
            out[blockIdx.x]+=CUDA_FLOAT_QUANT(sum,iwl_m,frac_m,f_mode);
        }
    } else {
        temp[threadIdx.x] = mat_a[ind_mat_a]*mat_b[ind_mat_b];

        __syncthreads();

        if(threadIdx.x==0) {
            float sum = 0;

            for(int i=0;i<blockDim.x;i++) {
                sum += temp[i];
            }
            out[blockIdx.x]+=sum;
        }
    }
}


__global__ void _cuda_mat_init(float *in_mat, float init_value) {
    unsigned int ind = blockIdx.x*blockDim.x + threadIdx.x;
    in_mat[ind] = init_value;

    //in_mat[blockIdx.x] = 0.0;

}

__global__ void _cuda_copy(float *src, float *dest) {
    dest[threadIdx.x] = src[threadIdx.x];

    /*
    if(dest[threadIdx.x]!=dest[threadIdx.x]) {
        printf("_cuda_copy: dest[%d] NaN : src %f\n",threadIdx.x,src[threadIdx.x]);
    }
    */

    /*
    if(dest[threadIdx.x] == INFINITY || dest[threadIdx.x] == -INFINITY) {
        printf("_cuda_copy: exceed max or min : %f\n",dest[threadIdx.x]);
    }
    */
}


__global__ void _cuda_vec_vec_sum(float *in_vec_a, float *in_vec_b, float *out_vec, bool f_fixed, unsigned int iwl, unsigned int frac, unsigned int f_mode) {
    if(f_fixed) {
        CUDA_FIXED_ADD(out_vec[threadIdx.x],in_vec_a[threadIdx.x],in_vec_b[threadIdx.x],iwl,frac,iwl,frac,f_mode);
        //out_vec[threadIdx.x] = in_vec_a[threadIdx.x] + in_vec_b[threadIdx.x];
    } else {
        out_vec[threadIdx.x] = in_vec_a[threadIdx.x] + in_vec_b[threadIdx.x];
    }
}


__global__ void _cuda_vec_vec_weighted_sum_accum(float *in_vec_a, float *in_vec_b, float *out_vec, float weight_a, float weight_b) {
    //printf("a : %f, b : %f\n",weight_a, weight_b);
    unsigned ind = blockIdx.x*blockDim.x + threadIdx.x;
    out_vec[ind] += weight_a*in_vec_a[ind] + weight_b*in_vec_b[ind];
}

__global__ void _cuda_vec_const_mult(float *in_vec_a, float *in_const, float *out_vec) {
    unsigned ind;

    //printf("%f\n",*in_const);

    ind = blockIdx.x*blockDim.x+threadIdx.x;
    out_vec[ind] = in_vec_a[ind] * (*in_const);
}

__global__ void _cuda_vec_vec_mult_accum_scalar(float *in_vec_a, float *in_vec_b, float *out_scalar) {
    __shared__ float tmp[CUDA_MAX_NUM_THREAD];
    unsigned ind;

    ind = blockIdx.x*blockDim.x+threadIdx.x;
    tmp[ind] = in_vec_a[ind] * in_vec_b[ind];

    __syncthreads();

    if(threadIdx.x==0) {
        float sum;
        sum = 0.0;

        for(int i=0;i<blockDim.x;i++) {
            sum += tmp[i];
        }

        *out_scalar = sum;
    }
}

__global__ void _cuda_vec_vec_mult(float *in_vec_a, float *in_vec_b, float *out_vec) {
    unsigned ind;

    ind = blockIdx.x*blockDim.x+threadIdx.x;
    out_vec[ind] = in_vec_a[ind] * in_vec_b[ind];
}

__global__ void _cuda_vec_vec_mult_accum(float *in_vec_a, float *in_vec_b, float *out_vec) {
    unsigned ind;

    ind = blockIdx.x*blockDim.x+threadIdx.x;
    out_vec[ind] += in_vec_a[ind] * in_vec_b[ind];
}


__global__ void _cuda_l2_norm(float *in, float *out) {
    __shared__ float tmp[CUDA_MAX_NUM_THREAD];

    unsigned int ind;

    ind = blockIdx.x*blockDim.x + threadIdx.x;

    tmp[threadIdx.x] = in[ind] * in[ind];

    __syncthreads();

    if(threadIdx.x==0) {
        float sum;
        sum = 0.0;

        for(int i=0;i<blockDim.x;i++) {
            sum += tmp[i];
        }
        //printf("%f -> %f\n",sum,sqrt(sum));
        sum = sqrt(sum);

        atomicAdd(out,sum);

        //printf("%f\n",*out);
    }
    //__syncthreads();
}

__global__ void _cuda_l1_norm(float *in, float *out) {
    __shared__ float tmp[CUDA_MAX_NUM_THREAD];

    unsigned int ind;

    ind = blockIdx.x*blockDim.x + threadIdx.x;

    tmp[threadIdx.x] = in[ind];

    __syncthreads();

    if(threadIdx.x==0) {
        float sum;
        sum = 0.0;

        for(int i=0;i<blockDim.x;i++) {
            sum += tmp[i];
        }
        //printf("%f -> %f\n",sum,sqrt(sum));

        atomicAdd(out,sum);

        //printf("%f\n",*out);
    }

    //__syncthreads();
}

__global__ void _cuda_div_const(float *dividend, float divisor_const, float *out) {
    __shared__ float tmp[CUDA_MAX_NUM_THREAD];

    unsigned int ind;

    ind = blockIdx.x*blockDim.x + threadIdx.x;

    out[ind] = dividend[ind] / divisor_const;
}



__global__ void _cuda_bypass(float *in, float *out, bool f_fixed, unsigned int iwl, unsigned int frac, unsigned int f_mode) {
    unsigned ind;

    ind = blockIdx.x*blockDim.x+threadIdx.x;

    if(f_fixed) {
        out[ind] = CUDA_FLOAT_QUANT(in[ind],iwl,frac,f_mode);
    } else {
        out[ind] = in[ind];
    }
}

__global__ void _cuda_sigmoid(float *in, float *out, bool f_fixed, unsigned int iwl, unsigned int frac, unsigned int f_mode) {
    unsigned ind;

    ind = blockIdx.x*blockDim.x+threadIdx.x;

    if(f_fixed) {
        out[ind] = CUDA_FLOAT_QUANT(1.0/(1.0+expf(-in[ind])),iwl,frac,f_mode);
    } else {
        out[ind] = 1.0/(1.0+expf(-in[ind]));
    }

}

__global__ void _cuda_relu(float *in, float *out, bool f_fixed, unsigned int iwl, unsigned int frac, unsigned int f_mode) {
    unsigned ind;

    ind = blockIdx.x*blockDim.x+threadIdx.x;

    if(in[ind] > 0.0) {
        out[ind] = in[ind];
    } else {
        out[ind] = 0.0;
    }

    if(f_fixed) {
        out[ind] = CUDA_FLOAT_QUANT(out[ind],iwl,frac,f_mode);
    }
}

__global__ void _cuda_sigmoid_bwd(float *out, float *grad_in, float *grad_out, bool f_fixed, unsigned int iwl, unsigned int frac, unsigned int f_mode) {
    unsigned ind;

    ind = blockIdx.x*blockDim.x+threadIdx.x;

    if(f_fixed) {
        grad_out[ind] = CUDA_FLOAT_QUANT(grad_in[ind]*out[ind]*(1.0-out[ind]),iwl,frac,f_mode);
    } else {
        grad_out[ind] = grad_in[ind]*out[ind]*(1.0-out[ind]);
    }
}

__global__ void _cuda_relu_bwd(float *out, float *grad_in, float *grad_out, bool f_fixed, unsigned int iwl, unsigned int frac, unsigned int f_mode) {
    unsigned ind;

    ind = blockIdx.x*blockDim.x+threadIdx.x;

    if(out[ind] > 0.0) {
        grad_out[ind] = grad_in[ind];
    } else {
        grad_out[ind] = 0.0;
    }

    if(f_fixed) {
        grad_out[ind] = CUDA_FLOAT_QUANT(grad_out[ind],iwl,frac,f_mode);
    }
}

// masking gradient from binary net
__global__ void _cuda_grad_mask_fixed(float *grad_in, float *out, unsigned int iwl, unsigned int frac) {
    unsigned ind = blockIdx.x*blockDim.x+threadIdx.x;

	if((out[ind]>CUDA_FIXED_MAX_FLOAT(iwl,frac))||(out[ind]<CUDA_FIXED_MIN_FLOAT(iwl,frac))) {
    	grad_in[ind] = 0.0;
	}
}

/*
__global__ void _cuda_normalize_vec(float *in, float *out) {
    __shared__ float tmp[CUDA_MAX_NUM_THREAD];

    unsigned int ind;
    float sum;

    ind = blockIdx.x*blockDim.x + threadIdx.x;

    tmp[threadIdx.x] = in[ind] * in[ind];

    __syncthreads();

    if(threadIdx.x==0) {
        sum = 0.0;

        for(int i=0;i<blockDim.x;i++) {
            sum += tmp[i];
        }
        //printf("%f -> %f\n",sum,sqrt(sum));
        sum = sqrt(sum);

        //printf("%f\n",*out);
    }
    __syncthreads();
    
    if(sum == 0.0) {
        out[ind] = in[ind];
    } else {
        out[ind] = in[ind]/sum;
    }
    printf("%f, sum : %f\n",out[ind],sum);


    __syncthreads();

}
*/

////////////////////////////////////////////////////////////

__global__ void _cuda_mat_w_up
(
    float *dev_w_mat_del,
    float *dev_w_mat,
    unsigned int batch_size,
    float lr,
    float lambda,
    float max_grad_l2_norm,
    float *dev_grad_l2_norm,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac,
	unsigned int f_mode
)
{
    unsigned int ind;

    ind = blockIdx.x*blockDim.x + threadIdx.x;

    //printf("lr : %f, batch_size : %d, lambda : %f lr/b : %f, max : %f, grad : %f\n",lr,batch_size,lambda, lr/batch_size,max_grad_l2_norm,*dev_grad_l2_norm);

    if(f_fixed) {
        if((*dev_grad_l2_norm) > max_grad_l2_norm) {
            dev_w_mat[ind] += CUDA_FLOAT_QUANT(lr/batch_size*dev_w_mat_del[ind]*max_grad_l2_norm/(*dev_grad_l2_norm),iwl,frac,f_mode) + CUDA_FLOAT_QUANT(lr*lambda*dev_w_mat[ind],iwl,frac,f_mode);
        } else {
            dev_w_mat[ind] += CUDA_FLOAT_QUANT(lr/batch_size*dev_w_mat_del[ind],iwl,frac,f_mode) + CUDA_FLOAT_QUANT(lr*lambda*dev_w_mat[ind],iwl,frac,f_mode);
        }
        dev_w_mat[ind] = CUDA_FLOAT_QUANT(dev_w_mat[ind],iwl,frac,f_mode);
    } else {
        if((*dev_grad_l2_norm) > max_grad_l2_norm) {
            dev_w_mat[ind] += lr/batch_size*dev_w_mat_del[ind]*max_grad_l2_norm/(*dev_grad_l2_norm) + lr*lambda*dev_w_mat[ind];
        } else {
            dev_w_mat[ind] += lr/batch_size*dev_w_mat_del[ind] + lr*lambda*dev_w_mat[ind];
        }
    }



    /*
    if(dev_w_mat[ind]==INFINITY || dev_w_mat[ind]==-INFINITY || dev_w_mat[ind]!=dev_w_mat[ind]) {
        printf("_cuda_mat_w_up: dev_w_mat[%d] %f : batch_size %d, dev_w_mat_del %f, dev_grad_l2_norm %f\n",ind, dev_w_mat[ind], batch_size, dev_w_mat_del[ind], *dev_grad_l2_norm);
    }
    */

    //dev_w_mat[threadIdx.x] = dev_w_mat[threadIdx.x] + (float)0.3/(float)32.0*dev_w_mat_del[threadIdx.x];
    //dev_w_mat[threadIdx.x] = dev_w_mat[threadIdx.x] + (float)0.3*dev_w_mat_del[threadIdx.x];

    //dev_w_mat_del[ind] = 0.0;

}

__global__ void _cuda_w_up
(
    float *dev_w_del,
    float *dev_w,
    unsigned int batch_size,
    float lr,
    float lambda
    //float max_grad_l2_norm,
    //float *dev_grad_l2_norm,
    //bool f_fixed,
    //unsigned int iwl,
    //unsigned int frac
)
{
    *dev_w += lr/batch_size*(*dev_w_del) + lr*lambda*(*dev_w);
}

__global__ void _cuda_bias_up
(
    float *dev_bias_del,
    float *dev_bias,
    unsigned int batch_size,
    float lr,
    float max_grad_l2_norm,
    float *dev_grad_bias_l2_norm
)
{
    unsigned int ind;

    ind = blockIdx.x*blockDim.x + threadIdx.x;

    dev_bias[ind] += lr/batch_size*dev_bias_del[ind];
}

/*
extern "C"
__device__ void _cuda_max(float *in, float *out, unsigned int dim) {
    __shared__ unsigned int ind_max[CUDA_MAX_NUM_THREAD];

    ind_max[threadIdx.x] = threadIdx.x;

    for(unsigned int ind_step=1;ind_step<dim;ind_step*=2) {
        if((threadIdx.x%(2*(ind_step)) == 0)&&(threadIdx.x+ind_step<dim)) {
            if(in[ind_max[threadIdx.x]] > in[ind_max[threadIdx.x+ind_step]]) {
                ind_max[threadIdx.x] = ind_max[threadIdx.x];
            } else {
                ind_max[threadIdx.x] = ind_max[threadIdx.x+ind_step];
            }
        }
        __syncthreads();
    }

    if(threadIdx.x==dim-1) {
        *out = in[ind_max[0]];
        //printf("max : %f\n",*out);
    }

    __syncthreads();
}
*/

__global__ void _cuda_max(float *in, float *out) {
    __shared__ unsigned int ind_max[CUDA_MAX_NUM_THREAD];

    ind_max[threadIdx.x] = threadIdx.x;

    for(unsigned int ind_step=1;ind_step<blockDim.x;ind_step*=2) {
        if((threadIdx.x%(2*(ind_step)) == 0)&&(threadIdx.x+ind_step<blockDim.x)) {
            if(in[ind_max[threadIdx.x]] > in[ind_max[threadIdx.x+ind_step]]) {
                ind_max[threadIdx.x] = ind_max[threadIdx.x];
            } else {
                ind_max[threadIdx.x] = ind_max[threadIdx.x+ind_step];
            }
        }
        __syncthreads();
    }

    if(threadIdx.x==blockDim.x-1) {
        *out = in[ind_max[0]];
        //printf("max : %f\n",*out);
    }

}

__global__ void _cuda_max_i(float *in, unsigned int *out_idx) {
    __shared__ unsigned int ind_max[CUDA_MAX_NUM_THREAD];

    ind_max[threadIdx.x] = threadIdx.x;

    for(unsigned int ind_step=1;ind_step<blockDim.x;ind_step*=2) {
        if((threadIdx.x%(2*(ind_step)) == 0)&&(threadIdx.x+ind_step<blockDim.x)) {
            if(in[ind_max[threadIdx.x]] > in[ind_max[threadIdx.x+ind_step]]) {
                ind_max[threadIdx.x] = ind_max[threadIdx.x];
            } else {
                ind_max[threadIdx.x] = ind_max[threadIdx.x+ind_step];
            }
        }
        __syncthreads();
    }

    if(threadIdx.x==blockDim.x-1) {
        *out_idx = ind_max[0];
        //printf("max : %f\n",*out);
    }

}



extern "C"
__device__ void _cuda_min(float *in, float *out, unsigned int dim) {
    __shared__ unsigned int ind_min[CUDA_MAX_NUM_THREAD];

    ind_min[threadIdx.x] = threadIdx.x;

    for(unsigned int ind_step=1;ind_step<dim;ind_step*=2) {
        if((threadIdx.x%(2*(ind_step)) == 0)&&(threadIdx.x+ind_step<dim)) {
            if(in[ind_min[threadIdx.x]] < in[ind_min[threadIdx.x+ind_step]]) {
                ind_min[threadIdx.x] = ind_min[threadIdx.x];
            } else {
                ind_min[threadIdx.x] = ind_min[threadIdx.x+ind_step];
            }
        }

        __syncthreads();
    }

    if(threadIdx.x==dim-1) {
        *out = in[ind_min[0]];
    }

    __syncthreads();
}

extern "C"
__global__ void _cuda_softmax_fwd(float *in, float *out, float *max, unsigned int dim, bool f_shift_based) {
    __shared__ double total;

    //__shared__ float max;
    //__shared__ float min;
    //__shared__ float abs_max;

    //_cuda_max(in,&max,dim);
    //_cuda_min(in,&min,dim);
    //abs_max = (fabsf(max) > fabsf(min)) ? fabsf(max) : fabsf(min);

    if(f_shift_based) {
        //out[threadIdx.x] = powf(2,in[threadIdx.x]-*max);

        out[threadIdx.x] = __expf(in[threadIdx.x]-*max);

        //out[threadIdx.x] = __expf(in[threadIdx.x]/abs_max);
        //out[threadIdx.x] = in[threadIdx.x]/abs_max;
        //out[threadIdx.x] = (in[threadIdx.x]-min);
        //out[threadIdx.x] = (in[threadIdx.x]-min)/abs_max;
        //out[threadIdx.x] = __expf(in[threadIdx.x]/abs_max*2.0);
        //out[threadIdx.x] = __expf((in[threadIdx.x]-max)/abs_max/2.0);
        //out[threadIdx.x] = __expf((in[threadIdx.x]-min)/abs_max);
        //out[threadIdx.x] = __expf((in[threadIdx.x]-max)/abs_max);
        //out[threadIdx.x] = __expf(in[threadIdx.x]-*max);

        /*
        if(in[threadIdx.x]>=0) {
            out[threadIdx.x] = in[threadIdx.x];
        } else {
            out[threadIdx.x] = 0.0;
        }
        */
        //printf("%f %f %f %f %f %f\n",in[threadIdx.x],min,max,abs_max,in[threadIdx.x]/abs_max,out[threadIdx.x]);
    } else {
        // exp softmax
        // original
        out[threadIdx.x] = __expf(in[threadIdx.x]-*max);
        // normalization test
        //out[threadIdx.x] = __expf((in[threadIdx.x]-max)/max);

        //out[threadIdx.x] = __expf(in[threadIdx.x]-max+1.0);
        
        // sqaure softmax
        //out[threadIdx.x] = (in[threadIdx.x])*(in[threadIdx.x]);
        
        // linear normalization
        //out[threadIdx.x] = in[threadIdx.x];
        //out[threadIdx.x] = __expf( (float)(in[threadIdx.x]-max+(float)1.0) );
        //out[threadIdx.x] = expf( (float)(in[threadIdx.x]+1.0-max) );
    }

    __syncthreads();

    if(threadIdx.x==0) {
        total = 0.0;
        for(int i=0;i<dim;i++) {
            //printf("%d, %f - %f + 1.0 = %f, %f, %f\n",dim, in[i],max,in[i]-max+1.0, out[i],total);
            total += out[i];
        }
        //printf("%f %f\n",max,total);
    }

    __syncthreads();


	if(f_shift_based) {
    	//out[threadIdx.x] = out[threadIdx.x] / floorf(log2f(total));
    	//out[threadIdx.x] = out[threadIdx.x] / ceilf(log2f(total));
    	out[threadIdx.x] = out[threadIdx.x] / llrintf(log2f(total));
	} else {

		//printf("out : %f , flo : %f\n",out[threadIdx.x],floorf(out[threadIdx.x]));
    	out[threadIdx.x] = out[threadIdx.x] / total;
	}

    /*
    if(out[threadIdx.x]!=out[threadIdx.x]) {
        printf("_cuda_softmax_fwd: out[%d] NaN : in %f, max %f, total %lf\n",threadIdx.x,in[threadIdx.x],max,total);
    }
    */

    /*
    if(out[threadIdx.x]==INFINITY || out[threadIdx.x]==-INFINITY) {
        printf("_cuda_softmax_fwd: out[%d] inf : in %f, max %f, total %lf\n",threadIdx.x,in[threadIdx.x],max,total);
    }
    */

    //printf("%d, %f - %f + 1.0 = %f, %f, %f\n",dim, in[threadIdx.x],max,in[threadIdx.x]-max+1.0, out[threadIdx.x],total);
    //printf("in[%d] : %f, out[%d] : %f, max : %f\n",threadIdx.x, in[threadIdx.x], threadIdx.x, out[threadIdx.x], max);
    //printf("total : %f\n",total);
}

extern "C"
__global__ void _cuda_softmax_bwd(float *out_vec, float *grad_in, float *in_vec, float *grad_out, bool f_shift_based) {
    __shared__ float sum;
    __shared__ float tmp[CUDA_MAX_NUM_THREAD];

    //__shared__ float max;
    //__shared__ float min;
    //__shared__ float abs_max;

    //_cuda_max(in_vec,&max,blockDim.x);
    //_cuda_min(in_vec,&min,blockDim.x);
    //abs_max = (fabsf(max) > fabsf(min)) ? fabsf(max) : fabsf(min);

    if(f_shift_based) {
        // test - linear softmax
        /*
        //tmp[threadIdx.x] = out_vec[threadIdx.x]*grad_in[threadIdx.x];
        tmp[threadIdx.x] = out_vec[threadIdx.x]*(grad_in[threadIdx.x]-min);

        //if(in_vec[threadIdx.x]>=0.0) {
        //    tmp[threadIdx.x] = out_vec[threadIdx.x]*grad_in[threadIdx.x];
        //} else {
        //    tmp[threadIdx.x] = 0.0;
        //}
        
        __syncthreads();
    
        if(threadIdx.x==0) {
            sum = 0.0;
    
            for(int i=0;i<blockDim.x;i++) {
                sum += tmp[i];
            }
        }
        __syncthreads();
    
        __shared__ double total;
    
        if(threadIdx.x==0) {
            total = 0.0;
            for(int i=0;i<blockDim.x;i++) {
                total += in_vec[i];
            }
        }
    
        __syncthreads();
    
        grad_out[threadIdx.x] = 1.0/total*(grad_in[threadIdx.x]-min-sum);
        */

        // original - exp softmax
        tmp[threadIdx.x] = out_vec[threadIdx.x]*grad_in[threadIdx.x];
    
        __syncthreads();
    
        if(threadIdx.x==0) {
            sum = 0.0;
    
            for(int i=0;i<blockDim.x;i++) {
                sum += tmp[i];
            }
        }
    
        __syncthreads();
    
        grad_out[threadIdx.x] = 0.7*out_vec[threadIdx.x]*(grad_in[threadIdx.x]-sum);
        //grad_out[threadIdx.x] = out_vec[threadIdx.x]*(grad_in[threadIdx.x]-sum)/abs_max/2.0;
        //grad_out[threadIdx.x] = out_vec[threadIdx.x]/2.0*(grad_in[threadIdx.x]-sum);
    } else {
        // original - exp softmax
        tmp[threadIdx.x] = out_vec[threadIdx.x]*grad_in[threadIdx.x];
    
        __syncthreads();
    
        if(threadIdx.x==0) {
            sum = 0.0;
    
            for(int i=0;i<blockDim.x;i++) {
                sum += tmp[i];
            }
        }
    
        __syncthreads();
    
        grad_out[threadIdx.x] = out_vec[threadIdx.x]*(grad_in[threadIdx.x]-sum);
    }
    // test - square softmax
    /*
    tmp[threadIdx.x] = in_vec[threadIdx.x]*grad_in[threadIdx.x];

    __syncthreads();

    if(threadIdx.x==0) {
        sum = 0.0;

        for(int i=0;i<blockDim.x;i++) {
            sum += tmp[i];
        }
    }
    __syncthreads();

    __shared__ double total;

    if(threadIdx.x==0) {
        total = 0.0;
        for(int i=0;i<blockDim.x;i++) {
            total += in_vec[i]*in_vec[i];
        }
    }

    __syncthreads();

    grad_out[threadIdx.x] = 2.0*in_vec[threadIdx.x]/total*grad_in[threadIdx.x] -2.0*out_vec[threadIdx.x]/total*sum;
    */
    

    /*
    if(grad_out[threadIdx.x]!=grad_out[threadIdx.x]) {
        printf("_cuda_softmax_bwd: grad_out[%d] NaN : out_vec %f, grad_in %f, sum %f\n",threadIdx.x,out_vec[threadIdx.x],grad_in[threadIdx.x],sum);
    }
    */

    /*
    if(grad_out[threadIdx.x]==INFINITY || grad_out[threadIdx.x]==-INFINITY) {
        printf("_cuda_softmax_bwd: grad_out[%d] inf : %f : out_vec %f, grad_in %f, sum %f\n",threadIdx.x,grad_out[threadIdx.x],out_vec[threadIdx.x],grad_in[threadIdx.x],sum);
    }
    */
}

__global__ void _cuda_cross_entropy_cost(float *dev_h, float *dev_y, float *dev_cost, unsigned int *max_i, unsigned int *m_cnt) {
    //__shared__ float cost[CUDA_MAX_NUM_THREAD];

    //cost[threadIdx.x]=0.0;

    if(dev_y[threadIdx.x]==1.0) {
        //cost[threadIdx.x]=-dev_h[threadIdx.x];
        *dev_cost += -1.0*dev_h[threadIdx.x];

		if(threadIdx.x==*max_i) {
			*m_cnt += 1;
		}
    }


    /*
    __syncthreads();

    if(threadIdx.x==0) {
        for(int i=0;i<blockDim.x;i++) {
            //*dev_cost = cost[i];
            //atomicAdd(dev_cost,cost[i]);
            //printf("%f\n",cost[i]);
        }
    }
    */

}


__global__ void _cuda_cross_entropy_grad(float *dev_h, float *dev_y, float *dev_grad_out) {
    //printf("%d : %f %f %f\n",threadIdx.x, dev_grad_out[threadIdx.x], dev_y[threadIdx.x], dev_h[threadIdx.x]);
    if(dev_y[threadIdx.x]==1.0) {
        //printf("TRUE\n");
        dev_grad_out[threadIdx.x] = 1.0 - dev_h[threadIdx.x];
        //printf("%f %f %f\n",dev_grad_out[threadIdx.x], dev_y[threadIdx.x], dev_h[threadIdx.x]);
    } else {
        //printf("FALSE\n");
        dev_grad_out[threadIdx.x] = - dev_h[threadIdx.x];
        //printf("%f %f %f\n",dev_grad_out[threadIdx.x], dev_y[threadIdx.x], dev_h[threadIdx.x]);
    }

	// dev_y -> one hot representation
	//dev_grad_out[threadIdx.x] = dev_y[threadIdx.x]-dev_h[threadIdx.x];

    /*
    if(dev_grad_out[threadIdx.x]!=dev_grad_out[threadIdx.x]) {
        printf("_cuda_cross_entropy_grad: dev_grad_out[%d] NaN : dev_y %f, dev_h %f\n",threadIdx.x, dev_y[threadIdx.x],dev_h[threadIdx.x]);
    }
    */

    /*
    if(dev_grad_out[threadIdx.x] == INFINITY || dev_grad_out[threadIdx.x] == -INFINITY) {
        printf("_cuda_cross_entropy_grad: dev_grad_out[%d] exceed max of min %f, dev_y %f, dev_h %f\n",threadIdx.x, dev_grad_out[threadIdx.x], dev_y[threadIdx.x], dev_h[threadIdx.x]);
    }
    */

    //printf("%f\n",dev_y[threadIdx.x]);
    //printf("%f\n",dev_h[threadIdx.x]);
    //printf("%f\n",dev_grad_out[threadIdx.x]);
}


extern "C"
//__global__ void _cuda_dense_w_up(float *dev_w_mat, float *dev_w_mat_del, unsigned int batch_size, float lr, float lambda ,float max_grad_l2_norm) {
__global__ void _cuda_dense_w_up(float *dev_w_mat, float *dev_w_mat_del) {
    __shared__ float grad_l2_norm;

    __shared__ float tmp[CUDA_MAX_NUM_THREAD];

    tmp[threadIdx.x] = dev_w_mat_del[threadIdx.x] * dev_w_mat_del[threadIdx.x];

    __syncthreads();

    if(threadIdx.x==0) {
        grad_l2_norm = 0;

        for(int i=0;i<blockDim.x;i++) {
            grad_l2_norm += tmp[i];
        }
        printf("before : %f\n",grad_l2_norm);
        grad_l2_norm = sqrt(grad_l2_norm);
        printf("after : %f\n",grad_l2_norm);
    }
    __syncthreads();

    /*
    if(grad_l2_norm > max_grad_l2_norm) {
        dev_w_mat[threadIdx.x] += lr/batch_size*dev_w_mat_del[threadIdx.x]*max_grad_l2_norm/grad_l2_norm + lr*lambda*dev_w_mat[threadIdx.x];
    } else {
        dev_w_mat[threadIdx.x] += lr/batch_size*dev_w_mat_del[threadIdx.x] + lr*lambda*dev_w_mat[threadIdx.x];
    }
    */

    //dev_w_mat[threadIdx.x] = dev_w_mat[threadIdx.x] + (float)0.3/(float)32.0*dev_w_mat_del[threadIdx.x];
    //dev_w_mat[threadIdx.x] = dev_w_mat[threadIdx.x] + (float)0.3*dev_w_mat_del[threadIdx.x];

    dev_w_mat_del[threadIdx.x] = 0.0;
}



////////////////////////////////////////////////////////////
// dot_mat_vec
////////////////////////////////////////////////////////////
extern "C"
void cuda_dot_mat_vec_constructor
(
    float **dev_out_vec,
    float **dev_grad_out_vec,
    float **dev_grad_out_mat,
    float **dev_f_overflow,
    float **dev_cliff_marker,
    unsigned int dim_mat_r,
    unsigned int dim_mat_c,
    bool f_trans
)
{
    char f_name[] = "cuda_dot_mat_vec_constructor";

#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
#endif

    if(f_trans) {
        CUDA_ERR(f_name, cudaMalloc( (void**) dev_out_vec, dim_mat_c*sizeof(float)));
        CUDA_ERR(f_name, cudaMalloc( (void**) dev_grad_out_vec, dim_mat_r*sizeof(float)));
        CUDA_ERR(f_name, cudaMalloc( (void**) dev_f_overflow, dim_mat_c*sizeof(float)));
    } else {
        CUDA_ERR(f_name, cudaMalloc( (void**) dev_out_vec, dim_mat_r*sizeof(float)));
        CUDA_ERR(f_name, cudaMalloc( (void**) dev_grad_out_vec, dim_mat_c*sizeof(float)));
        CUDA_ERR(f_name, cudaMalloc( (void**) dev_f_overflow, dim_mat_r*sizeof(float)));
    }

    CUDA_ERR(f_name, cudaMalloc( (void**) dev_grad_out_mat, dim_mat_r*dim_mat_c*sizeof(float)));
    CUDA_ERR(f_name, cudaMalloc( (void**) dev_cliff_marker, dim_mat_r*dim_mat_c*sizeof(float)));

#if CUDA_DEBUG
        CUDA_MCHK(f_name);
#endif




    /*
    float *test;
    size_t size = 1024*1;
    CUDA_ERR(f_name, cudaMalloc( (void**) &test, size));
    */

    //CUDA_ERR(f_name, cudaMalloc( (void**) &dev_grad_out_mat, 16000*sizeof(float)));
    //CUDA_ERR(f_name, cudaMalloc( (void**) &dev_grad_out_mat, size));
    //CUDA_ERR(f_name, cudaMalloc( (void**) dev_grad_out_mat, dim_mat_r*dim_mat_c));

    /*
    unsigned int num_block;
    unsigned int num_thread;

    num_block = (dim_mat_r*dim_mat_c)/CUDA_MAX_NUM_THREAD+1;
    num_thread = (dim_mat_r*dim_mat_c)%CUDA_MAX_NUM_THREAD;

    _cuda_mat_init<<<num_block,num_thread>>>(*dev_grad_out_mat,0.0);


    printf("%d %d\n", dim_mat_r, dim_mat_c);

    printf("%p\n",*dev_grad_out_mat);
    */


    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_dot_mat_vec_init
(
    float *dev_out_vec,
    float *dev_grad_out_vec,
    float *dev_grad_out_mat,
    float *dev_f_overflow,
    float *dev_cliff_marker,
    unsigned int dim_mat_r,
    unsigned int dim_mat_c,
    bool f_trans
)
{

    char f_name[] = "cuda_dot_mat_vec_init";
#if CUDA_DEBUG

        printf("\n< %s >\n",f_name);

        printf("dev_out_vec      : %p\n",dev_out_vec);
        printf("dev_grad_out_vec : %p\n",dev_grad_out_vec);
        printf("dev_grad_out_mat : %p\n",dev_grad_out_mat);
        CUDA_MCHK(f_name);
#endif
    if(f_trans) {
        _cuda_mat_init<<<dim_mat_c,1>>>(dev_out_vec,0.0);
        _cuda_mat_init<<<dim_mat_r,1>>>(dev_grad_out_vec,0.0);
        _cuda_mat_init<<<dim_mat_c,1>>>(dev_f_overflow,0.0);
    } else {
        _cuda_mat_init<<<dim_mat_r,1>>>(dev_grad_out_vec,0.0);
        _cuda_mat_init<<<dim_mat_c,1>>>(dev_out_vec,0.0);
        _cuda_mat_init<<<dim_mat_r,1>>>(dev_f_overflow,0.0);
    }

    _cuda_mat_init<<<dim_mat_r,dim_mat_c>>>(dev_grad_out_mat,0.0);
    _cuda_mat_init<<<dim_mat_r,dim_mat_c>>>(dev_cliff_marker,0.0);

    CUDA_ERR(f_name,cudaPeekAtLastError());
}


extern "C"
void cuda_dot_mat_vec_fwd
(
    float *dev_in_mat,
    float *dev_in_vec,
    float *dev_out_vec,
    float *dev_f_overflow,
    unsigned int dim_mat_r,
    unsigned int dim_mat_c,
    bool f_trans,
    bool f_fixed,
    unsigned int iwl_m,
    unsigned int frac_m,
    unsigned int iwl_v,
    unsigned int frac_v,
	unsigned int f_mode,
    bool verbose
)
{

    char f_name[] = "cuda_dot_mat_vec_fwd";

	//

    if(f_trans) {
        _cuda_mat_trans_mat_product<<<dim_mat_c,dim_mat_r>>>(dev_in_vec,dev_in_mat,dev_out_vec,1,dim_mat_c,true,dev_f_overflow,f_fixed,iwl_m,frac_m,iwl_m,frac_m,f_mode);
    } else {
/*
#ifdef EN_COSINE_SIM
        _cuda_normalize_vec<<<1,dim_mat_c>>>(dev_in_vec,dev_in_vec);
        _cuda_normalize_vec<<<dim_mat_r,dim_mat_c>>>(dev_in_mat,dev_in_mat);
#endif
*/
		 _cuda_mat_mat_trans_product<<<dim_mat_r,dim_mat_c>>>(dev_in_mat,dev_in_vec,dev_out_vec,1,true,dev_f_overflow,f_fixed,iwl_m,frac_m,iwl_v,frac_v,iwl_m,frac_m,f_mode);

		/*
		// binarization test - 170405
		float *dev_in_vec_bin;
        CUDA_ERR(f_name, cudaMalloc( (void**) &dev_in_vec_bin, dim_mat_c*sizeof(float)));
    	_cuda_binarization<<<1,dim_mat_c>>>(dev_in_vec, dev_in_vec_bin);
        _cuda_mat_mat_trans_product<<<dim_mat_r,dim_mat_c>>>(dev_in_mat,dev_in_vec_bin,dev_out_vec,1,true,dev_f_overflow,f_fixed,iwl,frac);
    	CUDA_ERR(f_name,cudaFree(dev_in_vec_bin));
		*/
    }

    if(verbose) {
        printf("\n< %s >\n",f_name);
        cudaDeviceSynchronize();
        if(f_trans) {
            printf("f_trans: true\n");
    
            cudaDeviceSynchronize();
            printf("dev_in_mat> dim_mat_r: %d, dim_mat_c: %d\n",dim_mat_r,dim_mat_c);
            _cuda_printf_mat<<<1,1>>>(dev_in_mat,dim_mat_r,dim_mat_c);
    
            cudaDeviceSynchronize();
            printf("dev_in_vec> dim: %d\n",dim_mat_r);
            _cuda_printf_vec<<<1,1>>>(dev_in_vec,dim_mat_r);
    
            cudaDeviceSynchronize();
            printf("dev_out_vec> dim: %d\n",dim_mat_c);
            _cuda_printf_vec<<<1,1>>>(dev_out_vec,dim_mat_c);
    
        } else {
            printf("f_trans: false\n");
    
            cudaDeviceSynchronize();
            printf("dev_in_mat> dim_mat_r: %d, dim_mat_c: %d\n",dim_mat_r,dim_mat_c);
            _cuda_printf_mat<<<1,1>>>(dev_in_mat,dim_mat_r,dim_mat_c);
    
            cudaDeviceSynchronize();
            printf("dev_in_vec> dim: %d\n",dim_mat_c);
            _cuda_printf_vec<<<1,1>>>(dev_in_vec,dim_mat_c);
    
            cudaDeviceSynchronize();
            printf("dev_out_vec> dim: %d\n",dim_mat_r);
            _cuda_printf_vec<<<1,1>>>(dev_out_vec,dim_mat_r);
        }
        cudaDeviceSynchronize();
    }

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

// approximate attention
extern "C"
void cuda_dot_mat_vec_fwd_appx
(
    float *dev_in_mat,
    float *dev_in_vec,
    float *dev_out_vec,
    float *dev_f_overflow,
    float *dev_cliff_marker,
    unsigned int dim_mat_r,
    unsigned int dim_mat_c,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac,
	unsigned int f_mode,
    unsigned int num_bit_attention,
    bool f_trans,
    bool verbose
)
{
    char f_name[] = "cuda_dot_mat_vec_fwd_appx";

    if(f_trans) {
        _cuda_mat_trans_mat_product<<<dim_mat_c,dim_mat_r>>>(dev_in_vec,dev_in_mat,dev_out_vec,1,dim_mat_c,true,dev_f_overflow,f_fixed,iwl,frac,iwl,frac,f_mode);
        //cudaDeviceSynchronize();
    } else {
        _cuda_approximate_attention<<<dim_mat_r,dim_mat_c>>>(dev_in_mat,dev_in_vec,dev_out_vec, iwl, 32-1-iwl, f_mode, num_bit_attention, dev_cliff_marker);
        //cudaDeviceSynchronize();
    }

    if(verbose) {
        printf("\n< %s >\n",f_name);
        cudaDeviceSynchronize();
        if(f_trans) {
            printf("f_trans: true\n");
    
            cudaDeviceSynchronize();
            printf("dev_in_mat> dim_mat_r: %d, dim_mat_c: %d\n",dim_mat_r,dim_mat_c);
            _cuda_printf_mat<<<1,1>>>(dev_in_mat,dim_mat_r,dim_mat_c);
    
            cudaDeviceSynchronize();
            printf("dev_in_vec> dim: %d\n",dim_mat_r);
            _cuda_printf_vec<<<1,1>>>(dev_in_vec,dim_mat_r);
    
            cudaDeviceSynchronize();
            printf("dev_out_vec> dim: %d\n",dim_mat_c);
            _cuda_printf_vec<<<1,1>>>(dev_out_vec,dim_mat_c);
    
        } else {
            printf("f_trans: false\n");
    
            cudaDeviceSynchronize();
            printf("dev_in_mat> dim_mat_r: %d, dim_mat_c: %d\n",dim_mat_r,dim_mat_c);
            _cuda_printf_mat<<<1,1>>>(dev_in_mat,dim_mat_r,dim_mat_c);
    
            cudaDeviceSynchronize();
            printf("dev_in_vec> dim: %d\n",dim_mat_c);
            _cuda_printf_vec<<<1,1>>>(dev_in_vec,dim_mat_c);
    
            cudaDeviceSynchronize();
            printf("dev_out_vec> dim: %d\n",dim_mat_r);
            _cuda_printf_vec<<<1,1>>>(dev_out_vec,dim_mat_r);
        }
        cudaDeviceSynchronize();
    }


    CUDA_ERR(f_name,cudaPeekAtLastError());
}


extern "C"
void cuda_dot_mat_vec_bwd
(
    float *dev_in_mat,
    float *dev_in_vec,
    float *dev_grad_in,
    float *dev_grad_out_mat,
    float *dev_grad_out_vec,
    float *dev_f_overflow,
    unsigned int dim_mat_r,
    unsigned int dim_mat_c,
    bool f_trans,
    bool f_fixed,
    unsigned int iwl_m,
    unsigned int frac_m,
    unsigned int iwl_v,
    unsigned int frac_v,
	unsigned int f_mode,
    bool verbose
)
{

    char f_name[] = "cuda_dot_mat_vec_bwd";

    if(f_trans) {
		/*
        if(f_fixed) {
            _cuda_vec_vec_mult<<<1,dim_mat_c>>>(dev_grad_in,dev_f_overflow,dev_grad_in);
        }
		*/
        // dev_grad_out_mat
        //_cuda_mat_mat_product<<<dim_mat_r*dim_mat_c,1>>>(dev_in_vec,dev_grad_in,dev_grad_out_mat,dim_mat_c,false,iwl_m,frac_m,f_mode);
        _cuda_mat_mat_product<<<dim_mat_r*dim_mat_c,1>>>(dev_in_vec,dev_grad_in,dev_grad_out_mat,dim_mat_c,f_fixed,iwl_m,frac_m,1,iwl_m+frac_m-1,f_mode);

        // dev_grad_out_vec
        //_cuda_mat_mat_trans_product<<<dim_mat_r,dim_mat_c>>>(dev_in_mat,dev_grad_in,dev_grad_out_vec,1,false,NULL,false,iwl_m,frac_m,iwl_m,frac_m,f_mode);
        _cuda_mat_mat_trans_product<<<dim_mat_r,dim_mat_c>>>(dev_in_mat,dev_grad_in,dev_grad_out_vec,1,false,NULL,f_fixed,iwl_m,frac_m,iwl_m,frac_m,1,iwl_m+frac_m-1,f_mode);
    } else {
		/*
        if(f_fixed) {
            _cuda_vec_vec_mult<<<1,dim_mat_r>>>(dev_grad_in,dev_f_overflow,dev_grad_in);
        }
		*/
        // dev_grad_out_mat
        //_cuda_mat_mat_product<<<dim_mat_r*dim_mat_c,1>>>(dev_grad_in,dev_in_vec,dev_grad_out_mat,dim_mat_c,false,iwl_m,frac_m,f_mode);
        _cuda_mat_mat_product<<<dim_mat_r*dim_mat_c,1>>>(dev_grad_in,dev_in_vec,dev_grad_out_mat,dim_mat_c,f_fixed,iwl_m,frac_m,1,iwl_m+frac_m-1,f_mode);

        // dev_grad_out_vec
        //_cuda_mat_trans_mat_product<<<dim_mat_c,dim_mat_r>>>(dev_grad_in,dev_in_mat,dev_grad_out_vec,1,dim_mat_c,false,NULL,false,iwl_m,frac_m,f_mode);
        _cuda_mat_trans_mat_product<<<dim_mat_c,dim_mat_r>>>(dev_grad_in,dev_in_mat,dev_grad_out_vec,1,dim_mat_c,false,NULL,f_fixed,iwl_m,frac_m,1,iwl_m+frac_m-1,f_mode);
    }

    if(verbose) {
        printf("\n< %s >\n",f_name);
        if(f_trans) {
            printf("f_trans: true\n");
    
            cudaDeviceSynchronize();
            printf("dev_in_mat> dim_mat_r: %d, dim_mat_c: %d\n",dim_mat_r,dim_mat_c);
            _cuda_printf_mat<<<1,1>>>(dev_in_mat,dim_mat_r,dim_mat_c);
    
            cudaDeviceSynchronize();
            printf("dev_in_vec> dim: %d\n",dim_mat_r);
            _cuda_printf_vec<<<1,1>>>(dev_in_vec,dim_mat_r);

            cudaDeviceSynchronize();
            printf("dev_grad_in> dim: %d\n",dim_mat_c);
            _cuda_printf_vec<<<1,1>>>(dev_grad_in,dim_mat_c);

            cudaDeviceSynchronize();
            printf("dev_grad_out_vec> dim: %d\n",dim_mat_r);
            _cuda_printf_vec<<<1,1>>>(dev_grad_out_vec,dim_mat_r);

            cudaDeviceSynchronize();
            printf("dev_grad_out_mat> dim_mat_r: %d, dim_mat_c: %d\n",dim_mat_r,dim_mat_c);
            _cuda_printf_mat<<<1,1>>>(dev_grad_out_mat,dim_mat_r,dim_mat_c);
   
        } else {
            printf("f_trans: false\n");
    
            cudaDeviceSynchronize();
            printf("dev_in_mat> dim_mat_r: %d, dim_mat_c: %d\n",dim_mat_r,dim_mat_c);
            _cuda_printf_mat<<<1,1>>>(dev_in_mat,dim_mat_c,dim_mat_r);
    
            cudaDeviceSynchronize();
            printf("dev_in_vec> dim: %d\n",dim_mat_c);
            _cuda_printf_vec<<<1,1>>>(dev_in_vec,dim_mat_c);
    
            cudaDeviceSynchronize();
            printf("dev_grad_in> dim: %d\n",dim_mat_r);
            _cuda_printf_vec<<<1,1>>>(dev_grad_in,dim_mat_r);

            cudaDeviceSynchronize();
            printf("dev_grad_out_vec> dim: %d\n",dim_mat_c);
            _cuda_printf_vec<<<1,1>>>(dev_grad_out_vec,dim_mat_c);

            cudaDeviceSynchronize();
            printf("dev_grad_out_mat> dim_mat_r: %d, dim_mat_c: %d\n",dim_mat_r,dim_mat_c);
            _cuda_printf_mat<<<1,1>>>(dev_grad_out_mat,dim_mat_r,dim_mat_c);
        }
        cudaDeviceSynchronize();
    }

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_dot_mat_vec_bwd_appx
(
    float *dev_in_mat,
    float *dev_in_vec,
    float *dev_grad_in,
    float *dev_grad_out_mat,
    float *dev_grad_out_vec,
    float *dev_f_overflow,
    float *dev_cliff_marker,
    unsigned int dim_mat_r,
    unsigned int dim_mat_c,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac,
	unsigned int f_mode,
    unsigned int num_bit_attention,
    bool f_trans,
    bool verbose,
    unsigned int hop
)
{

    char f_name[] = "cuda_dot_mat_vec_bwd_appx";

    if(f_trans) {
		/*
        if(f_fixed) {
            _cuda_vec_vec_mult<<<1,dim_mat_c>>>(dev_grad_in,dev_f_overflow,dev_grad_in);
        }
		*/
        // dev_grad_out_mat
        //_cuda_mat_mat_product<<<dim_mat_r*dim_mat_c,1>>>(dev_in_vec,dev_grad_in,dev_grad_out_mat,dim_mat_c,false,iwl,frac,f_mode);
        _cuda_mat_mat_product<<<dim_mat_r*dim_mat_c,1>>>(dev_in_vec,dev_grad_in,dev_grad_out_mat,dim_mat_c,f_fixed,iwl,frac,1,iwl+frac-1,f_mode);

        // dev_grad_out_vec
        //_cuda_mat_mat_trans_product<<<dim_mat_r,dim_mat_c>>>(dev_in_mat,dev_grad_in,dev_grad_out_vec,1,false,NULL,false,iwl,frac,iwl,frac,f_mode);
        _cuda_mat_mat_trans_product<<<dim_mat_r,dim_mat_c>>>(dev_in_mat,dev_grad_in,dev_grad_out_vec,1,false,NULL,f_fixed,iwl,frac,iwl,frac,1,iwl+frac-1,f_mode);
    } else {
        // dev_grad_out_mat
        _cuda_backprop_grad_out_mat<<<dim_mat_r,dim_mat_c>>>(dev_in_mat,dev_in_vec,dev_grad_in,dev_grad_out_mat,iwl,32-1-iwl,f_mode,num_bit_attention,hop,dev_cliff_marker);

        // dev_grad_out_vec
        _cuda_backprop_grad_out_vec<<<dim_mat_c,dim_mat_r>>>(dev_in_mat,dev_in_vec,dev_grad_in,dev_grad_out_vec,iwl,32-1-iwl,f_mode,num_bit_attention,hop,dev_cliff_marker);
    }

    if(verbose) {
        printf("\n< %s >\n",f_name);
        if(f_trans) {
            printf("f_trans: true\n");
    
            cudaDeviceSynchronize();
            printf("dev_in_mat> dim_mat_r: %d, dim_mat_c: %d\n",dim_mat_r,dim_mat_c);
            _cuda_printf_mat<<<1,1>>>(dev_in_mat,dim_mat_r,dim_mat_c);
    
            cudaDeviceSynchronize();
            printf("dev_in_vec> dim: %d\n",dim_mat_r);
            _cuda_printf_vec<<<1,1>>>(dev_in_vec,dim_mat_r);

            cudaDeviceSynchronize();
            printf("dev_grad_in> dim: %d\n",dim_mat_c);
            _cuda_printf_vec<<<1,1>>>(dev_grad_in,dim_mat_c);

            cudaDeviceSynchronize();
            printf("dev_grad_out_vec> dim: %d\n",dim_mat_r);
            _cuda_printf_vec<<<1,1>>>(dev_grad_out_vec,dim_mat_r);

            cudaDeviceSynchronize();
            printf("dev_grad_out_mat> dim_mat_r: %d, dim_mat_c: %d\n",dim_mat_r,dim_mat_c);
            _cuda_printf_mat<<<1,1>>>(dev_grad_out_mat,dim_mat_r,dim_mat_c);
   
        } else {
            printf("f_trans: false\n");
    
            cudaDeviceSynchronize();
            printf("dev_in_mat> dim_mat_r: %d, dim_mat_c: %d\n",dim_mat_r,dim_mat_c);
            _cuda_printf_mat<<<1,1>>>(dev_in_mat,dim_mat_c,dim_mat_r);
    
            cudaDeviceSynchronize();
            printf("dev_in_vec> dim: %d\n",dim_mat_c);
            _cuda_printf_vec<<<1,1>>>(dev_in_vec,dim_mat_c);
    
            cudaDeviceSynchronize();
            printf("dev_grad_in> dim: %d\n",dim_mat_r);
            _cuda_printf_vec<<<1,1>>>(dev_grad_in,dim_mat_r);

            cudaDeviceSynchronize();
            printf("dev_grad_out_vec> dim: %d\n",dim_mat_c);
            _cuda_printf_vec<<<1,1>>>(dev_grad_out_vec,dim_mat_c);

            cudaDeviceSynchronize();
            printf("dev_grad_out_mat> dim_mat_r: %d, dim_mat_c: %d\n",dim_mat_r,dim_mat_c);
            _cuda_printf_mat<<<1,1>>>(dev_grad_out_mat,dim_mat_r,dim_mat_c);
        }
        cudaDeviceSynchronize();
    }


    CUDA_ERR(f_name,cudaPeekAtLastError());
}



extern "C"
void cuda_dot_mat_vec_destructor
(
    float *dev_out_vec,
    float *dev_grad_out_vec,
    float *dev_grad_out_mat,
    float *dev_f_overflow,
    float *dev_cliff_marker
)
{
    char f_name[] = "cuda_dot_mat_vec_destructor";

    CUDA_ERR(f_name,cudaFree(dev_out_vec));
    CUDA_ERR(f_name,cudaFree(dev_grad_out_vec));
    CUDA_ERR(f_name,cudaFree(dev_grad_out_mat));
    CUDA_ERR(f_name,cudaFree(dev_f_overflow));
    CUDA_ERR(f_name,cudaFree(dev_cliff_marker));
}

////////////////////////////////////////////////////////////////////////////////
// softmax
////////////////////////////////////////////////////////////////////////////////
extern "C"
void cuda_softmax_constructor
(
    float **dev_out_vec,
    float **dev_grad_out,
	float **dev_max,
    unsigned int dim
)
{
    char f_name[] = "cuda_softmax_constructor";

#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
#endif

    CUDA_ERR(f_name,cudaMalloc( (void**) dev_out_vec, dim*sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_grad_out, dim*sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_max, sizeof(float)));

#if CUDA_DEBUG
        printf("dev_out_vec      : %p\n",*dev_out_vec);
        printf("dev_grad_out     : %p\n",*dev_grad_out);
        CUDA_MCHK(f_name);
#endif

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_softmax_init
(
    float *dev_out_vec,
    float *dev_grad_out,
	float *dev_max,
    unsigned int dim
)
{

    char f_name[] = "cuda_softmax_init";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);

        printf("dev_out_vec      : %p\n",dev_out_vec);
        printf("dev_grad_out     : %p\n",dev_grad_out);
        CUDA_MCHK(f_name);
#endif
    _cuda_mat_init<<<dim,1>>>(dev_out_vec,0.0);
    _cuda_mat_init<<<dim,1>>>(dev_grad_out,0.0);
    _cuda_mat_init<<<1,1>>>(dev_max,0.0);

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_softmax_fwd
(
    float *dev_out_vec,
    float *dev_in_vec,
    float *out_vec,
    float *in_vec,
	float *dev_max,
    unsigned int dim,
    bool f_shift_based,
    bool verbose
)
{
    char f_name[] = "cuda_softmax_fwd";

	//
	_cuda_max<<<1,dim>>>(dev_in_vec,dev_max);

	//
    _cuda_softmax_fwd<<<1,dim>>>(dev_in_vec,dev_out_vec,dev_max,dim,f_shift_based);

    // for the last layer output
    //cudaMemcpy(out_vec, dev_out_vec, dim*sizeof(float), cudaMemcpyDeviceToHost);

    if(verbose) {
        printf("\n< %s >\n",f_name);

        cudaDeviceSynchronize();
        printf("dev_in_vec> dim: %d\n",dim);
        _cuda_printf_vec<<<1,1>>>(dev_in_vec,dim);

        cudaDeviceSynchronize();
        printf("dev_out_vec> dim: %d\n",dim);
        _cuda_printf_vec<<<1,1>>>(dev_out_vec,dim);

        cudaDeviceSynchronize();
    }

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_softmax_bwd
(
    float *dev_grad_in,
    float *dev_out_vec,
    float *dev_grad_out,
    float *dev_in_vec,
    unsigned int dim,
    bool f_shift_based,
    bool verbose
)
{
    char f_name[] = "cuda_softmax_bwd";

    _cuda_softmax_bwd<<<1,dim>>>(dev_out_vec,dev_grad_in,dev_in_vec,dev_grad_out,f_shift_based);

    if(verbose) {
    //if(true) {
        printf("\n< %s >\n",f_name);

        cudaDeviceSynchronize();
        printf("dev_grad_in> dim: %d\n",dim);
        _cuda_printf_vec<<<1,1>>>(dev_grad_in,dim);

        cudaDeviceSynchronize();
        printf("dev_out_vec> dim: %d\n",dim);
        _cuda_printf_vec<<<1,1>>>(dev_out_vec,dim);

        cudaDeviceSynchronize();
        printf("dev_grad_out> dim: %d\n",dim);
        _cuda_printf_vec<<<1,1>>>(dev_grad_out,dim);

        cudaDeviceSynchronize();
    }

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_softmax_destructor
(
    float *dev_out_vec,
    float *dev_grad_out,
	float *dev_max
)
{
    char f_name[] = "cuda_softmax_destructor";

    CUDA_ERR(f_name,cudaFree(dev_out_vec));
    CUDA_ERR(f_name,cudaFree(dev_grad_out));
    CUDA_ERR(f_name,cudaFree(dev_max));
}

////////////////////////////////////////////////////////////////////////////////
// sum - element sum vector (vector + vector)
////////////////////////////////////////////////////////////////////////////////
extern "C"
void cuda_sum_vec_constructor
(
    float **dev_out_vec,
    float **dev_grad_out,
    unsigned int dim
)
{
    char f_name[] = "cuda_sum_vec_constructor";

#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
#endif

    CUDA_ERR(f_name,cudaMalloc( (void**) dev_out_vec, dim*sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_grad_out, dim*sizeof(float)));


#if CUDA_DEBUG
        printf("dev_out_vec      : %p\n",*dev_out_vec);
        printf("dev_grad_out     : %p\n",*dev_grad_out);
        CUDA_MCHK(f_name);
#endif

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_sum_vec_init
(
    float *dev_out_vec,
    float *dev_grad_out,
    unsigned int dim
)
{
    char f_name[] = "cuda_sum_vec_init";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);

        printf("dev_out_vec      : %p\n",dev_out_vec);
        printf("dev_grad_out     : %p\n",dev_grad_out);
        CUDA_MCHK(f_name);
#endif

    _cuda_mat_init<<<dim,1>>>(dev_out_vec,0.0);
    _cuda_mat_init<<<dim,1>>>(dev_grad_out,0.0);

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_sum_vec_fwd
(
    float *dev_in_vec_a,
    float *dev_in_vec_b,
    float *dev_out_vec,
    unsigned int dim,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac,
	unsigned int f_mode,
    bool verbose
)
{
    char f_name[] = "cuda_sum_vec_fwd";
    
    _cuda_vec_vec_sum<<<1,dim>>>(dev_in_vec_a, dev_in_vec_b, dev_out_vec, f_fixed, iwl, frac,f_mode);

    if(verbose) {

        printf("\n< %s >\n",f_name);
    
        cudaDeviceSynchronize();
        printf("dev_in_vec_a> dim: %d\n",dim);
        _cuda_printf_vec<<<1,1>>>(dev_in_vec_a,dim);
    
        cudaDeviceSynchronize();
        printf("dev_in_vec_b> dim: %d\n",dim);
        _cuda_printf_vec<<<1,1>>>(dev_in_vec_b,dim);
    
        cudaDeviceSynchronize();
        printf("dev_out_vec> dim: %d\n",dim);
        _cuda_printf_vec<<<1,1>>>(dev_out_vec,dim);
    
        cudaDeviceSynchronize();
    }

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_sum_vec_bwd
(
    float *dev_grad_out,
    float *dev_grad_in,
    float *grad_in,
    float *grad_out,
    unsigned int dim
)
{
    char f_name[] = "cuda_sum_vec_bwd";

    _cuda_copy<<<1,dim>>>(dev_grad_in,dev_grad_out);

#if CUDA_DEBUG
    printf("\n< %s >\n",f_name);

    cudaDeviceSynchronize();
    printf("dev_grad_in> dim: %d\n",dim);
    _cuda_printf_vec<<<1,1>>>(dev_grad_in,dim);

    cudaDeviceSynchronize();
    printf("dev_grad_out> dim: %d\n",dim);
    _cuda_printf_vec<<<1,1>>>(dev_grad_out,dim);

    cudaDeviceSynchronize();

#endif

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_sum_vec_destructor
(
    float *dev_out_vec,
    float *dev_grad_out
)
{
    char f_name[] = "cuda_sum_vec_destructor";

    CUDA_ERR(f_name,cudaFree(dev_out_vec));
    CUDA_ERR(f_name,cudaFree(dev_grad_out));
}

////////////////////////////////////////////////////////////
// dense
////////////////////////////////////////////////////////////
extern "C"
void cuda_dense_constructor
(
    float **dev_w_mat,
    float **dev_w_mat_del,
    float **dev_w_mat_best,
    float **dev_bias,
    float **dev_bias_del,
    float **dev_out_vec,
    float **dev_grad_out,
    float **dev_grad_l2_norm,
    float **dev_grad_bias_l2_norm,
    float **dev_f_overflow,
    unsigned int dim_in,
    unsigned int dim_out
)
{
    char f_name[] = "cuda_dense_constructor";

#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
#endif
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_w_mat, dim_out*dim_in*sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_w_mat_del, dim_out*dim_in*sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_w_mat_best, dim_out*dim_in*sizeof(float)));

    CUDA_ERR(f_name,cudaMalloc( (void**) dev_bias, dim_out*sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_bias_del, dim_out*sizeof(float)));

    CUDA_ERR(f_name,cudaMalloc( (void**) dev_out_vec, dim_out*sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_grad_out, dim_in*sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_grad_l2_norm, sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_f_overflow, dim_out*sizeof(float)));

#if CUDA_DEBUG
        printf("dev_w_mat        : %p\n",*dev_w_mat);
        printf("dev_w_mat_del    : %p\n",*dev_w_mat_del);
        printf("dev_out_vec      : %p\n",*dev_out_vec);
        printf("dev_grad_out     : %p\n",*dev_grad_out);
        CUDA_MCHK(f_name);
#endif

    CUDA_ERR(f_name,cudaPeekAtLastError());

}

extern "C"
void cuda_dense_init
(
    float *dev_out_vec,
    float *dev_grad_out,
    float *dev_w_mat_del,
    float *dev_w_mat,
    float *dev_bias,
    float *dev_bias_del,
    float *w_mat,
    float *bias,
    float *dev_f_overflow,
    unsigned int dim_in,
    unsigned int dim_out
)
{
    char f_name[] = "cuda_dense_init";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
        printf("dev_w_mat        : %p\n",dev_w_mat);
        printf("dev_w_mat_del    : %p\n",dev_w_mat_del);
        printf("dev_out_vec      : %p\n",dev_out_vec);
        printf("dev_grad_out     : %p\n",dev_grad_out);
        CUDA_MCHK(f_name);
#endif
    _cuda_mat_init<<<dim_out,1>>>(dev_out_vec,0.0);
    _cuda_mat_init<<<dim_in,1>>>(dev_grad_out,0.0);
    _cuda_mat_init<<<dim_out,dim_in>>>(dev_w_mat_del,0.0);
    _cuda_mat_init<<<dim_out,1>>>(dev_bias_del,0.0);
    _cuda_mat_init<<<dim_out,1>>>(dev_f_overflow,0.0);

    cudaMemcpy( dev_w_mat, w_mat, dim_out*dim_in*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_bias, bias, dim_out*sizeof(float), cudaMemcpyHostToDevice );

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_dense_fwd
(
    float *dev_w_mat,
    float *dev_bias,
    float *dev_in_vec,
    float *dev_out_vec,
    float *dev_f_overflow,
    unsigned int dim_in,
    unsigned int dim_out,
    char *activation,
    bool f_fixed,
    unsigned int iwl_in,
    unsigned int frac_in,
    unsigned int iwl_w,
    unsigned int frac_w,
	unsigned int f_mode,
    bool verbose
)
{
    char f_name[] = "cuda_dense_fwd";

    _cuda_mat_vec_product<<<dim_out,dim_in>>>(dev_w_mat,dev_in_vec,dev_out_vec,dev_f_overflow,f_fixed,iwl_w,frac_w,iwl_in,frac_in,f_mode);

    // test_170504_xnor_net_scale

    if((iwl_w+frac_w)==0) {
        float *dev_w_mat_l1_norm_temp;
        CUDA_ERR(f_name,cudaMalloc( (void**) &dev_w_mat_l1_norm_temp, sizeof(float)));
    
        _cuda_l1_norm<<<dim_out,dim_in>>>(dev_w_mat, dev_w_mat_l1_norm_temp);
    
        _cuda_div_const<<<1,1>>>(dev_w_mat_l1_norm_temp,dim_out*dim_in,dev_w_mat_l1_norm_temp);
        //*dev_w_mat_l1_norm_temp=__fdiv_rn(*dev_w_mat_l1_norm_temp,dim_out*dim_in);
    
        _cuda_vec_const_mult<<<1,dim_out>>>(dev_out_vec,dev_w_mat_l1_norm_temp,dev_out_vec);
    
        CUDA_ERR(f_name,cudaFree(dev_w_mat_l1_norm_temp));
    }

    // test_end

    if(!strcmp(activation,"SIGMOID")) {
        _cuda_sigmoid<<<1,dim_out>>>(dev_out_vec,dev_out_vec,f_fixed,iwl_w,frac_w,f_mode);
    } else if(!strcmp(activation,"RELU")) {
		_cuda_relu<<<1,dim_out>>>(dev_out_vec,dev_out_vec,f_fixed,iwl_w,frac_w,f_mode);
    }

    if(verbose) {
    //if(true) {
        printf("\n< %s >\n",f_name);
    
        cudaDeviceSynchronize();
        printf("dev_in_vec> dim: %d\n",dim_in);
        _cuda_printf_vec<<<1,1>>>(dev_in_vec,dim_in);
    
        cudaDeviceSynchronize();
        printf("dev_w_mat> dim_out: %d, dim_in: %d\n",dim_out,dim_in);
        _cuda_printf_mat<<<1,1>>>(dev_w_mat,dim_out,dim_in);
    
        cudaDeviceSynchronize();
        printf("dev_out_vec> dim: %d\n",dim_out);
        _cuda_printf_vec<<<1,1>>>(dev_out_vec,dim_out);
    
        cudaDeviceSynchronize();
    }

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_dense_bwd
(
    float *dev_w_mat,
    float *dev_w_mat_del,
    float *dev_bias,
    float *dev_bias_del,
    float *dev_in_vec,
    float *dev_out_vec,
    float *dev_grad_in,
    float *dev_grad_out,
    float *dev_f_overflow,
    unsigned int dim_in,
    unsigned int dim_out,
    char *activation,
    bool f_fixed,
    unsigned int iwl_in,
    unsigned int frac_in,
    unsigned int iwl_w,
    unsigned int frac_w,
	unsigned int f_mode,
    bool verbose
)
{

    char f_name[] = "cuda_dense_bwd";

    if(!strcmp(activation,"SIGMOID")) {
        _cuda_sigmoid_bwd<<<1,dim_out>>>(dev_out_vec,dev_grad_in,dev_grad_in,f_fixed,iwl_w,frac_w,f_mode);
    } else if(!strcmp(activation,"RELU")) {
        _cuda_relu_bwd<<<1,dim_out>>>(dev_out_vec,dev_grad_in,dev_grad_in,f_fixed,iwl_w,frac_w,f_mode);
    }

	// w_mat_del
    _cuda_mat_mat_product_accum<<<dim_out*dim_in,1>>>(dev_grad_in,dev_in_vec,dev_w_mat_del,dim_in,false,iwl_w,frac_w,iwl_in,frac_in,f_mode);


	// test_170410

	/*
    if(f_fixed) {
		_cuda_grad_mask_fixed<<<1,dim_out>>>(dev_grad_in, dev_out_vec, iwl_in, frac_in);
        //_cuda_vec_vec_mult<<<1,dim_out>>>(dev_grad_in, dev_f_overflow, dev_grad_in);

        //cudaDeviceSynchronize();
        //printf("dev_grad_in> dim: %d\n",dim_out);
        //_cuda_printf_vec<<<1,1>>>(dev_grad_in,dim_out);
        //cudaDeviceSynchronize();
    }
	*/

    // dev_grad_out
    _cuda_mat_trans_mat_product<<<dim_in,dim_out>>>(dev_w_mat,dev_grad_in,dev_grad_out,dim_in,1,false,NULL,false,iwl_w,frac_w,1,iwl_w+frac_w-1,f_mode);


    if(verbose) {
        printf("\n< %s >\n",f_name);
    
        cudaDeviceSynchronize();
        printf("dev_grad_in> dim: %d\n",dim_out);
        _cuda_printf_vec<<<1,1>>>(dev_grad_in,dim_out);
    
        cudaDeviceSynchronize();
        printf("dev_in_vec> dim: %d\n",dim_in);
        _cuda_printf_vec<<<1,1>>>(dev_in_vec,dim_in);
    
        cudaDeviceSynchronize();
        printf("dev_w_mat_del> dim_out: %d, dim_in: %d\n",dim_out,dim_in);
        _cuda_printf_mat<<<1,1>>>(dev_w_mat_del,dim_out,dim_in);
    
        cudaDeviceSynchronize();
        printf("dev_w_mat> dim_out: %d, dim_in: %d\n",dim_out,dim_in);
        _cuda_printf_mat<<<1,1>>>(dev_w_mat,dim_out,dim_in);
    
        cudaDeviceSynchronize();
        printf("dev_grad_out> dim: %d\n",dim_in);
        _cuda_printf_vec<<<1,1>>>(dev_grad_out,dim_in);
    
        cudaDeviceSynchronize();
    }


    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_dense_w_up
(
    float *dev_w_mat,
    float *dev_w_mat_del,
    float *dev_bias,
    float *dev_bias_del,
    float *dev_grad_l2_norm,
    float *dev_grad_bias_l2_norm,
    unsigned int dim_in,
    unsigned int dim_out,
    unsigned int batch_size,
    float *lr,
    float *lambda,
    float *max_grad_l2_norm,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac,
	unsigned int f_mode,
    bool verbose
)
{
    char f_name[] = "cuda_dense_w_up";

    _cuda_mat_init<<<1,1>>>(dev_grad_l2_norm,0.0);

    _cuda_l2_norm<<<dim_out,dim_in>>>(dev_w_mat_del, dev_grad_l2_norm);

    _cuda_mat_w_up<<<dim_out,dim_in>>>(dev_w_mat_del,dev_w_mat,batch_size,*lr,*lambda,*max_grad_l2_norm,dev_grad_l2_norm, f_fixed,iwl,frac,f_mode);

    if(verbose) {
        printf("\n< %s >\n",f_name);

        cudaDeviceSynchronize();
        printf("dev_w_mat_del> dim_out: %d, dim_in: %d\n",dim_out,dim_in);
        _cuda_printf_mat<<<1,1>>>(dev_w_mat_del,dim_out,dim_in);
        cudaDeviceSynchronize();

        printf("dev_w_mat> dim_out: %d, dim_in: %d\n",dim_out,dim_in);
        _cuda_printf_mat<<<1,1>>>(dev_w_mat,dim_out,dim_in);
        cudaDeviceSynchronize();
    }

    _cuda_mat_init<<<dim_out,dim_in>>>(dev_w_mat_del,0.0);

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_dense_destructor
(
    float *dev_w_mat,
    float *dev_w_mat_del,
    float *dev_w_mat_best,
    float *dev_out_vec,
    float *dev_grad_out,
    float *dev_grad_l2_norm,
    float *dev_grad_bias_l2_norm,
    float *dev_f_overflow
)
{
    char f_name[] = "cuda_dense_destructor";

    CUDA_ERR(f_name,cudaFree(dev_w_mat));
    CUDA_ERR(f_name,cudaFree(dev_w_mat_del));
    CUDA_ERR(f_name,cudaFree(dev_w_mat_best));
    CUDA_ERR(f_name,cudaFree(dev_out_vec));
    CUDA_ERR(f_name,cudaFree(dev_grad_out));
    CUDA_ERR(f_name,cudaFree(dev_grad_l2_norm));
    CUDA_ERR(f_name,cudaFree(dev_f_overflow));
}

extern "C"
void cuda_dense_test_dtoh
(
    float *dev_in_vec,
    float *in_vec,
    unsigned int dim_in,
    unsigned int dim_out
)
{
    cudaMemcpy( in_vec, dev_in_vec, dim_in*sizeof(float), cudaMemcpyDeviceToHost);
}

extern "C"
void cuda_dense_test_htod
(
    float *dev_in_vec,
    float *in_vec,
    unsigned int dim_in,
    unsigned int dim_out
)
{
    cudaMemcpy( dev_in_vec, in_vec, dim_in*sizeof(float), cudaMemcpyHostToDevice);
}


////////////////////////////////////////////////////////////
// dense_mat
////////////////////////////////////////////////////////////
extern "C"
void cuda_dense_mat_constructor
(
    float **dev_w_mat,
    float **dev_w_mat_del,
    float **dev_w_mat_best,
    float **dev_bias,
    float **dev_bias_del,
    float **dev_out_mat,
    float **dev_grad_out,
    float **dev_grad_l2_norm,
    float **dev_grad_bias_l2_norm,
    float **dev_f_overflow,
    unsigned int dim_in,
    unsigned int dim_out,
    unsigned int dim_len
)
{
    char f_name[] = "cuda_dense_mat_constructor";

#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
        printf("dim_in: %d, dim_out: %d, dim_len: %d\n",dim_in,dim_out,dim_len);
#endif

    CUDA_ERR(f_name,cudaMalloc( (void**) dev_w_mat, dim_out*dim_in*sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_w_mat_del, dim_out*dim_in*sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_w_mat_best, dim_out*dim_in*sizeof(float)));

    CUDA_ERR(f_name,cudaMalloc( (void**) dev_bias, dim_out*sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_bias_del, dim_out*sizeof(float)));

    CUDA_ERR(f_name,cudaMalloc( (void**) dev_out_mat, dim_len*dim_out*sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_grad_out, dim_len*dim_in*sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_grad_l2_norm, sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_grad_bias_l2_norm, sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_f_overflow, dim_len*dim_out*sizeof(float)));

#if CUDA_DEBUG
        printf("dev_w_mat        : %p\n",*dev_w_mat);
        printf("dev_w_mat_del    : %p\n",*dev_w_mat_del);
        printf("dev_out_mat      : %p\n",*dev_out_mat);
        printf("dev_grad_out     : %p\n",*dev_grad_out);
        printf("dev_dev_f_overflow : %p\n",*dev_f_overflow);
        CUDA_MCHK(f_name);
#endif


    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_dense_mat_init
(
    float *dev_out_mat,
    float *dev_grad_out,
    float *dev_w_mat,
    float *dev_w_mat_del,
    float *dev_bias,
    float *dev_bias_del,
    float *w_mat,
    float *bias,
    float *dev_f_overflow,
    unsigned int dim_in,
    unsigned int dim_out,
    unsigned int dim_len
)
{

    char f_name[] = "cuda_dense_mat_init";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);

        printf("dev_w_mat        : %p\n",dev_w_mat);
        printf("dev_w_mat_del    : %p\n",dev_w_mat_del);
        printf("dev_out_mat      : %p\n",dev_out_mat);
        printf("dev_grad_out     : %p\n",dev_grad_out);
        printf("dev_f_overflow : %p\n",dev_f_overflow);
        CUDA_MCHK(f_name);
#endif
    _cuda_mat_init<<<dim_len,dim_out>>>(dev_out_mat,0.0);
    _cuda_mat_init<<<dim_len,dim_in>>>(dev_grad_out,0.0);
    _cuda_mat_init<<<dim_out,dim_in>>>(dev_w_mat_del,0.0);
    _cuda_mat_init<<<dim_out,dim_in>>>(dev_bias_del,0.0);
    _cuda_mat_init<<<dim_len,dim_out>>>(dev_f_overflow,0.0);



    cudaMemcpy( dev_w_mat, w_mat, dim_out*dim_in*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_bias, bias, dim_out*sizeof(float), cudaMemcpyHostToDevice );

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_dense_mat_fwd
(
    float *dev_w_mat,
    float *dev_bias,
    float *dev_in_mat,
    float *dev_out_mat,
    float *dev_f_overflow,
    unsigned int dim_in,
    unsigned int dim_out,
    unsigned int dim_len,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac,
	unsigned int f_mode,
    bool verbose
)
{
    char f_name[] = "cuda_dense_mat_fwd";

    _cuda_mat_mat_trans_product<<<dim_len*dim_out,dim_in>>>(dev_in_mat,dev_w_mat,dev_out_mat,dim_out,true,dev_f_overflow,f_fixed,iwl,frac,iwl,frac,iwl,frac,f_mode);

    // test_170504_xnor_net_scale

    if((iwl+frac)==0) {
        float *dev_w_mat_l1_norm_temp;
        CUDA_ERR(f_name,cudaMalloc( (void**) &dev_w_mat_l1_norm_temp, sizeof(float)));
    
        _cuda_l1_norm<<<dim_out,dim_in>>>(dev_w_mat, dev_w_mat_l1_norm_temp);
    
        _cuda_div_const<<<1,1>>>(dev_w_mat_l1_norm_temp,dim_out*dim_in,dev_w_mat_l1_norm_temp);
        //*dev_w_mat_l1_norm_temp=__fdiv_rn(*dev_w_mat_l1_norm_temp,dim_out*dim_in);
    
        _cuda_vec_const_mult<<<dim_len,dim_out>>>(dev_out_mat,dev_w_mat_l1_norm_temp,dev_out_mat);
    
        CUDA_ERR(f_name,cudaFree(dev_w_mat_l1_norm_temp));
    }

    // test_end

    if(verbose) {
    //if(true) {
        printf("\n< %s >\n",f_name);
    
        cudaDeviceSynchronize();
        printf("dev_in_mat> dim_len: %d, dim_in: %d\n",dim_len,dim_in);
        _cuda_printf_mat<<<1,1>>>(dev_in_mat,dim_len,dim_in);
        cudaDeviceSynchronize();
        printf("dev_w_mat> dim_out: %d, dim_in: %d\n",dim_out,dim_in);
        _cuda_printf_mat<<<1,1>>>(dev_w_mat,dim_out,dim_in);
        cudaDeviceSynchronize();
        printf("dev_out_mat>dim_len: %d, dim_out: %d\n",dim_len,dim_out);
        _cuda_printf_mat<<<1,1>>>(dev_out_mat,dim_len,dim_out);
        cudaDeviceSynchronize();
    }
    CUDA_ERR(f_name,cudaPeekAtLastError());
}


extern "C"
void cuda_dense_mat_bwd
(
    float *dev_in_mat,
    float *dev_w_mat,
    float *dev_w_mat_del,
    float *dev_bias,
    float *dev_bias_del,
    float *dev_grad_in,
    float *dev_grad_out,
    float *dev_f_overflow,
    unsigned int dim_in,
    unsigned int dim_out,
    unsigned int dim_len,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac,
	unsigned int f_mode,
    bool verbose
)
{
    char f_name[] = "cuda_dense_mat_bwd";

    // w_mat_del
    _cuda_mat_trans_mat_product_accum<<<dim_out*dim_in,dim_len>>>(dev_grad_in,dev_in_mat,dev_w_mat_del,dim_out,dim_in,false,iwl,frac,f_mode);

	/*
    if(f_fixed) {
        _cuda_vec_vec_mult<<<dim_len,dim_out>>>(dev_grad_in,dev_f_overflow,dev_grad_in);
    }
	*/

    // grad out
    _cuda_mat_mat_product<<<dim_len*dim_in,dim_out>>>(dev_grad_in,dev_w_mat,dev_grad_out,dim_in,f_fixed,iwl,frac,1,iwl+frac-1,f_mode);

    if(verbose) {
        printf("\n< %s >\n",f_name);
    }

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_dense_mat_w_up
(
    float *dev_w_mat,
    float *dev_w_mat_del,
    float *dev_bias,
    float *dev_bias_del,
    float *dev_grad_l2_norm,
    float *dev_grad_bias_l2_norm,
    float *w_mat,
    float *w_mat_del,
    unsigned int dim_in,
    unsigned int dim_out,
    unsigned int batch_size,
    float *lr,
    float *lambda,
    float *max_grad_l2_norm,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac,
	unsigned int f_mode,
    bool verbose
)
{
    char f_name[] = "cuda_dense_mat_w_up";

    _cuda_mat_init<<<1,1>>>(dev_grad_l2_norm,0.0);
    _cuda_l2_norm<<<dim_out,dim_in>>>(dev_w_mat_del, dev_grad_l2_norm);

    _cuda_mat_w_up<<<dim_out,dim_in>>>(dev_w_mat_del,dev_w_mat,batch_size,*lr,*lambda,*max_grad_l2_norm,dev_grad_l2_norm,f_fixed,iwl,frac,f_mode);

    _cuda_mat_init<<<dim_out,dim_in>>>(dev_w_mat_del,0.0);

    if(verbose) {
        printf("\n< %s >\n",f_name);
    }

    CUDA_ERR(f_name,cudaPeekAtLastError());
}


extern "C"
void cuda_dense_mat_destructor
(
    float *dev_w_mat,
    float *dev_w_mat_del,
    float *dev_w_mat_best,
    float *dev_out_mat,
    float *dev_grad_out,
    float *dev_grad_l2_norm,
    float *dev_f_overflow
)
{
    char f_name[] = "cuda_dense_mat_destructor";

    CUDA_ERR(f_name,cudaFree(dev_w_mat));
    CUDA_ERR(f_name,cudaFree(dev_w_mat_del));
    CUDA_ERR(f_name,cudaFree(dev_w_mat_best));
    CUDA_ERR(f_name,cudaFree(dev_out_mat));
    CUDA_ERR(f_name,cudaFree(dev_grad_out));
    CUDA_ERR(f_name,cudaFree(dev_grad_l2_norm));
    CUDA_ERR(f_name,cudaFree(dev_f_overflow));
}

////////////////////////////////////////////////////////////////////////////////
// cost function - corss entropy
////////////////////////////////////////////////////////////////////////////////
extern "C"
void cuda_cross_entropy_constructor
(
    float **dev_cost_train,
    float **dev_cost_valid,
    float **dev_cost_test,
    unsigned int **dev_m_cnt_train,
    unsigned int **dev_m_cnt_valid,
    unsigned int **dev_m_cnt_test,
	unsigned int **dev_pred_i,
    float **dev_grad_out,
    unsigned int dim
)
{
    char f_name[] = "cuda_cross_entropy_constructor";

#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
#endif

    CUDA_ERR(f_name,cudaMalloc( (void**) dev_cost_train, sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_cost_valid, sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_cost_test, sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_m_cnt_train, sizeof(unsigned int)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_m_cnt_valid, sizeof(unsigned int)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_m_cnt_test, sizeof(unsigned int)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_pred_i, sizeof(unsigned int)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_grad_out, dim*sizeof(float)));

#if CUDA_DEBUG
        printf("dev_grad_out     : %p\n",*dev_grad_out);
        CUDA_MCHK(f_name);
#endif

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_cross_entropy_init
(
 	float *dev_cost_train,
	float *dev_cost_valid,
	float *dev_cost_test,
	unsigned int *dev_m_cnt_train,
	unsigned int *dev_m_cnt_valid,
	unsigned int *dev_m_cnt_test,
    float *dev_grad_out,
    unsigned int dim
)
{
     char f_name[] = "cuda_cross_entropy_init";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);

        printf("dev_grad_out     : %p\n",dev_grad_out);
        CUDA_MCHK(f_name);
#endif

	_cuda_mat_init<<<1,1>>>(dev_cost_train,0.0);
    _cuda_mat_init<<<1,1>>>(dev_cost_valid,0.0);
    _cuda_mat_init<<<1,1>>>(dev_cost_test,0.0);
    _cuda_mat_init<<<1,1>>>((float*)dev_m_cnt_train,0);
    _cuda_mat_init<<<1,1>>>((float*)dev_m_cnt_valid,0);
    _cuda_mat_init<<<1,1>>>((float*)dev_m_cnt_test,0);

    _cuda_mat_init<<<dim,1>>>(dev_grad_out,0.0);

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_cross_entropy_run
(
    float *dev_cost_train,
    float *dev_cost_valid,
    float *dev_cost_test,
	unsigned int *dev_m_cnt_train,
	unsigned int *dev_m_cnt_valid,
	unsigned int *dev_m_cnt_test,
	unsigned int *dev_pred_i,
    float *cost,
    float *dev_h,
    float *dev_y,
    float *h,
    float *y,
    float *dev_grad_out,
    float *grad_out,
    unsigned int dim,
	unsigned int mode
)
{
    char f_name[] = "cuda_cross_entropy_run";

	//
	_cuda_max_i<<<1,dim>>>(dev_h,dev_pred_i);


	if(mode==1) {
    	_cuda_cross_entropy_cost<<<1,dim>>>(dev_h, dev_y, dev_cost_train, dev_pred_i, dev_m_cnt_train);
	} else if(mode==2) {
    	_cuda_cross_entropy_cost<<<1,dim>>>(dev_h, dev_y, dev_cost_valid, dev_pred_i, dev_m_cnt_valid);
	} else if(mode==3) {
    	_cuda_cross_entropy_cost<<<1,dim>>>(dev_h, dev_y, dev_cost_test, dev_pred_i, dev_m_cnt_test);
	}

    _cuda_cross_entropy_grad<<<1,dim>>>(dev_h, dev_y, dev_grad_out);

    //cudaMemcpy(cost, dev_cost, sizeof(float), cudaMemcpyDeviceToHost );



#if CUDA_DEBUG
    printf("\n< %s >\n",f_name);

    cudaDeviceSynchronize();
    printf("dev_h> dim: %d\n",dim);
    _cuda_printf_vec<<<1,1>>>(dev_h,dim);

    cudaDeviceSynchronize();
    printf("dev_y> dim: %d\n",dim);
    _cuda_printf_vec<<<1,1>>>(dev_y,dim);

    cudaDeviceSynchronize();
    printf("dev_grad_out> dim: %d\n",dim);
    _cuda_printf_vec<<<1,1>>>(dev_grad_out,dim);

    cudaDeviceSynchronize();
#endif


    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_cross_entropy_cost_load
(
	float *dev_cost_train,
    float *dev_cost_valid,
    float *dev_cost_test,
	float *cost_train,
	float *cost_valid,
	float *cost_test
)
{
    cudaMemcpy(cost_train, dev_cost_train, sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy(cost_valid, dev_cost_valid, sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy(cost_test, dev_cost_test, sizeof(float), cudaMemcpyDeviceToHost );

    _cuda_mat_init<<<1,1>>>(dev_cost_train,0.0);
    _cuda_mat_init<<<1,1>>>(dev_cost_valid,0.0);
    _cuda_mat_init<<<1,1>>>(dev_cost_test,0.0);
}

extern "C"
void cuda_cross_entropy_m_cnt_load
(
	unsigned int *dev_m_cnt_train,
	unsigned int *dev_m_cnt_valid,
	unsigned int *dev_m_cnt_test,
	unsigned int *m_cnt_train,
	unsigned int *m_cnt_valid,
	unsigned int *m_cnt_test
)
{
    cudaMemcpy(m_cnt_train, dev_m_cnt_train, sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy(m_cnt_valid, dev_m_cnt_valid, sizeof(float), cudaMemcpyDeviceToHost );
    cudaMemcpy(m_cnt_test, dev_m_cnt_test, sizeof(float), cudaMemcpyDeviceToHost );

    _cuda_mat_init<<<1,1>>>((float*)dev_m_cnt_train,0);
    _cuda_mat_init<<<1,1>>>((float*)dev_m_cnt_valid,0);
    _cuda_mat_init<<<1,1>>>((float*)dev_m_cnt_test,0);

}

extern "C"
void cuda_cross_entropy_destructor
(
 	float *dev_cost_train,
    float *dev_cost_valid,
    float *dev_cost_test,
	float *dev_m_cnt_train,
	float *dev_m_cnt_valid,
	float *dev_m_cnt_test,
	float *dev_pred_i,
    float *dev_grad_out
)
{
    char f_name[] = "cuda_cross_entropy_destructor";

    CUDA_ERR(f_name,cudaFree(dev_cost_train));
    CUDA_ERR(f_name,cudaFree(dev_cost_valid));
    CUDA_ERR(f_name,cudaFree(dev_cost_test));
    CUDA_ERR(f_name,cudaFree(dev_m_cnt_train));
    CUDA_ERR(f_name,cudaFree(dev_m_cnt_valid));
    CUDA_ERR(f_name,cudaFree(dev_m_cnt_test));
    CUDA_ERR(f_name,cudaFree(dev_pred_i));
    CUDA_ERR(f_name,cudaFree(dev_grad_out));
}




////////////////////////////////////////////////////////////////////////////////
// etc
////////////////////////////////////////////////////////////////////////////////
extern "C"
void cuda_dup_grad_constructor
(
    float **dev_dup_grad,
    unsigned int num_hop,
    unsigned int dim
)
{
    char f_name[] = "cuda_dup_grad_constructor";

#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
#endif

    CUDA_ERR(f_name,cudaMalloc( (void**) dev_dup_grad, num_hop*dim*sizeof(float)));

#if CUDA_DEBUG
        printf("dev_dup_grad     : %p\n",*dev_dup_grad);
        CUDA_MCHK(f_name);
#endif

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_dup_grad_bwd
(
    float *dev_dup_grad,
    float *dotmv_dev_grad_out_vec,
    float *sv_dev_grad_out_vec,
    float *dup_grad,
    unsigned int dim,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac,
	unsigned int f_mode
)
{
    char f_name[] = "cuda_dup_grad_bwd";

    _cuda_vec_vec_sum<<<1,dim>>>(dotmv_dev_grad_out_vec,sv_dev_grad_out_vec,dev_dup_grad,f_fixed,1,iwl+frac-1,f_mode);


#if CUDA_DEBUG
    printf("\n< %s >\n",f_name);

    cudaDeviceSynchronize();
    printf("dotmv_dev_grad_out_vec> dim: %d\n",dim);
    _cuda_printf_vec<<<1,1>>>(dotmv_dev_grad_out_vec,dim);

    cudaDeviceSynchronize();
    printf("sv_dev_grad_out_vec> dim: %d\n",dim);
    _cuda_printf_vec<<<1,1>>>(sv_dev_grad_out_vec,dim);

    cudaDeviceSynchronize();
    printf("dev_dup_grad> dim: %d\n",dim);
    _cuda_printf_vec<<<1,1>>>(dev_dup_grad,dim);

    cudaDeviceSynchronize();
#endif

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_dup_grad_destructor
(
    float *dev_dup_grad
)
{
    char f_name[] = "cuda_dup_grad_desturctor";

    CUDA_ERR(f_name,cudaFree(dev_dup_grad));
}

extern "C"
void cuda_data_constructor
(
    float **dev_m,
    float **dev_q,
    float **dev_a,
    unsigned int dim_len,
    unsigned int dim_in,
	unsigned int num_sample
)
{
    char f_name[] = "cuda_data_constructor";

#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
#endif

    CUDA_ERR(f_name,cudaMalloc( (void**) dev_m, dim_len*dim_in*sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_q, num_sample*dim_in*sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_a, num_sample*dim_in*sizeof(float)));

#if CUDA_DEBUG
        printf("dev_m            : %p\n",*dev_m);
        printf("dev_q            : %p\n",*dev_q);
        printf("dev_a            : %p\n",*dev_a);
        CUDA_MCHK(f_name);
#endif

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_data_in
(
    float *dev_m,
    float *dev_q,
    float *dev_a,
    float *m,
    float *q,
    float *a,
    unsigned int dim_len,
    unsigned int dim_in,
	unsigned int num_sample
)
{
    char f_name[] = "cuda_data_in";

#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);

        printf("dev_m            : %p\n",dev_m);
        printf("dev_q            : %p\n",dev_q);
        printf("dev_a            : %p\n",dev_a);
        CUDA_MCHK(f_name);
#endif

    CUDA_ERR(f_name,cudaMemcpy( dev_m, m, dim_len*dim_in*sizeof(float), cudaMemcpyHostToDevice ));
    CUDA_ERR(f_name,cudaMemcpy( dev_q, q, num_sample*dim_in*sizeof(float), cudaMemcpyHostToDevice ));
    CUDA_ERR(f_name,cudaMemcpy( dev_a, a, num_sample*dim_in*sizeof(float), cudaMemcpyHostToDevice ));

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_data_destructor
(
    float *dev_m,
    float *dev_q,
    float *dev_a
)
{
    char f_name[] = "cuda_data_destructor";

    CUDA_ERR(f_name,cudaFree(dev_m));
    CUDA_ERR(f_name,cudaFree(dev_q));
    CUDA_ERR(f_name,cudaFree(dev_a));
}

__global__ void _cuda_copy_mat(float *src, float *dest, unsigned int col, unsigned int row, bool f_trans) {
    unsigned int ind_x;
    unsigned int ind_y;

    unsigned int ind_x_dest;
    unsigned int ind_y_dest;

    unsigned int ind_src;
    unsigned int ind_dest;

	/*
    ind_x = (threadIdx.x)%col;
    ind_y = (threadIdx.x+blockIdx.x*col)/col;

    if(f_trans) {
        ind_x_dest = ind_y;
        ind_y_dest = ind_x;
    } else {
        ind_x_dest = ind_x;
        ind_y_dest = ind_y;
    }

   	ind_dest = ind_x_dest+ind_y_dest*row;
   	ind_src = ind_x+ind_y*col;

    dest[ind_dest] = src[ind_src];
	*/

    ind_x = threadIdx.x;
    ind_y = blockIdx.x;

    if(f_trans) {
        ind_x_dest = ind_y;
        ind_y_dest = ind_x;

    	ind_dest = ind_x_dest+ind_y_dest*row;
    	ind_src = ind_x+ind_y*col;
    } else {
        ind_x_dest = ind_x;
        ind_y_dest = ind_y;

    	ind_dest = ind_x_dest+ind_y_dest*col;
    	ind_src = ind_x+ind_y*col;
    }

    dest[ind_dest] = src[ind_src];
}

__global__ void _cuda_accum_mat(float *src, float *dest, unsigned int col, unsigned int row, bool f_trans) {
    unsigned int ind_x;
    unsigned int ind_y;

    unsigned int ind_x_dest;
    unsigned int ind_y_dest;

    unsigned int ind_src;
    unsigned int ind_dest;

    ind_x = threadIdx.x;
    ind_y = blockIdx.x;

    if(f_trans) {
        ind_x_dest = ind_y;
        ind_y_dest = ind_x;

    	ind_dest = ind_x_dest+ind_y_dest*row;
    	ind_src = ind_x+ind_y*col;
    } else {
        ind_x_dest = ind_x;
        ind_y_dest = ind_y;

    	ind_dest = ind_x_dest+ind_y_dest*col;
    	ind_src = ind_x+ind_y*col;
    }

    dest[ind_dest] += src[ind_src];
}


extern "C"
void cuda_copy_mat
(
    float *dev_src,
    float *dev_dest,
    unsigned int dim_col,
    unsigned int dim_row,
    bool f_trans
)
{
    char f_name[] = "cuda_copy_mat";

#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
#endif
    _cuda_copy_mat<<<dim_row,dim_col>>>(dev_src,dev_dest,dim_col,dim_row,f_trans);

	/*
    cudaDeviceSynchronize();
	printf("src\n");
    _cuda_printf_mat<<<1,1>>>(dev_src,dim_row,dim_col);

    cudaDeviceSynchronize();
	printf("dest\n");
	if(f_trans) {
    	_cuda_printf_mat<<<1,1>>>(dev_dest,dim_col,dim_row);
	} else {
    	_cuda_printf_mat<<<1,1>>>(dev_dest,dim_row,dim_col);
	}
	*/

    //cudaDeviceSynchronize();

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_accum_mat
(
	float *dev_src,
	float *dev_dest,
	unsigned int dim_col,
	unsigned int dim_row,
	bool f_trans
)
{ 
	char f_name[] = "cuda_accum_mat";

#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
#endif
    _cuda_accum_mat<<<dim_row,dim_col>>>(dev_src,dev_dest,dim_col,dim_row,f_trans);

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

////////////////////////////////////////////////////////////////////////////////
// mult_e_vec - element-wise multiply vector (not used)
////////////////////////////////////////////////////////////////////////////////
extern "C"
void cuda_mult_e_vec_constructor
(
    float **dev_out_vec,
    float **dev_grad_out_a,
    float **dev_grad_out_b,
    unsigned int dim
)
{
    char f_name[] = "cuda_mult_e_vec_constructor";

#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
#endif

    CUDA_ERR(f_name,cudaMalloc( (void**) dev_out_vec, dim*sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_grad_out_a, dim*sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_grad_out_b, dim*sizeof(float)));

#if CUDA_DEBUG
        printf("dev_out_vec      : %p\n",*dev_out_vec);
        printf("dev_grad_out_a   : %p\n",*dev_grad_out_a);
        printf("dev_grad_out_b   : %p\n",*dev_grad_out_b);
        CUDA_MCHK(f_name);
#endif

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_mult_e_vec_init
(
    float *dev_out_vec,
    float *dev_grad_out_a,
    float *dev_grad_out_b,
    unsigned int dim
)
{
    char f_name[] = "cuda_mult_e_vec_init";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);

        printf("dev_out_vec      : %p\n",dev_out_vec);
        printf("dev_grad_out_a   : %p\n",*dev_grad_out_a);
        printf("dev_grad_out_b   : %p\n",*dev_grad_out_b);
        CUDA_MCHK(f_name);
#endif

    _cuda_mat_init<<<dim,1>>>(dev_out_vec,0.0);
    //cudaDeviceSynchronize();
    _cuda_mat_init<<<dim,1>>>(dev_grad_out_a,0.0);
    //cudaDeviceSynchronize();
    _cuda_mat_init<<<dim,1>>>(dev_grad_out_b,0.0);
    //cudaDeviceSynchronize();

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_mult_e_vec_fwd
(
    float *dev_in_vec_a,
    float *dev_in_vec_b,
    float *dev_out_vec,
    float *in_vec_a,
    float *in_vec_b,
    float *out_vec,
    unsigned int dim
)
{
    char f_name[] = "cuda_mult_e_vec_fwd";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
        printf("dev_out_vec      : %p\n",dev_out_vec);
#endif

    // modify_point
    //cudaMemcpy( dev_in_vec_a, in_vec_a, dim*sizeof(float), cudaMemcpyHostToDevice );
    //cudaMemcpy( dev_in_vec_b, in_vec_b, dim*sizeof(float), cudaMemcpyHostToDevice );

    _cuda_vec_vec_mult<<<1,dim>>>(dev_in_vec_a, dev_in_vec_b, dev_out_vec);
    //cudaDeviceSynchronize();

    // modify_point
    //cudaMemcpy( out_vec, dev_out_vec, dim*sizeof(float), cudaMemcpyDeviceToHost );
    //cudaMemcpy( dev_out_vec, out_vec, dim*sizeof(float), cudaMemcpyHostToDevice);

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_mult_e_vec_bwd
(
    float *dev_in_vec_a,
    float *dev_in_vec_b,
    float *dev_grad_out_a,
    float *dev_grad_out_b,
    float *dev_grad_in,
    float *grad_in,
    float *grad_out_a,
    float *grad_out_b,
    unsigned int dim
)
{
    char f_name[] = "cuda_mult_e_vec_bwd";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);

        printf("dev_grad_out_a   : %p\n",*dev_grad_out_a);
        printf("dev_grad_out_b   : %p\n",*dev_grad_out_b);
#endif
    // modify_point
    //cudaMemcpy( dev_grad_in, grad_in, dim*sizeof(float), cudaMemcpyHostToDevice );

    _cuda_vec_vec_mult<<<1,dim>>>(dev_grad_in,dev_in_vec_b,dev_grad_out_a);
    //cudaDeviceSynchronize();
    _cuda_vec_vec_mult<<<1,dim>>>(dev_grad_in,dev_in_vec_a,dev_grad_out_b);
    //cudaDeviceSynchronize();

    // modify_point
    //cudaMemcpy( grad_out, dev_grad_out, dim*sizeof(float), cudaMemcpyDeviceToHost );
    //cudaMemcpy( dev_grad_out, grad_out, dim*sizeof(float), cudaMemcpyHostToDevice );

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_mult_e_vec_destructor
(
)
{
}


////////////////////////////////////////////////////////////////////////////////
// mult_e_mat - element-wise multiply matrix - not used
////////////////////////////////////////////////////////////////////////////////
extern "C"
void cuda_mult_e_mat_constructor
(
    float **dev_out_mat,
    float **dev_grad_out_a,
    float **dev_grad_out_b,
    unsigned int dim_row,
    unsigned int dim_col
)
{
    char f_name[] = "cuda_mult_e_mat_constructor";

#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
#endif

    CUDA_ERR(f_name,cudaMalloc( (void**) dev_out_mat, dim_row*dim_col*sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_grad_out_a, dim_row*dim_col*sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_grad_out_b, dim_row*dim_col*sizeof(float)));

#if CUDA_DEBUG
        printf("dev_out_mat      : %p\n",*dev_out_mat);
        printf("dev_grad_out_a   : %p\n",*dev_grad_out_a);
        printf("dev_grad_out_b   : %p\n",*dev_grad_out_b);
        CUDA_MCHK(f_name);
#endif

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_mult_e_mat_init
(
    float *dev_out_mat,
    float *dev_grad_out_a,
    float *dev_grad_out_b,
    unsigned int dim_row,
    unsigned int dim_col
)
{
    char f_name[] = "cuda_mult_e_mat_init";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);

        printf("dev_out_mat      : %p\n",dev_out_mat);
        printf("dev_grad_out_a   : %p\n",*dev_grad_out_a);
        printf("dev_grad_out_b   : %p\n",*dev_grad_out_b);
        CUDA_MCHK(f_name);
#endif

    _cuda_mat_init<<<dim_row,dim_col>>>(dev_out_mat,0.0);
    //cudaDeviceSynchronize();
    _cuda_mat_init<<<dim_row,dim_col>>>(dev_grad_out_a,0.0);
    //cudaDeviceSynchronize();
    _cuda_mat_init<<<dim_row,dim_col>>>(dev_grad_out_b,0.0);
    //cudaDeviceSynchronize();

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_mult_e_mat_fwd
(
    float *dev_in_mat_a,
    float *dev_in_mat_b,
    float *dev_out_mat,
    float *in_mat_a,
    float *in_mat_b,
    float *out_mat,
    unsigned int dim_row,
    unsigned int dim_col
)
{
    char f_name[] = "cuda_mult_e_mat_fwd";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
        printf("dev_out_mat      : %p\n",dev_out_mat);
#endif
    _cuda_vec_vec_mult<<<dim_row,dim_col>>>(dev_in_mat_a, dev_in_mat_b, dev_out_mat);
    //cudaDeviceSynchronize();

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_mult_e_mat_bwd
(
    float *dev_in_mat_a,
    float *dev_in_mat_b,
    float *dev_grad_out_a,
    float *dev_grad_out_b,
    float *dev_grad_in,
    float *grad_in,
    float *grad_out_a,
    float *grad_out_b,
    unsigned int dim_row,
    unsigned int dim_col
)
{
    char f_name[] = "cuda_mult_e_mat_bwd";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);

        printf("dev_grad_out_a   : %p\n",*dev_grad_out_a);
        printf("dev_grad_out_b   : %p\n",*dev_grad_out_b);
#endif

    _cuda_vec_vec_mult<<<dim_row,dim_col>>>(dev_grad_in,dev_in_mat_b,dev_grad_out_a);
    //cudaDeviceSynchronize();
    _cuda_vec_vec_mult<<<dim_row,dim_col>>>(dev_grad_in,dev_in_mat_a,dev_grad_out_b);
    //cudaDeviceSynchronize();

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_mult_e_mat_destructor
(
)
{
}


extern "C"
void cuda_memcpy_dev_to_host
(
    float *host,
    float *dev,
    unsigned int size
)
{
    char f_name[] = "cuda_memcpy_dev_to_host";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
#endif
    cudaMemcpy( host, dev, size*sizeof(float), cudaMemcpyDeviceToHost );

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

//
extern "C"
void cuda_binarization
(
    float *dev_in_vec,
    unsigned int size
)
{
    char f_name[] = "cuda_binarization";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);

#endif

    unsigned int num_block=size/CUDA_MAX_NUM_THREAD+1;

    _cuda_binarization<<<num_block,CUDA_MAX_NUM_THREAD>>>(dev_in_vec,dev_in_vec);
    //cudaDeviceSynchronize();

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

//
extern "C"
void cuda_quantization
(
    float *dev_in_vec,
    unsigned int size,
    unsigned int iwl,
    unsigned int frac,
	unsigned int f_mode
)
{
    char f_name[] = "cuda_quantization";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);

#endif

    unsigned int num_block=size/CUDA_MAX_NUM_THREAD+1;

    _cuda_quantization<<<num_block,CUDA_MAX_NUM_THREAD>>>(dev_in_vec,iwl,frac,f_mode);

    //cudaDeviceSynchronize();
    CUDA_ERR(f_name,cudaPeekAtLastError());
}


////////////////////////////////////////////////////////////////////////////////
// activation
////////////////////////////////////////////////////////////////////////////////
extern "C"
void cuda_activation_constructor
(
    float **dev_out,
    float **dev_grad_out,
    unsigned int dim
)
{
    char f_name[] = "cuda_activation_constructor";

#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
#endif
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_out, dim*sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_grad_out, dim*sizeof(float)));

#if CUDA_DEBUG
        printf("not implemented debug log");
        CUDA_MCHK(f_name);
#endif

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_activation_init
(
    float *dev_out,
    float *dev_grad_out,
    unsigned int dim
)
{
    char f_name[] = "cuda_activation_init";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
        CUDA_MCHK(f_name);
#endif
    _cuda_mat_init<<<dim,1>>>(dev_out,0.0);
    _cuda_mat_init<<<dim,1>>>(dev_grad_out,0.0);

    CUDA_ERR(f_name,cudaPeekAtLastError());
}


extern "C"
void cuda_activation_fwd
(
    float *dev_in,
    float *dev_out,
    char *type_act,
    unsigned int dim,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac,
	unsigned int f_mode
)
{
    char f_name[] = "cuda_activation_fwd";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
#endif
    if(!strcmp(type_act,"NULL")) {
        _cuda_bypass<<<1,dim>>>(dev_in,dev_out,f_fixed,iwl,frac,f_mode);
    } else if(!strcmp(type_act,"SIGMOID")) {
        _cuda_sigmoid<<<1,dim>>>(dev_in,dev_out,f_fixed,iwl,frac,f_mode);
    } else if(!strcmp(type_act,"RELU")) {
		_cuda_relu<<<1,dim>>>(dev_in,dev_out,f_fixed,iwl,frac,f_mode);
		/*
		cudaDeviceSynchronize();
 		printf("in: ");
    	cudaDeviceSynchronize();
    	_cuda_printf_vec<<<1,1>>>(dev_in,dim);
		cudaDeviceSynchronize();
    	printf("out: ");
    	cudaDeviceSynchronize();
    	_cuda_printf_vec<<<1,1>>>(dev_out,dim);
		cudaDeviceSynchronize();
		*/
    }

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_activation_bwd
(
    float *dev_out,
    float *dev_grad_in,
    float *dev_grad_out,
    char *type_act,
    unsigned int dim,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac,
	unsigned int f_mode
)
{

    char f_name[] = "cuda_activation_bwd";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
#endif
    if(!strcmp(type_act,"NULL")) {
        _cuda_bypass<<<1,dim>>>(dev_grad_in,dev_grad_out,f_fixed,1,iwl+frac-1,f_mode);
    } else if(!strcmp(type_act,"SIGMOID")) {
        _cuda_sigmoid_bwd<<<1,dim>>>(dev_out,dev_grad_in,dev_grad_out,f_fixed,1,iwl+frac-1,f_mode);
    } else if(!strcmp(type_act,"RELU")) {
        _cuda_relu_bwd<<<1,dim>>>(dev_out,dev_grad_in,dev_grad_out,f_fixed,1,iwl+frac-1,f_mode);
    }

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_activation_destructor
(
    float *dev_out,
    float *dev_grad_out
)
{
    char f_name[] = "cuda_activation_destructor";

    CUDA_ERR(f_name,cudaFree(dev_out));
    CUDA_ERR(f_name,cudaFree(dev_grad_out));
}



////////////////////////////////////////////////////////////////////////////////
// set value
////////////////////////////////////////////////////////////////////////////////

__global__ void _cuda_set_value(float *dest, float value, unsigned int start_idx, unsigned int stride) {
    unsigned int ind;

    ind = blockIdx.x*blockDim.x+threadIdx.x;

    if( ind%stride == start_idx ) {
        dest[ind] = value;
    }
}

extern "C"
void cuda_set_value
(
    float *dest,
    float value,
    unsigned int dim,
    unsigned int start_idx,
    unsigned int stride
)
{
    unsigned int num_block;
    unsigned int num_thread;

    num_block = dim/CUDA_MAX_NUM_THREAD+1;
    num_thread = CUDA_MAX_NUM_THREAD;

    _cuda_set_value<<<num_block,num_thread>>>(dest,value,start_idx,stride);
}

////////////////////////////////////////////////////////////////////////////////
// gray code
////////////////////////////////////////////////////////////////////////////////
/*
extern "C"
void cuda_bin2gray_test
(
 	unsigned int *bin,
	unsigned int *gray,
    unsigned int *dev_bin,
    unsigned int *dev_gray,
	unsigned int dim,
    int idx_bit_low,
    int idx_bit_high
)
{

    char f_name[] = "cuda_bin2gray";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
#endif

	printf("%d %d\n",idx_bit_low,idx_bit_high);

	if((idx_bit_high<idx_bit_low)||(idx_bit_low<0)||(idx_bit_high>32)) {
        printf("*E: _cuda_bin2gray : out of range idx bit : %d - %d\n",idx_bit_high,idx_bit_low);
        exit(0);
    }

    CUDA_ERR(f_name, cudaMalloc( (void**) (&dev_bin), dim*sizeof(unsigned int)));
    CUDA_ERR(f_name, cudaMalloc( (void**) (&dev_gray), dim*sizeof(unsigned int)));

    cudaMemcpy(dev_bin, bin, dim*sizeof(unsigned int), cudaMemcpyHostToDevice);

	_cuda_bin2gray<<<1,dim>>>(dev_bin,dev_gray,idx_bit_low,idx_bit_high);

    cudaMemcpy(gray, dev_gray, dim*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_gray2bin_test
(
	unsigned int *gray,
 	unsigned int *bin,
    unsigned int *dev_gray,
    unsigned int *dev_bin,
	unsigned int dim,
    int idx_bit_low,
    int idx_bit_high
)
{

    char f_name[] = "cuda_bin2gray";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
#endif

	printf("%d %d\n",idx_bit_low,idx_bit_high);

	if((idx_bit_high<idx_bit_low)||(idx_bit_low<0)||(idx_bit_high>32)) {
        printf("*E: _cuda_gray2bin : out of range idx bit : %d - %d\n",idx_bit_high,idx_bit_low);
        exit(0);
    }

    CUDA_ERR(f_name, cudaMalloc( (void**) (&dev_bin), dim*sizeof(unsigned int)));
    CUDA_ERR(f_name, cudaMalloc( (void**) (&dev_gray), dim*sizeof(unsigned int)));

    cudaMemcpy(dev_gray, gray, dim*sizeof(unsigned int), cudaMemcpyHostToDevice);

	_cuda_gray2bin<<<1,dim>>>(dev_gray,dev_bin,idx_bit_low,idx_bit_high);

    cudaMemcpy(bin, dev_bin, dim*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    CUDA_ERR(f_name,cudaPeekAtLastError());
}
*/


////////////////////////////////////////////////////////////////////////////////
// scale layer
////////////////////////////////////////////////////////////////////////////////
extern "C"
void cuda_scale_constructor
(
    float **dev_w,
    float **dev_w_del,
    float **dev_w_best,
    float **dev_out,
    float **dev_grad_out,
    unsigned int dim
)
{
    char f_name[] = "cuda_scale_constructor";

#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
#endif
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_w, sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_w_del, sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_w_best, sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_out, dim*sizeof(float)));
    CUDA_ERR(f_name,cudaMalloc( (void**) dev_grad_out, dim*sizeof(float)));

#if CUDA_DEBUG
        printf("not implemented debug log");
        CUDA_MCHK(f_name);
#endif

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_scale_init
(
    float *dev_w,
    float *dev_w_del,
    float *dev_out,
    float *dev_grad_out,
    float *w,
    unsigned int dim
)
{
    char f_name[] = "cuda_scale_init";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
        CUDA_MCHK(f_name);
#endif
    _cuda_mat_init<<<1,1>>>(dev_w_del,0.0);
    _cuda_mat_init<<<dim,1>>>(dev_out,0.0);
    _cuda_mat_init<<<dim,1>>>(dev_grad_out,0.0);


    cudaMemcpy( dev_w, w, sizeof(float), cudaMemcpyHostToDevice );

    CUDA_ERR(f_name,cudaPeekAtLastError());
}


extern "C"
void cuda_scale_fwd
(
    float *dev_in,
    float *dev_w,
    float *dev_out,
    unsigned int dim,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac,
	unsigned int f_mode,
    bool verbose
)
{
    char f_name[] = "cuda_scale_fwd";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
#endif
    _cuda_vec_const_mult<<<1,dim>>>(dev_in,dev_w,dev_out);

    //_cuda_printf_vec<<<1,1>>>(dev_out,dim);

    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_scale_bwd
(
    float *dev_in,
    float *dev_grad_in,
    float *dev_w,
    float *dev_w_del,
    float *dev_grad_out,
    unsigned int dim,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac,
	unsigned int f_mode,
    bool verbose
)
{

    char f_name[] = "cuda_scale_bwd";
#if CUDA_DEBUG
        printf("\n< %s >\n",f_name);
#endif
    // dev_w_del
    _cuda_vec_vec_mult_accum_scalar<<<1,dim>>>(dev_grad_in,dev_in,dev_w_del);

    // dev_grad_out
    _cuda_vec_const_mult<<<1,dim>>>(dev_grad_in,dev_w,dev_grad_out);


    CUDA_ERR(f_name,cudaPeekAtLastError());
}

extern "C"
void cuda_scale_w_up
(
    float *dev_w,
    float *dev_w_del,
    //float *dev_grad_l2_norm,
    unsigned int dim,
    unsigned int batch_size,
    float *lr,
    float *lambda,
    //float *max_grad_l2_norm,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac,
	unsigned int f_mode,
    bool verbose
)
{
    char f_name[] = "cuda_scale_w_up";

    //_cuda_mat_init<<<1,1>>>(dev_grad_l2_norm,0.0);

    //_cuda_l2_norm<<<dim_out,dim_in>>>(dev_w_mat_del, dev_grad_l2_norm);


    _cuda_w_up<<<1,1>>>(dev_w_del,dev_w,(batch_size*dim),*lr,*lambda);
    
    /*
    printf("dev_w_del: ");
    cudaDeviceSynchronize();
    _cuda_printf_vec<<<1,1>>>(dev_w_del,1);
    cudaDeviceSynchronize();
    printf("dev_w: ");
    cudaDeviceSynchronize();
    _cuda_printf_vec<<<1,1>>>(dev_w,1);
    cudaDeviceSynchronize();
    printf("\n");
    */

    //_cuda_mat_w_up<<<dim_out,dim_in>>>(dev_w_mat_del,dev_w_mat,batch_size,*lr,*lambda,*max_grad_l2_norm,dev_grad_l2_norm, f_fixed,iwl,frac);


    _cuda_mat_init<<<1,1>>>(dev_w_del,0.0);

    CUDA_ERR(f_name,cudaPeekAtLastError());
}




extern "C"
void cuda_scale_destructor
(
    float *dev_w,
    float *dev_w_del,
    float *dev_w_best,
    float *dev_out,
    float *dev_grad_out
)
{
    char f_name[] = "cuda_scale_destructor";

    CUDA_ERR(f_name,cudaFree(dev_w));
    CUDA_ERR(f_name,cudaFree(dev_w_del));
    CUDA_ERR(f_name,cudaFree(dev_w_best));
    CUDA_ERR(f_name,cudaFree(dev_out));
    CUDA_ERR(f_name,cudaFree(dev_grad_out));
}


//
extern "C"
void cuda_copy_dev2host
(
    float *host,
    float *dev,
    unsigned int size
)
{
    char f_name[] = "cuda_copy_dev2host";

    cudaMemcpy(host, dev, size*sizeof(float), cudaMemcpyDeviceToHost);

}
