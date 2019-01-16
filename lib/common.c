#include "common.h"

int mat_copy(float **sr, float **de, unsigned int row, unsigned int col, bool f_tran) {
    unsigned int i, j;

    if(f_tran) {                                        // copy with transposed index
        for(i=0;i<row;i++) {
            for(j=0;j<col;j++) {
                de[j][i] = sr[i][j];
            }
        }
    } else {                                            // copy
        for(i=0;i<row;i++) {
            for(j=0;j<col;j++) {
                de[i][j] = sr[i][j];
            }
        }
    }

    /*
        for(j=0;j<dim_y;j++) {
            for(i=0;i<dim_x;i++) {
                printf(" %f %f \n",de[j][i],sr[j][i]);
            }
        }
    */

    return 0;
}

float gaussian_random(float mean, float stddev) {
    float r1, r2;
    float w;

    do {
        r1 = (float)2.0*rand()/(float)RAND_MAX - 1.0;
        r2 = (float)2.0*rand()/(float)RAND_MAX - 1.0;
        w = r1*r1 + r2*r2;
    } while ( w>= 1.0 );


    //printf("r1 : %.3f %.3f\n",r1,r2);


    w = sqrt( -2.0*log(w)/w );

    return mean + r1*w*stddev;
}

// Piecewise Linear Approximation for Non-linear function
float exp_plan(float in) {

    unsigned int i;

    float out;
    float tmp;

    out = w_exp_plan[0]*in+b_exp_plan[0];

    for(i=1;i<num_exp_plan;i++) {
        tmp = w_exp_plan[i]*in+b_exp_plan[i];

        if(out < tmp) {
            out = tmp;
        }

        //printf("tmp : %f, in : %f \n",tmp, in);
    }
    //printf("out : %f\n", out);
    //printf("\n");

    return out;
}

//
float sigmoid(float x) {
    return 1.0/(1.0+exp(-x));
}

float sigmoid_deriv(float x) {
    //return sigmoid(x)*(1.0-sigmoid(x));
    return x*(1.0-x);
}

float relu(float in) {
    if(in>0.0) {
        return in;
    } else {
        return 0.0;
    }
}

float relu_deriv(float out) {
    if(out>0.0) {
        return 1.0;
    } else {
        return 0.0;
    }
}

//
/*
int fixed_trans_64_32(long fixed_64) {
    int fixed_32=0x00000000;
    int sign;
    unsigned int i;
    int mask = 0x80000000;

    //printf("WL : %d, IWL : %d, FRAC : %d\n",BW_WL,BW_IWL,BW_FRAC);
    //printf("fixed_64 : %ld\n",fixed_64);
    //printf("fixed_64 : %lx\n",fixed_64);

    if( (fixed_64>>BW_FRAC) > FIXED_MAX_FIXED) {
        //printf("overflow\n");
        fixed_32 = FIXED_MAX_FIXED;
    } else if( (fixed_64>>BW_FRAC) < FIXED_MIN_FIXED) {
        //printf("underflow\n");
        fixed_32 = FIXED_MIN_FIXED;
    } else {
        //fixed_32 = fixed_trans_64_32(fixed_64);

        sign = fixed_64 >> 63 & 0x0000000000000001;
        //printf("sign : %x\n",sign);

        for(i=0;i<32-BW_WL+1;i++) {
            if(sign > 0) {
                fixed_32 = fixed_32 | mask&(0xffffffff);
            }
            mask = mask >> 1;
        }
        //printf("fixed_32_signed : %8x\n",fixed_32);

        fixed_64 = fixed_64 >> BW_FRAC;
        //printf("fixed_64_shfited : %lx\n",fixed_64);

        mask = 0x0000000000000001 << (BW_IWL+BW_FRAC-1);
        for(i=0;i<BW_IWL+BW_FRAC;i++) {
            //printf("mask : %lx\n",mask);
            fixed_32 = (fixed_32) | fixed_64&mask;
            //printf("fixed_32 : %8x\n",fixed_32);
            mask = mask >> 1;
        }
    }

    //printf("fixed_32 : %8x\n",fixed_32);

    return fixed_32;
}
*/


//
/*
float mult_fixed_32(float float_a_32, float float_b_32) {
    int fixed_a_32;
    int fixed_b_32;
    int fixed_c_32;
    long fixed_c_64;
    float float_32;

    //printf("float a : %f\n",float_a_32);
    //printf("float b : %f\n",float_b_32);

    fixed_a_32 = FLOAT2FIXED(float_a_32);
    fixed_b_32  = FLOAT2FIXED(float_b_32);
    fixed_c_64 = (long)fixed_a_32*(long)fixed_b_32;

    fixed_c_32 = fixed_trans_64_32(fixed_c_64);

    float_32 = FIXED2FLOAT(fixed_c_32);

    return float_32;
}
*/

//
/*
float add_fixed_32(float float_a_32, float float_b_32) {
    int fixed_a_32;
    int fixed_b_32;
    int fixed_c_32;
    long fixed_c_64;
    float float_32;

    fixed_a_32 = FLOAT2FIXED(float_a_32);
    fixed_b_32 = FLOAT2FIXED(float_b_32);
    fixed_c_64 = fixed_a_32+fixed_b_32;

    //printf("fixed_a_32 : %d\n",fixed_a_32);
    //printf("fixed_b_32 : %d\n",fixed_a_32);

    if(fixed_c_64 > FIXED_MAX_FIXED) {
        fixed_c_32 = FIXED_MAX_FIXED;
    } else if(fixed_c_64 < FIXED_MIN_FIXED) {
        fixed_c_32 = FIXED_MIN_FIXED;
    } else {
        fixed_c_32 = fixed_c_64;
        //printf("lf : %lf\n",fixed_c_64);
    }

    float_32 = FIXED2FLOAT(fixed_c_32);

    return float_32;
}
*/

// clip function
float clip(float in, float bound_low, float bound_high) {
    float result;

    if( in > bound_high ) {
        result = bound_high;
    } else if( in < bound_low ) {
        result = bound_low;
    } else {
        result = in;
    }

    return result;
}

// hamming distance
unsigned int hamming_similarity(int a, int b, unsigned int num_bit) {
    unsigned int bit_count=0;

    unsigned int i;

    if(num_bit > 32) {
        printf("*E : hamming_similarity : out of range ( >32) \n");

        return 0;
    }

    for(i=0;i<num_bit;i++) {
        if( (a&(0x80000000>>i)) == (b&(0x80000000>>i)) ) {
            bit_count++;
        }
    }
    /*
    printf("a : %08x\n",a);
    printf("b : %08x\n",b);
    printf("bit count : %d\n",bit_count);
    */

    return bit_count;
}

// hamming distance
float hamming_similarity_w(int a, int b, unsigned int num_bit, bool verbose) {
    float bit_count_w=0;

    float similarity_w;

    int i;

    float sign_a;
    float sign_b;

    if(num_bit > 32) {
        printf("*E : hamming_similarity_w : out of range ( >32) \n");

        return 0;
    }

    // last version
    /*
    for(i=0;i<num_bit;i++) {
        if( (a&(0x80000000>>i)) == (b&(0x80000000>>i)) ) {
            //bit_count_w += (int)pow(2,num_bit-i-1);
            bit_count_w += pow(2,(int)(-i-1));

            //printf("%d : %d\n",num_bit-i,(int)pow(2,num_bit-i));
            //bit_count += num_bit-i;
        }
    }

    similarity_w = bit_count_w;
    */

    // new
    if((a&0x80000000)==0x00000000) {
        sign_a = 1.0;
    } else {
        sign_a = -1.0;
    }

    if((b&0x80000000)==0x00000000) {
        sign_b = 1.0;
    } else {
        sign_b = -1.0;
    }

    for(i=1;i<num_bit;i++) {
        if( (a&(0x80000000>>i)) == (b&(0x80000000>>i)) ) {
            //bit_count_w += (int)pow(2,num_bit-i-1);
            bit_count_w += pow(2,(int)(-i-1));

            //printf("%d : %d\n",num_bit-i,(int)pow(2,num_bit-i));
            //bit_count += num_bit-i;
        }
    }

    similarity_w = sign_a*sign_b*bit_count_w;

    if(verbose) {
        printf("a : %08x\n",a);
        printf("b : %08x\n",b);
        printf("bit count : %d\n",bit_count_w);
    }

    return similarity_w;
}


void rand_perm(unsigned int *array, unsigned int size) {
    unsigned int i, j, t;

    for(i=0;i<size;i++) {
        array[i] = i;
    }

    for(i=0;i<size;i++) {
        j = rand()%(size-i) + i;
        t = array[j];
        array[j] = array[i];
        array[i] = t;
    }
}

int compare_function(const void *a, const void *b) {
    return ( *(int*)a - *(int*)b );
}

// unsigned binary to gray code
unsigned int bin2gray(unsigned int bin, int idx_bit_low, int idx_bit_high) {

    if((idx_bit_high<idx_bit_low)||(idx_bit_low<0)||(idx_bit_high>32)) {
        printf("*E: bin2gray : out of range idx bit : %d - %d\n",idx_bit_high,idx_bit_low);
        exit(0);
    }

    int i;
    unsigned int gray;

    gray = bin&(0x00000001<<(idx_bit_high));

    //printf("gray: %08x\n",gray);

    for(i=idx_bit_high-1;i>=idx_bit_low;i--) {
        gray = gray|((((bin>>(i+1))&(0x00000001))^((bin>>(i))&(0x00000001)))<<i);

        //printf("%d:%08x, %d:%08x -> %08x\n",i+1,(bin>>(i+1))&(0x00000001), i,(bin>>(i))&(0x00000001), ((bin>>(i+1))&(0x00000001))^((bin>>(i))&(0x00000001)));
    
        //printf("gray: %08x\n",gray);

    }
    
    return gray;
}

// gray code to unsigned binary
unsigned int gray2bin(unsigned int gray, int idx_bit_low, int idx_bit_high) {

    if((idx_bit_high<idx_bit_low)||(idx_bit_low<0)||(idx_bit_high>32)) {
        printf("*E: gray2bin : out of range idx bit : %d - %d\n",idx_bit_high,idx_bit_low);
        exit(0);
    }

    int i;
    unsigned int bin;
    unsigned int bin_last;

    //bin = gray&(0x00000001<<(idx_bit_high));
    bin_last = (gray>>idx_bit_high)&0x00000001;
    bin = bin_last<<idx_bit_high;

    //printf("gray: %08x\n",gray);
    //


    for(i=idx_bit_high-1;i>=idx_bit_low;i--) {
        if(((gray>>(i))&0x00000001)==0x00000001) {
            bin_last = (!bin_last)&0x00000001;
        }

        bin = bin|(bin_last<<i);
        //printf("%d:%08x, %d:%08x -> %08x\n",i+1,(bin>>(i+1))&(0x00000001), i,(bin>>(i))&(0x00000001), ((bin>>(i+1))&(0x00000001))^((bin>>(i))&(0x00000001)));
    
        //printf("gray: %08x\n",gray);

    }
    
    return bin;
}
