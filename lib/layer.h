#ifndef __LAYER__
#define __LAYER__

#include "common.h"



extern bool en_gpu_model;
extern bool en_cpu;
////////////////////////////////////////////////////////////////////////////////
// dot - mat vec
////////////////////////////////////////////////////////////////////////////////
typedef struct {
    unsigned int dim_mat_x_max;
    unsigned int dim_mat_x;
    unsigned int dim_mat_y;
    unsigned int dim_vec;

    float **in_mat;
    float *in_vec;

    float *out_vec;

    float *grad_in;

    float **grad_out_mat;
    float *grad_out_vec;

    bool f_trans;       // true : ( a x b )' x ( a x 1 ) // false : ( a x b ) x ( b x 1 )

    // fixed point
    bool f_fixed;
    unsigned int iwl_m;
    unsigned int frac_m;
    unsigned int iwl_v;
    unsigned int frac_v;
	unsigned int f_mode;

    // GPU
    float *dev_in_mat;
    float *dev_in_vec;

    float *dev_out_vec;

    float *dev_grad_in;
    float *dev_grad_out_mat;
    float *dev_grad_out_vec;

    float *dev_f_overflow;

    // for attention mode
    unsigned int attention_mode;

    // for hamming similiarity
    float *dev_cliff_marker;
} dot_mat_vec;

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
);
double dot_mat_vec_init(dot_mat_vec *dot);
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
);
double dot_mat_vec_fwd(dot_mat_vec *dot, bool verbose);
double dot_mat_vec_bwd(dot_mat_vec *dot, unsigned int hop, bool verbose);
double dot_mat_vec_destructor(dot_mat_vec *dot);

////////////////////////////////////////////////////////////////////////////////
// softmax
////////////////////////////////////////////////////////////////////////////////
typedef struct {
    unsigned int dim_max;
    unsigned int dim;

    bool f_exp_plan;
    bool f_shift_based;

    float *in_vec;
    float *out_vec;
    float *grad_in;
    float *grad_out;

    // gpu
    float *dev_in_vec;
    float *dev_out_vec;
    float *dev_grad_in;
    float *dev_grad_out;
    float *dev_max;
} softmax;

double softmax_constructor(softmax *sf, unsigned int dim_max, bool f_exp_table_based, bool f_shift_based, FILE *fp_result);
double softmax_init(softmax *sf);
double softmax_in
(
    softmax *sf,
    unsigned int dim,
    float *in_vec,
    float *grad_in,
    float *dev_in_vec,
    float *dev_grad_in
);
double softmax_fwd(softmax *sf, bool verbose);
double softmax_bwd(softmax *sf, bool verbose);
double softmax_distructor(softmax *sf);

////////////////////////////////////////////////////////////////////////////////
// sum - element sum vector (vector + vector)
////////////////////////////////////////////////////////////////////////////////
typedef struct {
    unsigned int dim;

    float *in_vec_a;
    float *in_vec_b;

    float *out_vec;

    float *grad_in;
    float *grad_out;

    // fixed point
    bool f_fixed;
    unsigned int iwl;
    unsigned int frac;
    unsigned int f_mode;

    // gpu
    float *dev_in_vec_a;
    float *dev_in_vec_b;

    float *dev_out_vec;

    float *dev_grad_in;
    float *dev_grad_out;
} sum_vec;


double sum_vec_constructor
(
    sum_vec *sv,
    unsigned int dim,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac,
    unsigned int f_mode,
    FILE *fp_result
);

double sum_vec_init(sum_vec *sv);
double sum_vec_in
(
    sum_vec *sv,
    float *in_vec_a,
    float *in_vec_b,
    float *grad_in,
    float *dev_in_vec_a,
    float *dev_in_vec_b,
    float *dev_grad_in
);
double sum_vec_fwd(sum_vec *sv, bool verbose);
double sum_vec_bwd(sum_vec *sv);
double sum_vec_destructor(sum_vec *sv);

////////////////////////////////////////////////////////////////////////////////
// dense - fully connected
////////////////////////////////////////////////////////////////////////////////
typedef struct {
    unsigned int dim_in;
    unsigned int dim_out;

    float *in_vec;

    float *in_vec_stored;
    float *out_vec;

    float **w_mat;
    float **w_mat_del;
    float **w_mat_momentum;

    float *bias;
    float *bias_del;

    float *grad_in;
    float *grad_out;

    char activation[10];

    float grad_l2_norm;
    float grad_bias_l2_norm;

    // rmsporp
    float **grad_accum;

    bool en_max_grad_l2_norm;
    float max_grad_l2_norm;

    // adaMax
    float **adam_m;
    float **adam_v;
    float adam_beta_1;
    float adam_beta_2;

    // fixed point
    bool f_fixed;
    unsigned int iwl_in;
    unsigned int frac_in;
    unsigned int iwl_w;
    unsigned int frac_w;
    unsigned int f_mode;

    float *f_overflow;       // over & under flow

    // GPU
    float *dev_in_vec;
    float *dev_in_vec_stored;

    float *dev_out_vec;

    float *dev_w_mat;
    float *dev_w_mat_del;

    float *dev_w_mat_best;

    float *dev_bias;
    float *dev_bias_del;

    float *dev_grad_in;
    float *dev_grad_out;

    float *dev_grad_l2_norm;
    float *dev_grad_bias_l2_norm;

    float *dev_f_overflow;
} dense;

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
);
double dense_init(dense *ds);
double dense_in
(
    dense *ds,
    float *in_vec,
    float *grad_in,
    float *dev_in_vec,
    float *dev_grad_in
);
double dense_fwd(dense *ds, bool verbose);
double dense_fwd_binary(dense *ds, bool verbose);
double dense_bwd(dense *ds, bool verbose);
double dense_w_up(dense *ds, float lr, unsigned int m, float lambda, bool verbose);
double dense_destructor(dense *ds);

////////////////////////////////////////////////////////////////////////////////
// dense_mat - fully connected
////////////////////////////////////////////////////////////////////////////////
typedef struct {
    unsigned int dim_len_max;
    unsigned int dim_len;
    unsigned int dim_in;
    unsigned int dim_out;

    float **in_mat;

    float **out_mat;

    float **w_mat;
    float **w_mat_del;

    float *bias;
    float *bias_del;

    float **grad_in;
    float **grad_out;

    float grad_l2_norm;
    float grad_bias_l2_norm;

    bool en_max_grad_l2_norm;
    float max_grad_l2_norm;

    // fixed point
    bool f_fixed;
    unsigned int iwl;
    unsigned int frac;
    unsigned int f_mode;

    float **f_overflow;

    // GPU
    float *dev_w_mat;
    float *dev_w_mat_del;

    float *dev_w_mat_best;

    float *dev_bias;
    float *dev_bias_del;
    float *dev_in_mat;
    float *dev_out_mat;
    float *dev_grad_in;
    float *dev_grad_out;
    float *dev_grad_l2_norm;
    float *dev_grad_bias_l2_norm;
    float *dev_f_overflow;

} dense_mat;

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
);

double dense_mat_init(dense_mat *ds_m);
double dense_mat_in
(
    dense_mat *ds_m,
    unsigned int dim_len,
    float **in_mat,
    float **grad_in,
    float *dev_in_mat,
    float *dev_grad_in
);
double dense_mat_fwd(dense_mat *ds_m, bool verbose);
double dense_mat_bwd(dense_mat *ds_m, bool verbose);
double dense_mat_w_up(dense_mat *ds_m, float lr, unsigned int m, float lambda, bool verbose);
double dense_mat_destructor(dense_mat *ds_m);


////////////////////////////////////////////////////////////////////////////////
// cost function - corss entropy
////////////////////////////////////////////////////////////////////////////////
typedef struct {
    unsigned int k;     // number of class
    float *h;          // exptected value
    float *y;

    float cost;

    float *grad_out;

    float lambda;

    // gpu
    float *dev_cost_train;
    float *dev_cost_valid;
    float *dev_cost_test;
    unsigned int *dev_m_cnt_train;
    unsigned int *dev_m_cnt_valid;
    unsigned int *dev_m_cnt_test;
    unsigned int *dev_pred_i;
    float *dev_h;
    float *dev_y;
    float *dev_grad_out;


} cross_entropy;

double cross_entropy_constructor
(
    cross_entropy *ce,
    unsigned int k,
    FILE *fp_result
);
double cross_entropy_init(cross_entropy *ce);
double cross_entropy_in
(
    cross_entropy *ce,
    float *h,
    float *y,
    float *dev_h,
    float *dev_y
);
double cross_entropy_run(cross_entropy *ce, unsigned int mode);
double cross_entropy_cost_load(cross_entropy *ce, float *cost_train, float *cost_valid, float *cost_test);
double cross_entropy_m_cnt_load(cross_entropy *ce, unsigned int *m_cnt_train, unsigned int *m_cnt_valid, unsigned int *m_cnt_test);
double cross_entropy_destructor(cross_entropy *ce);


////////////////////////////////////////////////////////////////////////////////
// maxout
////////////////////////////////////////////////////////////////////////////////
typedef struct {
    unsigned int dim_in_max;
    unsigned int dim_hd_max;
    unsigned int dim_out_max;

    unsigned int dim_in;
    unsigned int dim_hd;
    unsigned int dim_out;

    float *in_vec;

    float ***w_mat;
    float ***w_mat_del;

    float **b_mat;
    float **b_mat_del;

    unsigned int *ind_max;

    float *out_vec;

    float *grad;

    float sq_sum;
} maxout;

double maxout_constructor(maxout *mxout, unsigned int dim_in_max, unsigned int dim_hd_max, unsigned int dim_out_max);
double maxout_init(maxout *mxout);
double maxout_in(maxout *mxout, unsigned int dim_in, unsigned int dim_hd, unsigned int dim_out, float *in_vec);
double maxout_fwd(maxout *mxout);
double maxout_bwd(maxout *mxout, float *grad);
double maxout_w_up(maxout *mxout, float lr, unsigned int m, float lambda, bool verbose);
double maxout_destructor(maxout *mxout);


////////////////////////////////////////////////////////////////////////////////
// cost function - SE - Squared Error
////////////////////////////////////////////////////////////////////////////////
typedef struct {
    unsigned int dim_x_max;
    unsigned int dim_x;

    float *h_vec;      //

    float *y_vec;      //

    float cost;
    float *grad;
} se;

double se_constructor(se *se, unsigned int dim_x_max);
double se_in(se *se, unsigned int dim_x, float *h_vec, float *y_vec);
double se_run(se *se);
double se_destructor(se *se);


////////////////////////////////////////////////////////////////////////////////
// mult_e_vec - element-wise multiply vector
////////////////////////////////////////////////////////////////////////////////
typedef struct {
    unsigned int dim_max;

    unsigned int dim;

    float *in_vec_a;
    float *in_vec_b;

    float *out_vec;

    float *grad_in;
    float *grad_out_a;
    float *grad_out_b;

    // fixed point
    bool f_fixed;
    unsigned int iwl;
    unsigned int frac;

    // gpu
    float *dev_in_vec_a;
    float *dev_in_vec_b;

    float *dev_out_vec;

    float *dev_grad_in;
    float *dev_grad_out_a;
    float *dev_grad_out_b;
} mult_e_vec;

double mult_e_vec_constructor
(
    mult_e_vec *mev,
    unsigned int dim_max,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac
);

double mult_e_vec_init(mult_e_vec *mev);
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
);
double mult_e_vec_fwd(mult_e_vec *mev, bool verbose);
double mult_e_vec_bwd(mult_e_vec *mev);
double mult_e_vec_destructor(mult_e_vec *mev);


////////////////////////////////////////////////////////////////////////////////
// mult_e_mat - element-wise multiply matrix
////////////////////////////////////////////////////////////////////////////////
typedef struct {
    unsigned int dim_row_max;
    unsigned int dim_col_max;

    unsigned int dim_row;
    unsigned int dim_col;

    float **in_mat_a;
    float **in_mat_b;

    float **out_mat;

    float **grad_in;
    float **grad_out_a;
    float **grad_out_b;

    // fixed point
    bool f_fixed;
    unsigned int iwl;
    unsigned int frac;

    // gpu
    float **dev_in_mat_a;
    float **dev_in_mat_b;

    float **dev_out_mat;

    float **dev_grad_in;
    float **dev_grad_out_a;
    float **dev_grad_out_b;
} mult_e_mat;

double mult_e_mat_constructor
(
    mult_e_mat *mem,
    unsigned int dim_row_max,
    unsigned int dim_col_max,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac
);

double mult_e_mat_init(mult_e_mat *mem);
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
);
double mult_e_mat_fwd(mult_e_mat *mem, bool verbose);
double mult_e_mat_bwd(mult_e_mat *mem);
double mult_e_mat_destructor(mult_e_mat *mem);


////////////////////////////////////////////////////////////////////////////////
// activation
////////////////////////////////////////////////////////////////////////////////
typedef struct {
    unsigned int dim;

    float *in;
    float *out;

    float *grad_in;
    float *grad_out;

    char type_act[10];

    // fixed point
    bool f_fixed;
    unsigned int iwl;
    unsigned int frac;
    unsigned int f_mode;

    // gpu
    float *dev_in;
    float *dev_out;

    float *dev_grad_in;
    float *dev_grad_out;
} activation;

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
);

double activation_init(activation *act);
double activation_in
(
    activation *act,
    float *in,
    float *grad_in,
    float *dev_in,
    float *dev_grad_in
);
double activation_fwd(activation *act, bool verbose);
double activation_bwd(activation *act, bool verbose);
double activation_destructor(activation *act);

/*
////////////////////////////////////////////////////////////////////////////////
// lookup table
////////////////////////////////////////////////////////////////////////////////
typedef struct {
    unsigned int addr_in_max;
    unsigned int dim_in_max;
    unsigned int dim_out_max;
    unsigned int dim_in;
    unsigned int dim_out;

    float *in_vec;

    float *in_vec_stored;
    float *out_vec;

    float **w_mat;
    float **w_mat_del;
    float **w_mat_momentum;

    float *bias;
    float *bias_del;

    float *grad_in;
    float *grad_out;

    char activation[10];

    float grad_l2_norm;
    float grad_bias_l2_norm;

    // rmsporp
    float **grad_accum;

    bool en_max_grad_l2_norm;
    float max_grad_l2_norm;

    // adaMax
    float **adam_m;
    float **adam_v;
    float adam_beta_1;
    float adam_beta_2;

    // fixed point
    bool f_fixed;

    // GPU
    float *dev_in_vec;
    float *dev_in_vec_stored;

    float *dev_out_vec;

    float *dev_w_mat;
    float *dev_w_mat_del;

    float *dev_bias;
    float *dev_bias_del;

    float *dev_grad_in;
    float *dev_grad_out;

    float *dev_grad_l2_norm;
    float *dev_grad_bias_l2_norm;
} dense;

double dense_constructor
(
    dense *ds,
    unsigned int dim_in_max,
    unsigned int dim_out_max,
    bool en_max_grad_l2_norm,
    float max_grad_l2_norm,
    char *activation,
    bool f_fixed
);
double dense_init(dense *ds);
double dense_in
(
    dense *ds,
    unsigned int dim_in,
    unsigned int dim_out,
    float *in_vec,
    float *grad_in,
    float *dev_in_vec,
    float *dev_grad_in
);
double dense_fwd(dense *ds, bool verbose);
double dense_fwd_binary(dense *ds, bool verbose);
double dense_bwd(dense *ds, bool verbose);
double dense_w_up(dense *ds, float lr, unsigned int m, float lambda, bool verbose);
double dense_destructor(dense *ds);

*/

////////////////////////////////////////////////////////////////////////////////
// scale - learnalbe scale layer
////////////////////////////////////////////////////////////////////////////////
typedef struct {
    unsigned int dim_max;
    unsigned int dim;

    float *in;
    float *out;

    float *w;
    float *w_del;

    float *grad_in;
    float *grad_out;

    // fixed point
    bool f_fixed;
    unsigned int iwl;
    unsigned int frac;
    unsigned int f_mode;

    // gpu
    float *dev_in;
    float *dev_out;

    float *dev_w;
    float *dev_w_del;

    float *dev_w_best;

    float *dev_grad_in;
    float *dev_grad_out;
} scale;

double scale_constructor
(
    scale *sc,
    unsigned int dim,
    bool f_fixed,
    unsigned int iwl,
    unsigned int frac,
    unsigned int f_mode,
    FILE *fp_result
);

double scale_init(scale *sc);
double scale_in
(
    scale *sc,
    unsigned int dim,
    float *in,
    float *grad_in,
    float *dev_in,
    float *dev_grad_in
);
double scale_fwd(scale *sc, bool verbose);
double scale_bwd(scale *sc, bool verbose);
double scale_w_up(scale *sc, float lr, unsigned int m, float lambda, bool verbose);
double scale_destructor(scale *sc);

#endif
