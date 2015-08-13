#ifndef _LIBSVM_PREDICT_H
#define _LIBSVM_PREDICT_H

#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "svm.h"

//int print_null(const char *s,...) {return 0;}

static int (*info)(const char *fmt,...) = &printf;

static char* readline(FILE *input);
void exit_input_error(int line_num);
void ls_predict(FILE *input, FILE *output, struct svm_model * model);
int predictlr(FILE *input, FILE *output, struct svm_model * model);
void predict_pro(FILE *input, FILE *output, struct svm_model * model,double *prob_estimates);
void exit_with_help();
void mypredict(const char * mytopredictfile_one, const char * mymodelfile, const char * myresult, struct svm_model * model);
int mypredictlr(const char * mytopredictfile_one, const char * mymodelfile, const char * myresult, struct svm_model * model);
void mypredict_probability(const char * mytopredictfile_one, const char * mymodelfile, const char * myresult, struct svm_model* model, double *prob_estimates);

#endif /* _LIBSVM_PREDICT_H */
