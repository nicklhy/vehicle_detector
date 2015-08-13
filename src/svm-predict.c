#include "svm-predict.h"

int max_nr_attr = 64;
int predict_probability=0;
static char *line = NULL;
struct svm_node *x;
//struct svm_model* model;
static int max_line_len;

static char* readline(FILE *input)
{
    int len;

    if(fgets(line,max_line_len,input) == NULL)
        return NULL;

    while(strrchr(line,'\n') == NULL)
    {
        max_line_len *= 2;
        line = (char *) realloc(line,max_line_len);
        len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    return line;
}

void exit_input_error(int line_num)
{
    fprintf(stderr,"Wrong input format at line %d\n", line_num);
    //exit(1);
}

void ls_predict(FILE *input, FILE *output, struct svm_model * model)
{
    int correct = 0;
    int total = 0;
    double error = 0;
    double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

    int svm_type=svm_get_svm_type(model);
    int nr_class=svm_get_nr_class(model);
    double *prob_estimates=NULL;
    int j;

    if(predict_probability)
    {
        if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
            info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
        else
        {
            int *labels=(int *) malloc(nr_class*sizeof(int));
            svm_get_labels(model,labels);
            prob_estimates = (double *) malloc(nr_class*sizeof(double));
            fprintf(output,"labels");
            for(j=0;j<nr_class;j++)
                fprintf(output," %d",labels[j]);
            fprintf(output,"\n");
            free(labels);
        }
    }

    max_line_len = 1024;
    line = (char *)malloc(max_line_len*sizeof(char));
    while(readline(input) != NULL)
    {
        int i = 0;
        double target_label, predict_label;
        char *idx, *val, *label, *endptr;
        int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

        label = strtok(line," \t\n");
        if(label == NULL) // empty line
        {
            exit_input_error(total+1);
            return ;
        }

        target_label = strtod(label,&endptr);
        if(endptr == label || *endptr != '\0')
        {
            exit_input_error(total+1);
            return ;
        }

        while(1)
        {
            if(i>=max_nr_attr-1)	// need one more for index = -1
            {
                max_nr_attr *= 2;
                x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
            }

            idx = strtok(NULL,":");
            val = strtok(NULL," \t");

            if(val == NULL)
                break;
            errno = 0;
            x[i].index = (int) strtol(idx,&endptr,10);
            if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
            {
                exit_input_error(total+1);
                return ;
            }
            else
                inst_max_index = x[i].index;

            errno = 0;
            x[i].value = strtod(val,&endptr);
            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
            {
                exit_input_error(total+1);
                return ;
            }

            ++i;
        }
        x[i].index = -1;

        if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))
        {
            predict_label = svm_predict_probability(model,x,prob_estimates);
            fprintf(output,"%g",predict_label);
            for(j=0;j<nr_class;j++)
                fprintf(output," %g",prob_estimates[j]);
            fprintf(output,"\n");
        }
        else
        {
            predict_label = svm_predict(model,x);
            printf("predict label %lf:",predict_label);
            fprintf(output,"%g\n",predict_label);
        }

        if(predict_label == target_label)
            ++correct;
        error += (predict_label-target_label)*(predict_label-target_label);
        sump += predict_label;
        sumt += target_label;
        sumpp += predict_label*predict_label;
        sumtt += target_label*target_label;
        sumpt += predict_label*target_label;
        ++total;
    }
    if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
    {
        info("Mean squared error = %g (regression)\n",error/total);
        info("Squared correlation coefficient = %g (regression)\n",
                ((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
                ((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
            );
    }
    /*else
      info("Accuracy = %g%% (%d/%d) (classification)\n",
      (double)correct/total*100,correct,total);
      不计算Accuracy，不显示出来，因为label都是假的
      */
    if(predict_probability)
        free(prob_estimates);
}

int predictlr(FILE *input, FILE *output, struct svm_model * model)
{
    int lastresult=0;
    int correct = 0;
    int total = 0;
    double error = 0;
    double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

    int svm_type=svm_get_svm_type(model);
    int nr_class=svm_get_nr_class(model);
    double *prob_estimates=NULL;
    int j;

    if(predict_probability)
    {
        if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
            info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
        else
        {
            int *labels=(int *) malloc(nr_class*sizeof(int));
            svm_get_labels(model,labels);
            prob_estimates = (double *) malloc(nr_class*sizeof(double));
            fprintf(output,"labels");
            for(j=0;j<nr_class;j++)
                fprintf(output," %d",labels[j]);
            fprintf(output,"\n");
            free(labels);
        }
    }

    max_line_len = 1024;
    line = (char *)malloc(max_line_len*sizeof(char));
    while(readline(input) != NULL)
    {
        int i = 0;
        double target_label, predict_label;
        char *idx, *val, *label, *endptr;
        int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

        label = strtok(line," \t\n");
        if(label == NULL) // empty line
        {
            exit_input_error(total+1);
            return 0;
        }

        target_label = strtod(label,&endptr);
        if(endptr == label || *endptr != '\0')
        {
            exit_input_error(total+1);
            return 0;
        }

        while(1)
        {
            if(i>=max_nr_attr-1)	// need one more for index = -1
            {
                max_nr_attr *= 2;
                x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
            }

            idx = strtok(NULL,":");
            val = strtok(NULL," \t");

            if(val == NULL)
                break;
            errno = 0;
            x[i].index = (int) strtol(idx,&endptr,10);
            if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
            {
                exit_input_error(total+1);
                return 0;
            }
            else
                inst_max_index = x[i].index;

            errno = 0;
            x[i].value = strtod(val,&endptr);
            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
            {
                exit_input_error(total+1);
                return 0;
            }

            ++i;
        }
        x[i].index = -1;

        if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))
        {
            predict_label = svm_predict_probability(model,x,prob_estimates);
            lastresult=predict_label;
            //printf("%d \n",lastresult);
            fprintf(output,"%g",predict_label);
            for(j=0;j<nr_class;j++)
                fprintf(output," %g",prob_estimates[j]);
            fprintf(output,"\n");
        }
        else
        {
            predict_label = svm_predict(model,x);
            lastresult=predict_label;
            //printf("%d \n",lastresult);
            fprintf(output,"%g\n",predict_label);
        }

        if(predict_label == target_label)
            ++correct;
        error += (predict_label-target_label)*(predict_label-target_label);
        sump += predict_label;
        sumt += target_label;
        sumpp += predict_label*predict_label;
        sumtt += target_label*target_label;
        sumpt += predict_label*target_label;
        ++total;
    }
    if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
    {
        info("Mean squared error = %g (regression)\n",error/total);
        info("Squared correlation coefficient = %g (regression)\n",
                ((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
                ((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
            );
    }
    /*else
      info("Accuracy = %g%% (%d/%d) (classification)\n",
      (double)correct/total*100,correct,total);
      不计算Accuracy，不显示出来，因为label都是假的
      */
    if(predict_probability)
        free(prob_estimates);
    return lastresult;
}

void predict_pro(FILE *input, FILE *output, struct svm_model * model, double *prob_estimates)
{
    int correct = 0;
    int total = 0;
    double error = 0;
    double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

    int svm_type=svm_get_svm_type(model);
    int nr_class=svm_get_nr_class(model);
    //double *prob_estimates=NULL;
    int j;

    if(predict_probability)
    {
        if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
            info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
        else
        {
            int *labels=(int *) malloc(nr_class*sizeof(int));
            svm_get_labels(model,labels);
            //prob_estimates = (double *) malloc(nr_class*sizeof(double));
            fprintf(output,"labels");
            for(j=0;j<nr_class;j++)
                fprintf(output," %d",labels[j]);
            fprintf(output,"\n");
            free(labels);
        }
    }

    max_line_len = 1024;
    line = (char *)malloc(max_line_len*sizeof(char));
    while(readline(input) != NULL)
    {
        int i = 0;
        double target_label, predict_label;
        char *idx, *val, *label, *endptr;
        int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

        label = strtok(line," \t\n");
        if(label == NULL) // empty line
        {
            exit_input_error(total+1);
            return ;
        }

        target_label = strtod(label,&endptr);
        if(endptr == label || *endptr != '\0')
        {
            exit_input_error(total+1);
            return ;
        }

        while(1)
        {
            if(i>=max_nr_attr-1)	// need one more for index = -1
            {
                max_nr_attr *= 2;
                x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
            }

            idx = strtok(NULL,":");
            val = strtok(NULL," \t");

            if(val == NULL)
                break;
            errno = 0;
            x[i].index = (int) strtol(idx,&endptr,10);
            if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
            {
                exit_input_error(total+1);
                return ;
            }
            else
                inst_max_index = x[i].index;

            errno = 0;
            x[i].value = strtod(val,&endptr);
            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
            {
                exit_input_error(total+1);
                return ;
            }

            ++i;
        }
        x[i].index = -1;

        if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))
        {
            predict_label = svm_predict_probability(model,x,prob_estimates);
            fprintf(output,"%g",predict_label);
            for(j=0;j<nr_class;j++)
                fprintf(output," %g",prob_estimates[j]);
            fprintf(output,"\n");
        }
        else
        {
            predict_label = svm_predict(model,x);
            printf("predict label %lf:",predict_label);
            fprintf(output,"%g\n",predict_label);
        }

        if(predict_label == target_label)
            ++correct;
        error += (predict_label-target_label)*(predict_label-target_label);
        sump += predict_label;
        sumt += target_label;
        sumpp += predict_label*predict_label;
        sumtt += target_label*target_label;
        sumpt += predict_label*target_label;
        ++total;
    }
    if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
    {
        info("Mean squared error = %g (regression)\n",error/total);
        info("Squared correlation coefficient = %g (regression)\n",
                ((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
                ((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
            );
    }
    /*else
      info("Accuracy = %g%% (%d/%d) (classification)\n",
      (double)correct/total*100,correct,total);
      不计算Accuracy，不显示出来，因为label都是假的
      */
    if(predict_probability)
        free(prob_estimates);
}

void exit_with_help()
{
    printf(
            "Usage: svm-predict [options] test_file model_file output_file\n"
            "options:\n"
            "-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n"
            "-q : quiet mode (no outputs)\n"
          );
    //exit(1);
    return ;
}

void mypredict(const char * mytopredictfile_one, const char * mymodelfile, const char * myresult, struct svm_model* model)
{
    FILE *input, *output;
    // parse options

    input = fopen(mytopredictfile_one,"r");
    if(input == NULL)
    {
        fprintf(stderr,"can't open input file %s\n",mytopredictfile_one);
        //exit(1);
        return ;
    }

    output = fopen(myresult,"w");
    if(output == NULL)
    {
        fprintf(stderr,"can't open output file %s\n",myresult);
        //exit(1);
        return ;
    }
    /*
       if((model=svm_load_model(mymodelfile))==0)
       {
       fprintf(stderr,"can't open model file %s\n",mymodelfile);
       exit(1);
       }*/

    x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));
    /*
       if(predict_probability)
       {
       if(svm_check_probability_model(model)==0)
       {
       fprintf(stderr,"Model does not support probabiliy estimates\n");
       return ;
    //exit(1);
    }
    }
    else
    {
    if(svm_check_probability_model(model)!=0)
    info("Model supports probability estimates, but disabled in prediction.\n");
    }*/
    /*int predictlabel=0;
    //predict(input,output);
    predict(input,output,predictlabel);
    printf("predict label:%d",predictlabel);*/

    //svm_free_and_destroy_model(&model);
    free(x);
    free(line);
    fclose(input);
    fclose(output);
}

int mypredictlr(const char * mytopredictfile_one, const char * mymodelfile, const char * myresult, struct svm_model* model)
{
    FILE *input, *output;
    // parse options
    int predictlabel=0;

    input = fopen(mytopredictfile_one,"r");
    if(input == NULL)
    {
        fprintf(stderr,"can't open input file %s\n",mytopredictfile_one);
        return 0;
        //exit(1);
    }

    output = fopen(myresult,"w");
    if(output == NULL)
    {
        fprintf(stderr,"can't open output file %s\n",myresult);
        return 0;
        //exit(1);
    }
    /*
       if((model=svm_load_model(mymodelfile))==0)
       {
       fprintf(stderr,"can't open model file %s\n",mymodelfile);
       exit(1);
       }*/

    x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));
    /*
       if(predict_probability)
       {
       if(svm_check_probability_model(model)==0)
       {
       fprintf(stderr,"Model does not support probabiliy estimates\n");
       return 0;
    //exit(1);
    }
    }
    else
    {
    if(svm_check_probability_model(model)!=0)
    info("Model supports probability estimates, but disabled in prediction.\n");
    }*/

    //predict(input,output);

    predictlabel=predictlr(input,output,model);
    //printf("predict label:%d",predictlabel);

    //svm_free_and_destroy_model(&model);
    free(x);
    free(line);
    fclose(input);
    fclose(output);
    return predictlabel;
}

void mypredict_probability(const char * mytopredictfile_one, const char * mymodelfile, const char * myresult, struct svm_model* model,double *prob_estimates)
{
    FILE *input, *output;
    // parse options
    int predictlabel=0;
    predict_probability=1;

    input = fopen(mytopredictfile_one,"r");
    if(input == NULL)
    {
        fprintf(stderr,"can't open input file %s\n",mytopredictfile_one);
        return ;
        //exit(1);
    }

    output = fopen(myresult,"w");
    if(output == NULL)
    {
        fprintf(stderr,"can't open output file %s\n",myresult);
        return ;
        //exit(1);
    }
    /*
       if((model=svm_load_model(mymodelfile))==0)
       {
       fprintf(stderr,"can't open model file %s\n",mymodelfile);
       exit(1);
       }*/

    x = (struct svm_node *) malloc(max_nr_attr*sizeof(struct svm_node));
    if(predict_probability)
    {
        if(svm_check_probability_model(model)==0)
        {
            fprintf(stderr,"Model does not support probabiliy estimates\n");
            return ;
            //exit(1);
        }
    }
    else
    {
        if(svm_check_probability_model(model)!=0)
            info("Model supports probability estimates, but disabled in prediction.\n");
    }

    //predict(input,output);

    predict_pro(input,output,model,prob_estimates);
    //printf("predict label:%d",predictlabel);

    //svm_free_and_destroy_model(&model);
    free(x);
    free(line);
    fclose(input);
    fclose(output);
    return ;
}
