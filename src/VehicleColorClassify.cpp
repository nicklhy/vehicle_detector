#include "VehicleColorClassify.h"

const char VehicleColorClassify::color_map[8][16] = {"Black", "Blue", "Green", "Orange", "Red", "Gray", "White", "Yellow"};

VehicleColorClassify::VehicleColorClassify(void) {
    svmpath="VehicleColorSVM";
    modelfile="car_color_detect.model";
    topredictfile_one="topredictoneimage.txt";
    topredictfile_batch="topredictbatchimages.txt";
    histogram="histogram.txt";
    resulttxt="result.txt";
    join="/";
    std::string paramsb = svmpath+join+modelfile;
    // char paramsb[200];
    // memset(paramsb,0,sizeof(paramsb));
    // sprintf(paramsb,svmpath);
    // strcat(paramsb,join);
    // strcat(paramsb,modelfile);
    assert((mymodel=svm_load_model(paramsb.c_str()))!=0);
}

VehicleColorClassify::VehicleColorClassify(const char * mysvmpath, const char * mymodelfile, const char * mytopredictfile_one, const char * mytopredictfile_batch, const char * myhistogram, const char * myresult)
{
    join = "/";
    svmpath = mysvmpath;
    modelfile = mymodelfile;
    topredictfile_one = mytopredictfile_one;
    topredictfile_batch = mytopredictfile_batch;
    histogram = myhistogram;
    resulttxt = myresult;
    std::string paramsb = svmpath+join+modelfile;
    // strcpy(svmpath,mysvmpath);
    // strcpy(modelfile,mymodelfile);
    // strcpy(topredictfile_one,mytopredictfile_one);
    // strcpy(topredictfile_one,mytopredictfile_one);
    // strcpy(topredictfile_batch,mytopredictfile_batch);
    // strcpy(resulttxt,myresult);
    // char paramsb[200];
    // memset(paramsb,0,sizeof(paramsb));
    // sprintf(paramsb,svmpath);
    // strcat(paramsb,join);
    // strcat(paramsb,modelfile);
    assert((mymodel=svm_load_model(paramsb.c_str()))!=0);
}

VehicleColorClassify::~VehicleColorClassify(void)
{
    svm_free_and_destroy_model(&mymodel);
}

void VehicleColorClassify::SetPredictContent(toPredictContent* topreContent, const char * filepath, int label, int l, int t, int r, int b, toPredictContent * next)
{
    memset(topreContent->filepath,0,sizeof(topreContent->filepath));
    sprintf(topreContent->filepath,filepath);
    topreContent->label=label;
    topreContent->l=l;
    topreContent->t=t;
    topreContent->r=r;
    topreContent->b=b;
    topreContent->next=next;

}
int VehicleColorClassify::OneImageVehicleColorClassify(const char * toclassifyimage, int l, int t, int r, int b)
{
    return OneImageVehicleColorClassify(toclassifyimage,0,l,t,r,b);
}
int VehicleColorClassify::OneImageVehicleColorClassify(const char * toclassifyimage, int label, int l, int t, int r, int b)
{
    IplImage *image=cvLoadImage(toclassifyimage,1);
    topredictFile(image,label,l,t,r,b);
    int predictlabel=0;

    std::string paramsa, paramsb, paramsc;
    paramsa = svmpath+join+topredictfile_one;
    paramsb = svmpath+join+modelfile;
    paramsc = svmpath+join+resulttxt;


    // char paramsa[200];
    // memset(paramsa,0,sizeof(paramsa));
    // char paramsb[200];
    // memset(paramsb,0,sizeof(paramsb));
    // char paramsc[200];
    // memset(paramsc,0,sizeof(paramsc));
    // sprintf(paramsa,svmpath);
    // strcat(paramsa,join);
    // strcat(paramsa,topredictfile_one);
    // sprintf(paramsb,svmpath);
    // strcat(paramsb,join);
    // strcat(paramsb,modelfile);
    // sprintf(paramsc,svmpath);
    // strcat(paramsc,join);
    // strcat(paramsc,resulttxt);

    predictlabel=mypredictlr(paramsa.c_str(), paramsb.c_str(), paramsc.c_str(), mymodel);
    //printf("predict label:%d \n",predictlabel);
    cvReleaseImage(&image);
    return predictlabel;

}
int VehicleColorClassify::OneImageVehicleColorClassify(IplImage * topredictimage, int l, int t, int r, int b)
{
    return OneImageVehicleColorClassify(topredictimage, 0, l, t, r, b);
}
int VehicleColorClassify::OneImageVehicleColorClassify(IplImage * topredictimage, int label, int l, int t, int r, int b)
{
    topredictFile(topredictimage,label,l,t,r,b);
    int predictlabel=0;

    std::string paramsa, paramsb, paramsc;
    paramsa = svmpath+join+topredictfile_one;
    paramsb = svmpath+join+modelfile;
    paramsc = svmpath+join+resulttxt;

    // char paramsa[200];
    // memset(paramsa,0,sizeof(paramsa));
    // char paramsb[200];
    // memset(paramsb,0,sizeof(paramsb));
    // char paramsc[200];
    // memset(paramsc,0,sizeof(paramsc));
    // sprintf(paramsa,svmpath);
    // strcat(paramsa,join);
    // strcat(paramsa,topredictfile_one);
    // sprintf(paramsb,svmpath);
    // strcat(paramsb,join);
    // strcat(paramsb,modelfile);
    // sprintf(paramsc,svmpath);
    // strcat(paramsc,join);
    // strcat(paramsc,resulttxt);

    predictlabel=mypredictlr(paramsa.c_str(), paramsb.c_str(), paramsc.c_str(), mymodel);
    //printf("predict label:%d \n",predictlabel);

    return predictlabel;

}

int VehicleColorClassify::OneImageVehicleColorClassify(toPredictContent * topreContent)
{
    IplImage *image=cvLoadImage(topreContent->filepath,1);
    topredictFile(image,topreContent->label,topreContent->l,topreContent->t,topreContent->r,topreContent->b);
    int predictlabel=0;

    std::string paramsa, paramsb, paramsc;
    paramsa = svmpath+join+topredictfile_one;
    paramsb = svmpath+join+modelfile;
    paramsc = svmpath+join+resulttxt;
    // char paramsa[200];
    // memset(paramsa,0,sizeof(paramsa));
    // char paramsb[200];
    // memset(paramsb,0,sizeof(paramsb));
    // char paramsc[200];
    // memset(paramsc,0,sizeof(paramsc));
    // sprintf(paramsa,svmpath);
    // strcat(paramsa,join);
    // strcat(paramsa,topredictfile_one);
    // sprintf(paramsb,svmpath);
    // strcat(paramsb,join);
    // strcat(paramsb,modelfile);
    // sprintf(paramsc,svmpath);
    // strcat(paramsc,join);
    // strcat(paramsc,resulttxt);

    predictlabel=mypredictlr(paramsa.c_str(), paramsb.c_str(), paramsc.c_str(), mymodel);
    //printf("predict label:%d \n",predictlabel);
    return predictlabel;
}
void VehicleColorClassify::OneImageVehicleColorClassify(IplImage * topredictimage, int l, int t, int r, int b,double *prob_estimates)
{
    OneImageVehicleColorClassify(topredictimage, 0, l, t, r, b,prob_estimates);
}
void VehicleColorClassify::OneImageVehicleColorClassify(IplImage * topredictimage, int label, int l, int t, int r, int b,double *prob_estimates)
{
    topredictFile(topredictimage,label,l,t,r,b);
    int predictlabel=0;

    std::string paramsa, paramsb, paramsc;
    paramsa = svmpath+join+topredictfile_one;
    paramsb = svmpath+join+modelfile;
    paramsc = svmpath+join+resulttxt;
    // char paramsa[200];
    // memset(paramsa,0,sizeof(paramsa));
    // char paramsb[200];
    // memset(paramsb,0,sizeof(paramsb));
    // char paramsc[200];
    // memset(paramsc,0,sizeof(paramsc));
    // sprintf(paramsa,svmpath);
    // strcat(paramsa,join);
    // strcat(paramsa,topredictfile_one);
    // sprintf(paramsb,svmpath);
    // strcat(paramsb,join);
    // strcat(paramsb,modelfile);
    // sprintf(paramsc,svmpath);
    // strcat(paramsc,join);
    // strcat(paramsc,resulttxt);

    mypredict_probability(paramsa.c_str(), paramsb.c_str(), paramsc.c_str(), mymodel, prob_estimates);
}

void VehicleColorClassify::OneImageVehicleColorClassify(const char * topredictfile, int l, int t, int r, int b,double *prob_estimates)
{
    OneImageVehicleColorClassify(topredictfile, 0, l, t, r, b,prob_estimates);
}
void VehicleColorClassify::OneImageVehicleColorClassify(const char * topredictfile, int label, int l, int t, int r, int b,double *prob_estimates)
{
    IplImage *image=cvLoadImage(topredictfile,1);
    topredictFile(image,label,l,t,r,b);
    int predictlabel=0;

    std::string paramsa, paramsb, paramsc;
    paramsa = svmpath+join+topredictfile_one;
    paramsb = svmpath+join+modelfile;
    paramsc = svmpath+join+resulttxt;
    // char paramsa[200];
    // memset(paramsa,0,sizeof(paramsa));
    // char paramsb[200];
    // memset(paramsb,0,sizeof(paramsb));
    // char paramsc[200];
    // memset(paramsc,0,sizeof(paramsc));
    // sprintf(paramsa,svmpath);
    // strcat(paramsa,join);
    // strcat(paramsa,topredictfile_one);
    // sprintf(paramsb,svmpath);
    // strcat(paramsb,join);
    // strcat(paramsb,modelfile);
    // sprintf(paramsc,svmpath);
    // strcat(paramsc,join);
    // strcat(paramsc,resulttxt);

    mypredict_probability(paramsa.c_str(), paramsb.c_str(), paramsc.c_str(), mymodel,prob_estimates);
    //printf("predict label:%d \n",predictlabel);

    return ;

}

int VehicleColorClassify::topredictFile(IplImage *image, int label, int l, int t, int r, int b)
{
    CvScalar s;

    int rr=0,gg=0,bb=0;
    int index=0;
    //int mycount=0;
    // double max = 0;
    double **his = new double*[NUM];

    double line[CLASS_NUM-2];
    double sumline=0;

    std::string histogrampath, topredictsvmfilepath;
    histogrampath = svmpath+join+histogram;
    topredictsvmfilepath = svmpath+join+topredictfile_one;
    // char *histogrampath=new char[200];
    // char *topredictsvmfilepath=new char[200];
    // strcpy(histogrampath,svmpath);
    // strcat(histogrampath,join);
    // strcat(histogrampath,histogram);
    // strcpy(topredictsvmfilepath,svmpath);
    // strcat(topredictsvmfilepath,join);
    // strcat(topredictsvmfilepath,topredictfile_one);

    std::ifstream fin(histogrampath);
    std::ofstream fout(topredictsvmfilepath);
    if(!fin){
        // cout << "Unable to open myfile";
        //exit(1); // terminate with error
        return -1;
    }
    if(!fout){
        // cout << "Unable to open otfile";
        //exit(1); // terminate with error
        return -1;
    }
    std::string ss;
    char *sss = new char[50];
    int count = 0;
    int m=0;
    int n=0;
    double norm=0;
    for(int k=0;k<CLASS_NUM-2;k++)
    {
        line[k]=0.0;
    }
    for(int i=0;i<NUM;i++)
    {
        his[i]= new double[CLASS_NUM];
        for(int j=0;j<CLASS_NUM;j++)
            his[i][j]=0.0;
    }

    while(fin >> ss )
    {
        double db1=0;
        ss.copy(sss,18,0);
        //ss.copy(sss,8,0);
        //histogramÖÐ±£ÁôÁË6Î»Ð¡Êý£¬ËãÉÏÕûÊýÎ»ºÍÐ¡ÊýµãÊÇ8
        db1 = atof(sss);
        //cout << count << ":" << db1 <<endl;
        m=count/CLASS_NUM;
        n=count%CLASS_NUM;
        //cout << "m" << m << "n" << n <<endl;
        his[m][n]=db1;
        //cout << his[m][n] <<endl;
        count++;
    }
    if((r>image->width)||(b>image->height))
    {
        // cout<<"Out Of Range!"<<endl;
        //exit(1);
        return -1;
    }
    for(int i=0;i<image->height;i++)
    {
        for(int j=0;j<image->width;j++)
        {
            s=cvGet2D(image,i,j);
            bb=(int)s.val[0];
            gg=(int)s.val[1];
            rr=(int)s.val[2];
            //printf("B=%d, G=%d, R=%d \n",rr,gg,bb);
            cvSet2D(image,i,j,s);//set the (i,j) pixel value

            if(j>=l-1&&j<=r-1&&i>=t-1&&i<=b-1)
            {
                index=(rr/8)+32*(gg/8)+32*32*(bb/8);

                for(int k=0;k<CLASS_NUM-2;k++)
                {
                    //if(mycount<100)
                    //	printf("%d %d %f\n",index,k,his[index][k]);
                    line[k]+=his[index][k];

                }
                //mycount++;
            }
        }
    }
    //printf("\n\n%d\n",mycount);

    for(int k=0;k<CLASS_NUM-2;k++)
    {
        double a=line[k]*line[k];
        sumline+=a;
    }
    //È¥µô9ºÍ10µÄ²£Á§ºÍµØÃæ
    norm=sqrt(sumline);
    char str[10];
    sprintf(str,"%d",label);
    strcat(str," ");
    fout<< str;
    for(int k=0;k<CLASS_NUM-2;k++)
    {
        line[k]=line[k]/norm;
        int i=k+1;
        fout<<i<<":"<<line[k]<<" ";
    }

    for(int i=0;i<NUM;i++)
    {
        delete[] his[i];
    }
    // delete[] histogrampath;
    // delete[] topredictsvmfilepath;
    delete[] his;
    return 0;
}

void VehicleColorClassify::topredictFileData(IplImage *image,int l,int t, int r, int b, double* line)
{
    CvScalar s;

    int rr=0,gg=0,bb=0;
    int index=0;
    // double max = 0;
    double **his = new double*[NUM];

    double sumline=0;

    std::string histogrampath;
    histogrampath = svmpath+join+histogram;
    // char *histogrampath=new char[200];
    // char *topredictsvmfilepath=new char[200];
    // strcpy(histogrampath,svmpath);
    // strcat(histogrampath,join);
    // strcat(histogrampath,histogram);

    ifstream fin(histogrampath);
    assert(fin);

    string ss;
    char *sss = new char[50];
    int count = 0;
    int m=0;
    int n=0;
    double norm=0;
    for(int k=0;k<CLASS_NUM-2;k++)
    {
        line[k]=0.0;
    }
    for(int i=0;i<NUM;i++)
    {
        his[i]= new double[CLASS_NUM];
        for(int j=0;j<CLASS_NUM;j++)
            his[i][j]=0.0;
    }

    while(fin >> ss )
    {
        double db1=0;
        ss.copy(sss,18,0);
        //ss.copy(sss,8,0);
        db1 = atof(sss);
        m=count/CLASS_NUM;
        n=count%CLASS_NUM;
        his[m][n]=db1;
        count++;
    }
    assert((r<=image->width)&&(b<=image->height));
    for(int i=0;i<image->height;i++)
    {
        for(int j=0;j<image->width;j++)
        {
            s=cvGet2D(image,i,j); // »ñÈ¡ÏñËØÖµ
            bb=(int)s.val[0];
            gg=(int)s.val[1];
            rr=(int)s.val[2];
            //printf("B=%d, G=%d, R=%d \n",rr,gg,bb);
            cvSet2D(image,i,j,s);//set the (i,j) pixel value

            if(j>=l-1&&j<=r-1&&i>=t-1&&i<=b-1)
            {
                index=(rr/8)+32*(gg/8)+32*32*(bb/8); //¼ÆËãÏñËØ¶ÔÓ¦µÄpixel

                for(int k=0;k<CLASS_NUM-2;k++)
                {
                    line[k]+=his[index][k];
                }
            }
        }
    }

    for(int k=0;k<CLASS_NUM-2;k++)
    {
        double a=line[k]*line[k];
        sumline+=a;
    }
    //È¥µô9ºÍ10µÄ²£Á§ºÍµØÃæ
    norm=sqrt(sumline);

    for(int k=0;k<CLASS_NUM-2;k++)
    {
        line[k]=line[k]/norm;
        // int i=k+1;
    }

    for(int i=0;i<NUM;i++)
    {
        delete[] his[i];
    }
    // delete[] histogrampath;
    // delete[] topredictsvmfilepath;
    delete[] his;
}
void VehicleColorClassify::topredictFiles(toPredictContent * topreContents)
{
    std::string topredictsvmfilepath = svmpath+join+topredictfile_batch;
    // char *topredictsvmfilepath=new char[200];
    // strcpy(topredictsvmfilepath,svmpath);
    // strcat(topredictsvmfilepath,join);
    // strcat(topredictsvmfilepath,topredictfile_batch);
    ofstream fout(topredictsvmfilepath);

    if(!fout){
        cout << "Unable to open otfile";
        //exit(1); // terminate with error
        return ;
    }

    int FileNums=0;
    toPredictContent * head = topreContents;
    toPredictContent * topre = topreContents;
    while(topre->next!=NULL)
    {
        topre=topre->next;
        FileNums++;
    }
    FileNums++;
    double** lines = new double*[FileNums];
    topre=head;
    for(int i=0;i<FileNums;i++)
    {
        IplImage *image = cvLoadImage(topre->filepath);
        lines[i]=new double[CLASS_NUM-2];
        topredictFileData(image,topre->l,topre->t, topre->r, topre->b, lines[i]);
        /*for(int j=0;j<CLASS_NUM-2;j++)
          {
          cout<<lines[i][j]<<endl;
          }*/
        char str[10];
        sprintf(str,"%d",topre->label);
        strcat(str," ");
        fout<< str;
        for(int j=0;j<CLASS_NUM-2;j++)
        {
            fout<<j+1<<":"<<lines[i][j]<<" ";
        }
        fout<<endl;
        cout<<" "<<endl;
        topre=topre->next;
        cvReleaseImage(&image);
    }
}

void VehicleColorClassify::BatchImagesVehicleColorClassify(toPredictContent* topreContents)
{
    topredictFiles(topreContents);

    std::string paramsa, paramsb, paramsc;
    paramsa = svmpath+join+topredictfile_batch;
    paramsb = svmpath+join+modelfile;
    paramsc = svmpath+join+resulttxt;

    // char paramsa[200];
    // memset(paramsa,0,sizeof(paramsa));
    // char paramsb[200];
    // memset(paramsb,0,sizeof(paramsb));
    // char paramsc[200];
    // memset(paramsc,0,sizeof(paramsc));
    // sprintf(paramsa,svmpath);
    // strcat(paramsa,join);
    // strcat(paramsa,topredictfile_batch);
    // sprintf(paramsb,svmpath);
    // strcat(paramsb,join);
    // strcat(paramsb,modelfile);
    // sprintf(paramsc,svmpath);
    // strcat(paramsc,join);
    // strcat(paramsc,resulttxt);

    mypredict(paramsa.c_str(), paramsb.c_str(), paramsc.c_str(), mymodel);
}
