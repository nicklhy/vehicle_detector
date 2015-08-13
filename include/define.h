#ifndef _DEFINE_H_
#define _DEFINE_H_

typedef struct CarLocation
{
	int index;
	int x;
	int y;
	int width;
	int height;
}CarLocation;

typedef struct CarAttr
{
	int index;//Í¼ÏñÐòºÅ
	char color[128]; //ÑÕÉ«
	char type[8]; //³µÐÍ
}CarAttr;

#endif