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
	int index;//ͼ�����
	char color[128]; //��ɫ
	char type[8]; //����
}CarAttr;

#endif