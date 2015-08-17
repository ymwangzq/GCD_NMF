#include <iostream>
#include <Windows.h>
#include <cblas.h>
#include <fstream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include "matrix.h"

using namespace std;
using namespace cv;

typedef void(WINAPI* _cblas_dgemm)(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
	OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST double *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST double beta, double *C, OPENBLAS_CONST blasint ldc);

int main()
{
	HMODULE _cblas = NULL;
	_cblas = LoadLibrary(TEXT("libopenblas.dll"));
	if (_cblas == NULL)
	{
		cout << "LoadLibrary Failed! Err code : " << GetLastError() << endl;
	}
	else
		cout << "LoadLibrary sussessed!" << endl;
	double A[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
	double B[6] = { 1, 0, 1, 0, 1, 1 };
	double C[12] = { 0 };
	int m = 4, k = 2, n = 3;
	cout << "matrix A:\n";
	for (int i = 0; i < m*k; i++)
	{
		if (i % k == 0 && i != 0)
			cout << endl;
		cout << A[i] << ",\t";
	}
	cout << "\n\nmatrix B:\n";
	for (int i = 0; i < k*n; i++)
	{
		if (i % n == 0 && i != 0)
			cout << endl;
		cout << B[i] << ",\t";
	}
	cout << endl;
	_cblas_dgemm cblas_dgemm = (_cblas_dgemm)GetProcAddress(_cblas, TEXT("cblas_dgemm"));
	//cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 1, 1, 1, 1, C, 1, C, 1, 1, C, 1);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, k, B, n, 0, C, n);
	cout << "\nmatrix C = A*B:\n";
	for (int i = 0; i < m*n; i++)
	{
		if (i % n == 0 && i != 0)
			cout << endl;
		cout << C[i] << ",\t";
	}
	cout << endl << endl;

	for (int i = 0; i < m * n; i++)
	{
		C[i] = 0;
	}

	_Matrix mA(m, k, A);
	_Matrix mB(k, n, B);

	cout << "mA" << endl;
	mA.show();
	cout << "mB" << endl;
	mB.show();

	/*
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mA.getd1(), mB.getd2(), mA.getd2(),\
		1, mA.getbuf(), mA.getd2(), mB.getbuf(), mB.getd2(), 0, C, n);

	const float* tmp = mA.getbuf();
	tmp++;
	//tmp[0] = 0;
	*/
	
	_Matrix mC;
	mC = mA*mB;
	cout << "mC" << endl;
	mC.show();
	_Matrix mD = mA*(double)4.157+mA;
	cout << "mD" << endl;
	mD.show();
	mD = mD.dotDivision(mA);
	cout << "mD" << endl;
	mD.show();
	mD = -mA;
	cout << "mD" << endl;
	mD.show();
	mD = mA * 4 - mA;
	cout << "mD" << endl;
	mD.show();
	mD = mA.dotProduct(mA);
	cout << "mD" << endl;
	mD.show();
	_Matrix mE = mD.transpose();
	cout << "mE" << endl;
	mE.show();

	double w[] = { 3, 6, 4, 8 };
	normalize_w(w, 2, 2);

	_Matrix W;
	_Matrix H;
	double c[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
	mC = _Matrix(4, 4, c);
	cout << "mC" << endl;
	mC.show();
	mC.NMF(4, 50, W, H);
	cout << "W" << endl;
	W.show();
	cout << "H" << endl;
	H.show();
	_Matrix R = W * H;
	cout << "R" << endl;
	R.show();
	cout << "Rmax = " << R.getmax() << endl;
	cout << "Rmin = " << R.getmin() << endl;

	mD = mC.getCol(0);
	cout << "mD" << endl;
	mD.cat(mD,0).show();
	mD.show();
	mD.cat(mD, 1).show();
	mD.show();

	int vd1 = 400;
	int vd2 = 400;
	double* v = new double[vd1 * vd2];
	srand(GetTickCount());
	parallel_for(0, vd1*vd2, [&](int i){
		v[i] = ((int)rand()) % 256;
		if (v[i] < 0)
			v[i] = -v[i];
	});
	_Matrix V(vd1, vd2, v);
	V.NMF(10, 50, W, H);
	R = W * H;
	_Matrix Err = V - R;
	cout << "ERRmax = " << Err.getmax() << endl;
	cout << "ERRmin = " << Err.getmin() << endl;

	Mat imgtmp = imread("1.JPG", 1);
	namedWindow("test", CV_WINDOW_NORMAL);
	imshow("test", imgtmp);
	waitKey();
	destroyWindow("test");

	uchar* p = imgtmp.data;
	cout << imgtmp.total()*imgtmp.channels() << endl;
	if (imgtmp.isContinuous())
		cout << "isContinuous" << endl;
	else
		cout << "not isContinuous" << endl;
	MatIterator_<Vec3b> it, begin, end;
	begin = imgtmp.begin<Vec3b>();
	end = imgtmp.end<Vec3b>();
	/*
	for (int i = 0; i < imgtmp.total()*imgtmp.channels(); i += imgtmp.channels())
	{
		cout << (int)p[i] << "\t" << (int)p[i + 1] << "\t" << (int)p[i + 2] << endl;
		getchar();
	}
	for (it = begin; it < end; it++)
	{
		cout << (int)it[0][0] << "\t" << (int)it[0][1] << "\t" << (int)it[0][2] << endl;
		getchar();
	}
	*/
	WIN32_FIND_DATA FindFileData;
	HANDLE hFind;

	hFind = FindFirstFile(TEXT("C:\\test\\*"), &FindFileData);
	while (hFind != INVALID_HANDLE_VALUE)
	{
		if (!((FindFileData.cFileName[0] == '.' && FindFileData.cFileName[1] == 0) || \
			(FindFileData.cFileName[0] == '.' && FindFileData.cFileName[1] == '.' && FindFileData.cFileName[2] == 0)))
		{
			cout << FindFileData.cFileName << endl;
		}
		if (!FindNextFile(hFind, &FindFileData))
		{
			FindClose(hFind);
			hFind = INVALID_HANDLE_VALUE;
		}
	}

	getchar();
	return 0;
}