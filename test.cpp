#include <iostream>
#include <Windows.h>
#include <cblas.h>
#include <fstream>
#include <stdio.h>
#include "matrix.h"

using namespace std;
using namespace cv;

void test1();
void test2();
void test3();
void test4();

int main()
{
	test1();
	test2();
	test3();
	test4();

	cout << "ready" << endl;
	getchar();
	return 0;
}

void test1()
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
	_Matrix mD = mA*(double)4.157 + mA;
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
	mD.cat(mD, 0).show();
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

	string jpg1 = { "1.jpg" };
	Mat imgtmp = imread(jpg1, 1);
	namedWindow("test", CV_WINDOW_NORMAL);
	imshow("test", imgtmp);
	//waitKey();
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
	int row = imgtmp.rows;
	cout << row << endl;
	int col = imgtmp.cols;
	cout << col << endl;
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
	/*
	for (it = begin; it < end; it++)
	{
	cout << (int)it[0][0] << "\t" << (int)it[0][1] << "\t" << (int)it[0][2] << endl;
	char s = getchar();
	if (s == '0')
	break;
	}
	*/
	DWORD start = GetTickCount();
	_Matrix tmp = mat2matrix(imgtmp);
	DWORD time = GetTickCount() - start;
	cout << (int)time << endl;
	/*
	for (int i = 0; i < 9; i++)
	{
	if (i % 3 == 0)
	cout << endl;
	cout << (int)tmp.getbuf()[i] << "\t";
	}
	*/
	start = GetTickCount();
	Mat _imgtmp = matrix2mat(tmp);
	time = GetTickCount() - start;
	cout << (int)time << endl;
	namedWindow("test", CV_WINDOW_NORMAL);
	imshow("test", _imgtmp);
	//waitKey();
	destroyWindow("test");
	begin = imgtmp.begin<Vec3b>();
	end = imgtmp.end<Vec3b>();
	/*
	for (it = begin; it < end; it++)
	{
	cout << (int)it[0][0] << "\t" << (int)it[0][1] << "\t" << (int)it[0][2] << endl;
	char s = getchar();
	if (s == '0')
	break;
	}
	*/
	int i = 260;
	Size s = { i, 2 * i };
	cout << imgtmp.size() << endl;
	cout << imgtmp.type() << endl;

	imgtmp = imread("3.bmp", 0);
	cout << imgtmp.channels() << endl;
	namedWindow("test", CV_WINDOW_NORMAL);
	imshow("test", imgtmp);
	//waitKey();
	destroyWindow("test");
	start = GetTickCount();
	tmp = mat2matrix(imgtmp);
	time = GetTickCount() - start;
	cout << (int)time << endl;
}

void test2()
{
	char path[] = { "C:\\test\\*" };
	Size size;
	_Matrix test = getPicMat(path, size);
	_Matrix col = test.getCol(0);
	col.reshape(size.height, size.width * col.MatChannels);
	Mat tmp = matrix2mat(col);
	namedWindow("test", CV_WINDOW_NORMAL);
	imshow("test", tmp);
	//waitKey();
	destroyWindow("test");
}

void test3()
{
	char path[] = { "C:\\test\\*" };
	Size size;
	_Matrix test = getPicMat(path, size);
	_Matrix W, H;
	test.NMF(2, 20, W, H);
	_Matrix R = W * H;
	_Matrix Err = test - R;
	cout << "ERRmax = " << Err.getmax() << endl;
	cout << "ERRmin = " << Err.getmin() << endl;
	/*
	ofstream outfile("WH.nmf", ios::out | ios::binary);
	outfile << size.height << endl << size.width << endl;
	*/

	matrix2file("W.nmf", W);
	matrix2file("H.nmf", H);

	_Matrix Wr, Hr;
	file2matrix("W.nmf", Wr);
	file2matrix("H.nmf", Hr);

	_Matrix We, He;
	We = W - Wr;
	He = H - Hr;
	cout << "WEmax = " << We.getmax() << endl;
	cout << "WEmin = " << We.getmin() << endl;
	cout << "HEmax = " << He.getmax() << endl;
	cout << "HEmin = " << He.getmin() << endl;
}

void test4()
{
	string jpg1 = { "1.jpg" };
	Mat imgtmp = imread(jpg1, 1);
	namedWindow("test", CV_WINDOW_NORMAL);
	imshow("test", imgtmp);
	//waitKey();
	destroyWindow("test");

	_Matrix M = mat2matrix(imgtmp);
	matrix2file("M.nmf", M);
	_Matrix N;
	file2matrix("M.nmf", N);

	Mat img = matrix2mat(N);
	namedWindow("test", CV_WINDOW_NORMAL);
	imshow("test", imgtmp);
	//waitKey();
	destroyWindow("test");

	_Matrix E = M - N;

	cout << "DBL_MIN = " << DBL_MIN << endl;
	cout << "Emax = " << E.getmax() << endl;
	cout << "Emin = " << E.getmin() << endl;
}