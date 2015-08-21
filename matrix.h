#include <iostream>
#include <Windows.h>
#include <iomanip>
#include <cblas.h>
#include <math.h>
#include <ppl.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;
using namespace concurrency;

typedef void(WINAPI* _cblas_dgemm)(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
	OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST double *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST double beta, double *C, OPENBLAS_CONST blasint ldc);

class _Matrix
{
public:
	int MatChannels;
	int MatType;

	_Matrix();
	_Matrix(int _d1, int _d2, double* _buf);
	_Matrix(const _Matrix& a);
	_Matrix& operator=(const _Matrix& a);
	~_Matrix();

	const double* getbuf();
	double* getbuf2set();

	void setbuf(double* _buf, int len);
	int getd1();
	int getd2();
	
	void show();

	void resize(int _d1, int _d2);
	void resize(int _d1, int _d2, double* _buf);
	bool reshapeToColWithRowFirst();
	bool reshape(int _d1, int _d2);

	double getmax();
	double getmin();

	_Matrix getCol(int c2);
	_Matrix cat(_Matrix b, bool d);

	_Matrix transpose();

	_Matrix operator*(_Matrix b);
	_Matrix operator*(double b);
	_Matrix operator+(_Matrix b);
	_Matrix operator-();
	_Matrix operator-(_Matrix b);

	_Matrix dotProduct(_Matrix b);
	_Matrix dotDivision(_Matrix b);

	void NMF(int rank, int iter, _Matrix& W, _Matrix& H);

	void setavilable(bool _avilable);
	bool isavilable();

private:
	static HMODULE _openBlas;
	static _cblas_dgemm cblas_dgemm;

	int d1;
	int d2;
	bool avilable;
	double* buf;
};

void normalize_w(double* W, int d1, int d2);
_Matrix mat2matrix(Mat M);
Mat matrix2mat(_Matrix M);

_Matrix getPicMat(LPCSTR path, Size& size);

void matrix2file(LPCSTR filename, _Matrix M);
void file2matrix(LPCSTR filename, _Matrix& M);