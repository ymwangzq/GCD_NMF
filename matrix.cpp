#include "matrix.h"
#include <string>

double goWiter(double *GW, double *HH, double *W, double tol, int n, int k, int maxinner);

HMODULE _Matrix::_openBlas = LoadLibrary(TEXT("libopenblas.dll"));
_cblas_dgemm _Matrix::cblas_dgemm = (_cblas_dgemm)GetProcAddress(_Matrix::_openBlas, TEXT("cblas_dgemm"));

_Matrix::_Matrix():avilable(false), d1(0), d2(0), MatChannels(0){}

_Matrix::_Matrix(int _d1, int _d2, double* _buf)
	: d1(_d1), d2(_d2), avilable(false), MatChannels(0)
{
	buf = NULL;
	buf = new double[d1*d2];
	if (buf == NULL)
		cout << "Err in fMatrix!" << endl;
	if (_buf != NULL && 0 != memcpy_s(buf, d1 * d2 * sizeof(double), _buf, d1 * d2 * sizeof(double)))
		cout << "Err in memcpy!" << endl;
	else
		avilable = true;
}

_Matrix::_Matrix(const _Matrix& a)
{
	MatChannels = a.MatChannels;
	MatType = a.MatType;
	d1 = a.d1, d2 = a.d2;
	avilable = a.avilable;
	buf = NULL;
	buf = new double[d1*d2];
	if (buf == NULL)
		cout << "Err in fMatrix!" << endl;
	if (a.buf != NULL && 0 != memcpy_s(buf, d1 * d2 * sizeof(double), a.buf, d1 * d2 * sizeof(double)))
	{
		cout << "Err in memcpy!" << endl;
		avilable = false;
	}
}

_Matrix& _Matrix::operator=(const _Matrix& a)
{
	MatChannels = a.MatChannels;
	MatType = a.MatType;
	d1 = a.d1, d2 = a.d2;
	avilable = a.avilable;
	if (buf != NULL)
		delete buf;
	buf = NULL;
	buf = new double[d1*d2];
	if (buf == NULL)
		cout << "Err in fMatrix!" << endl;
	if (a.buf != NULL && 0 != memcpy_s(buf, d1 * d2 * sizeof(double), a.buf, d1 * d2 * sizeof(double)))
	{
		cout << "Err in memcpy!" << endl;
		avilable = false;
	}
	return *this;
}

_Matrix::~_Matrix()
{
	delete buf;
}

const double* _Matrix::getbuf()
{
	return buf;
}

void _Matrix::setbuf(double* _buf, int len)
{
	if (len != d1 * d2)
		cout << "Err in setbuf:size not match!" << endl;
	if (0 != memcpy_s(buf, len * sizeof(double), _buf, len * sizeof(double)))
		cout << "Err in memcpy!" << endl;
	else
		avilable = true;
}

int _Matrix::getd1(){	return d1;}

int _Matrix::getd2(){	return d2;}

void _Matrix::show()
{
	if (!avilable)
	{
		cout << "Err in show:matrix not avilable!" << endl;
		return;
	}
	for (int i = 0; i < d1 * d2; i++)
	{
		if (i != 0 && i % d2 == 0)
			cout << endl;
		cout << left << setw(10) << buf[i];
		if (i % d2 != d2 - 1)
			cout << ", ";
	}
	cout << endl << endl;
}

void _Matrix::resize(int _d1, int _d2)
{
	if (buf != NULL)
		delete buf;
	d1 = _d1, d2 = _d2;
	buf = new double[d1*d2];
	avilable = false;
}

void _Matrix::resize(int _d1, int _d2, double* _buf)
{
	if (buf != NULL)
		delete buf;
	d1 = _d1, d2 = _d2;
	avilable = false;
	buf = new double[d1*d2];
	if (buf == NULL)
		cout << "Err in fMatrix!" << endl;
	if (_buf != NULL && 0 != memcpy_s(buf, d1 * d2 * sizeof(double), _buf, d1 * d2 * sizeof(double)))
		cout << "Err in memcpy!" << endl;
	else
		avilable = true;
}

bool _Matrix::reshapeToColWithRowFirst()
{
	bool ret = false;
	if (!avilable)
	{
		cout << "Err in reshapeToColWithRowFirst:matrix not avilable!" << endl;
		return ret;
	}
	d1 = 1;
	d2 = d1 * d2;
	ret = true;
	return ret;
}

double _Matrix::getmax()
{
	double ret = DBL_MIN;
	combinable<double> Max([]{return DBL_MIN; });
	parallel_for(0, d1, [&](int i)
	{
		double* tmp = &buf[i*d2];
		for (int j = 0; j < d2; j++)
		{
			if (tmp[j] > Max.local())
				Max.local() = tmp[j];
		}
	});
	Max.combine_each([&](double local)
	{
		if (local > ret)
			ret = local;
	});
	return ret;
}

double _Matrix::getmin()
{
	double ret = DBL_MAX;
	combinable<double> Min([]{return DBL_MAX; });
	parallel_for(0, d1, [&](int i)
	{
		double* tmp = &buf[i*d2];
		for (int j = 0; j < d2; j++)
		{
			if (tmp[j] < Min.local())
				Min.local() = tmp[j];
		}
	});
	Min.combine_each([&](double local)
	{
		if (local < ret)
			ret = local;
	});
	return ret;
}

_Matrix _Matrix::getCol(int c2)
{
	_Matrix ret;
	ret.resize(d1, 1);
	double* newbuf = ret.getbuf2set();
	for (int i = 0; i < d1; i++)
	{
		newbuf[i] = buf[i * d2 + c2];
	}
	ret.setavilable(true);
	return ret;
}

_Matrix _Matrix::cat(_Matrix b, bool d)
{
	_Matrix ret;
	if (d == 0)
	{
		if (d2 != b.getd2())
			return ret;
		ret.resize(d1 + b.getd1(), d2);
		double* retbuf = ret.getbuf2set();
		memcpy_s(retbuf, d1*d2*sizeof(double), buf, d1*d2*sizeof(double));
		memcpy_s(&retbuf[d1*d2], b.getd1()*b.getd2()*sizeof(double), b.getbuf(), b.getd1()*b.getd2()*sizeof(double));
		ret.setavilable(true);
	}
	else
	{
		if (d1 != b.getd1())
			return ret;
		ret.resize(d1, d2 + b.getd2());
		double* retbuf = ret.getbuf2set();
		for (int i = 0; i < d1; i++)
		{
			for (int j = 0; j < d2 + b.getd2(); j++)
			{
				if (j < d2)
					retbuf[i*(d2 + b.getd2()) + j] = buf[i*d2 + j];
				else
					retbuf[i*(d2 + b.getd2()) + j] = b.getbuf()[i*b.getd2() + j - d2];
			}
		}
		ret.setavilable(true);
	}
	return ret;
}

_Matrix _Matrix::transpose()
{
	_Matrix ret;

	ret.resize(d2, d1);
	double* newbuf = ret.getbuf2set();
	parallel_for((int)0, d1, [=](int i)
	{
		for (int j = 0; j < d2; j++)
		{
			newbuf[j * d1 + i] = buf[i * d2 + j];
		}
	});
	ret.avilable = avilable;
	return ret;
}

_Matrix _Matrix::operator*(_Matrix b)
{
	_Matrix ret;
	if (!(avilable && b.isavilable()))
	{
		cout << "Err in operator*:matrix not avilable!" << endl;
		return ret;
	}
	if (d2 != b.d1)
	{
		cout << "Err in product:demention not match!" << endl;
		return ret;
	}
	ret.resize(d1, b.d2);

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, d1, b.getd2(), d2, \
		1, buf, d2, b.getbuf(), b.getd2(), 0, ret.getbuf2set(), b.getd2());
	
	ret.setavilable(true);
	return ret;
}

_Matrix _Matrix::operator*(double b)
{
	_Matrix ret;
	if (!avilable)
	{
		cout << "Err in operator*:matrix not avilable!" << endl;
		return ret;
	}
	ret.resize(d1, d2);
	double* pretbuf = ret.getbuf2set();
	for (int i = 0; i < d1*d2; i++)
	{
		pretbuf[i] = buf[i] * b;
	}
	ret.avilable = true;
	return ret;
}

_Matrix _Matrix::operator+(_Matrix b)
{
	_Matrix ret;
	if (!(avilable && b.isavilable()))
	{
		cout << "Err in operator+:matrix not avilable!" << endl;
		return ret;
	}
	if (d1 != b.d1 || d2 != b.d2)
	{
		cout << "Err in operator+:demension not match!" << endl;
		return ret;
	}
	ret.resize(d1, d2);
	double* pretbuf = ret.getbuf2set();
	const double* pbbuf = b.getbuf();
	for (int i = 0; i < d1*d2; i++)
	{
		pretbuf[i] = buf[i] + pbbuf[i];
	}
	ret.avilable = true;
	return ret;
}

_Matrix _Matrix::operator-()
{
	_Matrix ret;
	if (!avilable)
	{
		cout << "Err in operator-:matrix not avilable!" << endl;
		return ret;
	}
	ret.resize(d1, d2);
	double* pretbuf = ret.getbuf2set();
	for (int i = 0; i < d1*d2; i++)
	{
		pretbuf[i] = -buf[i];
	}
	ret.avilable = true;
	return ret;
}

_Matrix _Matrix::operator-(_Matrix b)
{
	_Matrix ret;
	if (!(avilable && b.isavilable()))
	{
		cout << "Err in operator-:matrix not avilable!" << endl;
		return ret;
	}
	if (d1 != b.d1 || d2 != b.d2)
	{
		cout << "Err in operator-:demension not match!" << endl;
		return ret;
	}
	ret.resize(d1, d2);
	double* pretbuf = ret.getbuf2set();
	const double* pbbuf = b.getbuf();
	for (int i = 0; i < d1*d2; i++)
	{
		pretbuf[i] = buf[i] - pbbuf[i];
	}
	ret.avilable = true;
	return ret;
}

_Matrix _Matrix::dotProduct(_Matrix b)
{
	_Matrix ret;
	if (!(avilable && b.isavilable()))
	{
		cout << "Err in dotProduct:matrix not avilable!" << endl;
		return ret;
	}
	if (d1 != b.d1 || d2 != b.d2)
	{
		cout << "Err in dotProduct:demension not match!" << endl;
		return ret;
	}
	ret.resize(d1, d2);
	double* pretbuf = ret.getbuf2set();
	const double* pbbuf = b.getbuf();
	for (int i = 0; i < d1*d2; i++)
	{
		pretbuf[i] = buf[i] * pbbuf[i];
	}
	ret.avilable = true;
	return ret;
}

_Matrix _Matrix::dotDivision(_Matrix b)
{
	_Matrix ret;
	if (!(avilable && b.isavilable()))
	{
		cout << "Err in dotDivision:matrix not avilable!" << endl;
		return ret;
	}
	if (d1 != b.d1 || d2 != b.d2)
	{
		cout << "Err in dotDivision:demension not match!" << endl;
		return ret;
	}
	ret.resize(d1, d2);
	double* pretbuf = ret.getbuf2set();
	const double* pbbuf = b.getbuf();
	for (int i = 0; i < d1*d2; i++)
	{
		pretbuf[i] = buf[i] / pbbuf[i];
	}
	ret.avilable = true;
	return ret;
}

void _Matrix::setavilable(bool _avilable)
{
	avilable = _avilable;
}

bool _Matrix::isavilable()
{
	return avilable;
}


double* _Matrix::getbuf2set()
{
	return buf;
}

void normalize_w(double* W, int d1, int d2)
{
	/*非并行
	double sum = 0;
	for (int i = 0; i < d1; i++)
	{
		sum = 0;
		for (int j = i; j < d1 * d2; j += d2)
		{
			sum += (W[j] * W[j] + DBL_MIN);
		}
		sum = sqrt(sum);
		for (int j = i; j < d1 * d2; j += d2)
		{
			W[j] = (double)(W[j] / sum) + DBL_MIN;
		}
	}
	/**/
	/*并行*/
	parallel_for((int)0, d1, [=](int i)
	{
		double sum = 0;
		for (int j = i; j < d1 * d2; j += d2)
		{
			sum += (W[j] * W[j] + DBL_MIN);
		}
		sum = sqrt(sum);
		for (int j = i; j < d1 * d2; j += d2)
		{
			W[j] = (double)(W[j] / sum) + DBL_MIN;
		}
	});
	/**/
}

/*parallel_for 示例代码

void parallel_matrix_multiply(double** m1, double** m2, double** result, size_t size)
{
	parallel_for(size_t(0), size, [&](size_t i)
	{
		for (size_t j = 0; j < size; j++)
		{
			double temp = 0;
			for (int k = 0; k < size; k++)
			{
				temp += m1[i][k] * m2[k][j];
			}
			result[i][j] = temp;
		}
	});
}

*/

void _Matrix::NMF(int rank, int iter, _Matrix& W, _Matrix& H)
{
	cout << "Processing in NMF...\tMax iter = " << iter << endl;
	int wd1 = d1, wd2 = rank;
	int hd1 = rank, hd2 = d2;
	W.resize(wd1, wd2);
	H.resize(hd1, hd2);
	double* pwbuf = W.getbuf2set();
	double* phbuf = H.getbuf2set();
	srand(GetTickCount());
	for (int i = 0; i < max(wd1*wd2, hd1*hd2); i++)
	{
		if (i < wd1*wd2)
		{
			pwbuf[i] = (double)rand() / RAND_MAX;
			if (pwbuf[i] < 0)
				pwbuf[i] = -pwbuf[i];
		}
		if (i < hd1*hd2)
		{
			phbuf[i] = (double)rand() / RAND_MAX;
			if (phbuf[i] < 0)
				phbuf[i] = -phbuf[i];
		}
	}
	W.setavilable(true);
	H.setavilable(true);

	//W.show();
	//H.show();

	double tol = 0.01;
	
	//_Matrix R = W * H;

	for (int i = 0; i < iter; i++)
	{
		cout << "\riter = " << i + 1 << "\t";
		_Matrix VH = *this * H.transpose();
		//VH.show();
		_Matrix HH = H * H.transpose();
		//HH.show();
		_Matrix GW = W * HH - VH;
		//GW.show();
		goWiter(GW.getbuf2set(), HH.getbuf2set(), W.getbuf2set(), tol, W.getd1(), W.getd2(), rank*rank);
		//W.show();

		//normalize_w(W.getbuf2set(), W.getd1(), W.getd2());

		_Matrix WV = W.transpose() * *this;
		//WV.show();
		_Matrix WW = W.transpose() * W;
		//WW.show();
		_Matrix GH = WW * H - WV;
		//GH.show();
		H = H.transpose();
		GH = GH.transpose();
		double obj = goWiter(GH.getbuf2set(), WW.getbuf2set(), H.getbuf2set(), tol, H.getd1(), H.getd2(), rank*rank);
		H = H.transpose();
		GH = GH.transpose();
		//H.show();
		cout << "obj = " << obj;
	}
	cout << endl;
}

double goWiter(
	double *GW, 
	double *HH, 
	double *W, 
	double tol, 
	int n, 
	int k, 
	int maxinner
	){
	// initial maximum function value decreasing over all coordinates. 
	//double *_init = new double[n];
	combinable<double> _init;
	double init = 0;
	// Get init value 
	parallel_for((int)0, n, [&](int i)
	{
		_init.local() = 0;
		for (int j = 0; j<k; j++)
		{
			int nowidx = i * k + j;
			double s = GW[nowidx] / (HH[j*k + j] + DBL_MIN);
			s = W[nowidx] - s;
			if (s< 0)
				s = 0;
			s = s - W[nowidx];
			double diffobj = (-1)*s*GW[nowidx] - 0.5*HH[j*k + j] * s*s;
			if (diffobj > _init.local())
				_init.local() = diffobj;
		}
	});
	_init.combine_each([&](double local)
	{
		init = max(init, local);
	});

	// stopping condition

	// coordinate descent 
	combinable<double> obj([]{return 0; });
	parallel_for((int)0, n, [&](int p)
	{
		// Create SWt : store step size for each variables 
		double SWt = 0;
		double bestvalue = 0;

		double *GWp = &(GW[p*k]);
		double *Wp = &(W[p*k]);
		for (int winner = 0; winner < maxinner; winner++)
		{
			// find the best coordinate 
			int q = -1;

			for (int i = 0; i<k; i++)
			{
				double ss = GWp[i] / (HH[i*k + i] + DBL_MIN);
				ss = Wp[i] - ss;
				if (ss < 0)
					ss = 0;
				ss = ss - Wp[i];
				double diffobj = (-1)*(ss*GWp[i] + 0.5*HH[i*k + i] * ss*ss);
				if (diffobj > bestvalue)
				{
					bestvalue = diffobj;
					q = i;
					SWt = ss;
				}
			}
			if (q == -1)
				break;

			Wp[q] += SWt;
			int base = q*k;
			for (int i = 0; i<k; i++)
				GWp[i] += SWt * HH[base + i];
			if (bestvalue < init*tol)
				break;
		}
		obj.local() = bestvalue;
	});
	double ret = 0;
	obj.combine_each([&](double local)
	{
		if (local > ret)
			ret = local;
	});
	return ret;
}

_Matrix mat2matrix(Mat M)
{
	_Matrix ret;
	ret.MatChannels = M.channels();
	ret.MatType = M.type();
	ret.resize(M.rows, M.cols * M.channels());
	double* retbuf = ret.getbuf2set();

	if (M.isContinuous())
	{
		parallel_for((int)0, ret.getd2() * ret.getd1(), [&](int i)
		{
			retbuf[i] = (double)M.data[i];
		});
		ret.setavilable(true);
	}
	else
	{
		int i = 0;
		for_each(M.begin<Vec3b>(), M.end<Vec3b>(), [&](Vec3b it)
		{
			retbuf[i] = it[0];
			retbuf[i + 1] = it[1];
			retbuf[i + 2] = it[2];
		});
	}

	return ret;
}

Mat matrix2mat(_Matrix M)
{
	Size s = { M.getd2() / M.MatChannels, M.getd1() };
	Mat ret(s, M.MatType);
	if (!M.isavilable())
	{
		ret.release();
		return ret;
	}
	const double* buf = M.getbuf();

	int d1 = 0, d2 = 0;
	const double* mbuf = M.getbuf();
	/*
	for_each(ret.begin<Vec3b>(), ret.end<Vec3b>(), [&](Vec3b it)
	{
		double v1 = mbuf[d1 * M.getd2() + d2 * M.MatChannels];
		double v2 = mbuf[d1 * M.getd2() + d2 * M.MatChannels + 1];
		double v3 = mbuf[d1 * M.getd2() + d2 * M.MatChannels + 2];
		it = Vec3b{ (uchar)v1, (uchar)v2, (uchar)v3 };
	});
	*/
	if (ret.isContinuous())
	{
		parallel_for((int)0, M.getd2() * M.getd1(), [&](int i)
		{
			ret.data[i] = (uchar)mbuf[i];
		});
	}
	else
	{

	}

	return ret;
}

_Matrix getPicMat(LPCSTR path)
{
	_Matrix ret;

	WIN32_FIND_DATA FindFileData;
	HANDLE hFind;
	hFind = FindFirstFile(path, &FindFileData);
	Mat tmpM;
	_Matrix tmp_M;
	while (hFind != INVALID_HANDLE_VALUE)
	{
		if (!((FindFileData.cFileName[0] == '.' && FindFileData.cFileName[1] == 0) || \
			(FindFileData.cFileName[0] == '.' && FindFileData.cFileName[1] == '.' && FindFileData.cFileName[2] == 0)))
		{
			cout << FindFileData.cFileName << endl;
			string picfmane = FindFileData.cFileName;
			tmpM = imread(picfmane, 1);
			namedWindow("test", CV_WINDOW_NORMAL);
			imshow("test", tmpM);
			waitKey();
			destroyWindow("test");
			tmp_M = mat2matrix(tmpM);
			tmp_M.reshapeToColWithRowFirst();
			ret.cat(tmp_M, 1);
		}
		if (!FindNextFile(hFind, &FindFileData))
		{
			FindClose(hFind);
			hFind = INVALID_HANDLE_VALUE;
		}
	}

	return ret;
}