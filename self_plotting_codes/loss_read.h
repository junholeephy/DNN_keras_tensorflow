#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>

class loss_read
{
	public:
		ifstream infile;
		loss_read();
		int read_in_lineNum(string *str);
		vector<double*> one_line_read_in(string *str, int Linenum);
		vector<double*> all_line_read_in(string *str, int Linenum);	
	
};

loss_read::loss_read()
{
}

int loss_read::read_in_lineNum(string *str)
{
    char *pathvar;
    pathvar = getenv("DNN");
    string file = "/self_loss_values/";
	file = (pathvar) + file + (*str);
//	cout<<file<<endl;

    infile.open(file.data());
    char c;
    int lineCnt=0;
    while(infile.get(c))
    {
        if(c=='\n')
        {lineCnt++; }
    }
    cout<<"Read in lines : "<<lineCnt<<endl;
	infile.close();	
	return lineCnt;
}


vector<double*> loss_read::one_line_read_in(string *str, int Linenum)
{
	vector<double*> Line1;
	char *pathvar;
	pathvar = getenv("DNN");
	string file = "/self_loss_values/";
	file = (pathvar) + file + (*str);
	cout<<"file name : "<<file<<endl;

    string temp_training;
    double train[Linenum];
    int linenum = 0;
    infile.open(file.data());
    while(getline(infile, temp_training))
    {
        infile >> temp_training;
        train[linenum] = atof(temp_training.c_str());
//        cout<<train[linenum]<<endl;
        linenum++;
    }
	Line1.push_back(train);

	infile.close();	
	return Line1;
}


vector<double*> loss_read::all_line_read_in(string *str, int Linenum)
{
    vector<double*> Line1;
    char *pathvar;
    pathvar = getenv("DNN");
    string file = "/self_loss_values/";
    file = (pathvar) + file + (*str);
    cout<<"file name : "<<file<<endl;

    string temp_training;
	string test_training;
    double train[Linenum+1];
	double test[Linenum+1];
    int linenum = 0;
    infile.open(file.data());
    while(getline(infile, temp_training))
    {
        infile >> temp_training >> test_training;
        train[linenum] = atof(temp_training.c_str());
		test[linenum] = atof(test_training.c_str());
//        cout<<train[linenum]<<" ";
//		cout<<test[linenum]<<endl;
        linenum++;
    }
    Line1.push_back(train);
	Line1.push_back(test);

    infile.close();
    return Line1;
}



/*
vector<std::pair<double,double>> loss_read::read_in(string *str)
{
	char *pathvar;
	pathvar = getenv("DNN");
	string file = pathfile + "/self_loss_values/";
	file = file + str;

}
*/


/*
	ifstream infile;
	char *pathvar;
	pathvar = getenv("DNN");
	string file = "/self_loss_values/loss_MEM_EN90000_LN1_E5000_NN50_B500_adam_L1e-05_DR0.0.txt";
	file = pathvar + file;
	infile.open(file.data());
	if(!infile) cout<<"error"<<endl;

	char c;
    int lineCnt=0;
    while(infile.get(c))
    {
        if(c=='\n')
        {lineCnt++; }
    }
	cout<<lineCnt<<endl;	
	infile.close();

    string temp_training;
    string temp_testing;
    double train[lineCnt];
    double test[lineCnt];
	int linenum = 0;
	infile.open(file.data());
	while(getline(infile, temp_training))
	{
		infile >> temp_training >> temp_testing;
		train[linenum] = atof(temp_training.c_str());
		test[linenum] = atof(temp_testing.c_str());
		cout<<train[linenum]<<" ";
		cout<<test[linenum]<<endl;
//		cout<<atof(temp_training.c_str())<<" ";
//		cout<<atof(temp_testing.c_str())<<endl;
		linenum++;
	}
	cout<<lineCnt<<endl;
	cout<<"!!!!!"<<endl;
	cout<<"line num :"<<linenum<<endl;

	infile.close();
}
*/
