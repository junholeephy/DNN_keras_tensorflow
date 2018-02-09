#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
void loss_read_test()
{
	ifstream infile;
	char *pathvar;
	pathvar = getenv("DNN");
	string file = "/self_loss_values/loss_MEM_EN10000_LN2_E5000_NN50_B100_adam_L1e-05_DR0.0.txt";
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
