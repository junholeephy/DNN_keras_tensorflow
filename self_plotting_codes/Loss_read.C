#include "loss_read.h"
#include "loss_plotter.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include "TGraph.h"

void Loss_read()
{
//	string Str = "loss_MEM_EN80000_LN5_E1000_NN256_B500_adam_L1e-05_DR0.2.txt";
	string Str = "loss_MEM_EN90000_LN2_E5000_NN25_B500_Adamax_L1e-05_DR0.0.txt";
//	gROOT->ProcessLine(".x ../setTDRStyle.C");
	char *Pathvar;
	Pathvar = getenv("DNN");

//	vector<double*> OLRI;
	vector<double*> ALRI;
	loss_read *LR = new loss_read();
	loss_plotter *LP = new loss_plotter();

	int LINENUM = LR->read_in_lineNum(&Str);
//	OLRI = LR->one_line_read_in(&Str, LINENUM);
	ALRI = LR->all_line_read_in(&Str, LINENUM);
	cout<<"Size of Vector : "<<ALRI.size()<<endl;
	
	LP->Draw(ALRI,LINENUM,Str);	

	double *TRAIN, *TEST; 
	TRAIN = (ALRI.at(0));	
	TEST  = (ALRI.at(1));  

	int ii = 0;	
	for(int iii=0; iii<LINENUM-300; iii++)
	{
//		cout<<TRAIN[iii]<<" ";
//		cout<<TEST[iii]<<endl;
		ii++;
	}
//	cout<<ii<<endl;

	

}

