#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include "TGraph.h"
#include "TLegend.h"
#include "TCanvas.h"

class loss_plotter
{
	public:
		loss_plotter();
		void Draw(vector<double*>, int ln, string S);
};

loss_plotter::loss_plotter()
{
}

void loss_plotter::Draw(vector<double*> VV, int ln, string S)
{
	double *TRAIN, *TEST;
    TRAIN = (VV.at(0));
    TEST  = (VV.at(1));
	int LN = ln;
    double *Xaxis = new double[LN];
    double *hTest = new double[LN];
    double *hTrain = new double[LN];	
	int jj=0;
	int ii;

	int ijk = 0;
//	cout<<ijk<<endl;
	for(ii=0; ii<LN-400; ii++)
	{
		jj = ii;
		Xaxis[ii] = jj+1;
		hTest[ii] = TEST[ii];
		hTrain[ii]= TRAIN[ii];
		cout<<hTrain[ii]<<" ";
		cout<<hTest[ii]<<" ";
		cout<<Xaxis[ii]<<endl;
	}
	cout<<jj<<endl;
//	ijk = ijk + 1;
//	cout<<ijk<<endl;

//	cout<<"!!"<<sizeof(hTest) / sizeof(hTest[0])<<endl;
//	cout<<"!!"<<sizeof(hTest)<<endl;
	TGraph *gr1 = new TGraph(jj+1,Xaxis,hTrain);	gr1->SetLineColor(kBlue);	gr1->SetMaximum(80);	gr1->GetXaxis()->SetTitle("epoch num");		gr1->GetYaxis()->SetTitle("loss fuction value");
	TGraph *gr2 = new TGraph(jj+1,Xaxis,hTest);		gr2->SetLineColor(kRed);
	gr1->Draw();
	gr2->Draw("same");

	TLegend* legend = new TLegend(0.54,0.7,0.88,0.88);
	legend->SetFillColor(kWhite);
	legend->AddEntry(gr1,"train");
	legend->AddEntry(gr2,"test");
	legend->Draw();
}


