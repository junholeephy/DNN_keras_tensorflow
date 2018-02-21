#include <iostream>
#include <vector>
#include <stdio.h>
#include <cstring>
#include "TTree.h"
#include "TFile.h"
#include "root_tree_reader.h"
#include "branch_from_tree.h"
#include "TH1F.h"
#include <string.h>
using namespace std;

vector<TH1F*> define_histos(string *str, int NBIN, double LOW, double HIGH, string *TARGET, string *OUTPUT, string *TRAIN, string *TEST)
{
    vector<TH1F*> vhisto;
    TH1F* hist = new TH1F(); 
    string Str = str->data();

    string Target_ = TARGET->data();
    string Output_ = OUTPUT->data();
    string Train_ = TRAIN->data();
    string Test_ = TEST->data();

    root_tree_reader *RTR = new root_tree_reader(&Str);
    vector<TTree*> vtree = RTR->Read_tree(&Str);
    branch_from_tree *BFT = new branch_from_tree();
    vector<TBranch*> vbranch;

    for(int i1 = 0; i1<vtree.size(); i1++)
    {
        TTree *tree = new TTree();
        tree = vtree.at(i1);
        cout<<"Name of the tree : "<<tree->GetName()<<endl;
        vbranch = BFT->Read_branch(tree);
//        cout<<"Branch size of tree : "<<tree->GetName()<<" is : "<<vbranch.size()<<endl;
        if(i1 == (vtree.size()-1))
            for(int i2=0; i2<vbranch.size(); i2++)
            {
                const char* histName = vbranch.at(i2)->GetName();
                string histName_ = vbranch.at(i2)->GetName();
                hist = new TH1F(histName,histName,NBIN,LOW,HIGH);
                if(histName_.find(Target_)!= std::string::npos) 
                {
//                   cout<<"target!!!"<<"  "<<histName_<<endl;
                    if(histName_.find(Train_)!= std::string::npos) {hist->SetLineColor(kBlue); cout<<"Blue line Histo : "<<histName_<<endl;}
					else if(histName_.find(Test_)!= std::string::npos) {hist->SetLineColor(kRed); cout<<"Red line Histo : "<<histName_<<endl;}
                }
				else if (histName_.find(Output_)!= std::string::npos)
				{
//					cout<<"output!!!"<<"  "<<histName_<<endl;
					hist->SetMarkerSize(0.6);
					hist->SetMarkerStyle(kFullCircle);
					if(histName_.find(Train_)!= std::string::npos) {hist->SetMarkerColor(kBlue); cout<<"Blue dot Histo : "<<histName_<<endl;}
					else if(histName_.find(Test_)!= std::string::npos) {hist->SetMarkerColor(kRed); cout<<"Red dot Histo : "<<histName_<<endl;}
				}

                vhisto.push_back(hist);
            }
    }
/*   
    cout<<"vector hist size : "<<vhisto.size()<<endl;
    for(int j1=0; j1<vhisto.size(); j1++)
    {
        cout<<vhisto.at(j1)->GetName()<<endl;
    }
*/
    return vhisto;

}
