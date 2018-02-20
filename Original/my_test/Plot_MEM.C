#include <iostream>
#include <vector>
#include "TTree.h"
#include "TFile.h"
#include "root_tree_reader.h"
#include "branch_from_tree.h"

void Plot_MEM()
{
//    string str = "../../self_trees/self_tree_mem_EN90000_LN1_E5000_NN50_B500_adam_LR1e-05_DR0.0.root"; 
    string str = "generative_tree_1D.root";
    int NBIN = 100;
    double LOW = -15;
    double HIGH = 2;

    root_tree_reader *RTR = new root_tree_reader(&str);
    vector<TTree*> vtree = RTR->Read_tree(&str);
    branch_from_tree *BFT = new branch_from_tree();
    vector<TBranch*> vbranch;

    TH1F* HistoOutput_Train = new TH1F("HistoOutput_Train","HistoOutput_Train",NBIN, LOW, HIGH);  HistoOutput_Train->SetMarkerSize(0.6);  HistoOutput_Train->SetMarkerStyle(kFullCircle); HistoOutput_Train->SetMarkerColor(kBlue);
    TH1F* HistoOutput_Test  = new TH1F("HistoOutput_Test","HistoOutput_Test",NBIN, LOW, HIGH);    HistoOutput_Test->SetMarkerSize(0.6);  HistoOutput_Test->SetMarkerStyle(kFullCircle); HistoOutput_Test->SetMarkerColor(kRed);
    TH1F* HistoTarget_Train = new TH1F("HistoTarget_Train","HistoTarget_Train",NBIN, LOW, HIGH);  HistoTarget_Train->SetLineColor(kBlue);
    TH1F* HistoTarget_Test  = new TH1F("HistoTarget_Test","HistoTarget_Test",NBIN, LOW, HIGH);    HistoTarget_Test->SetLineColor(kRed);
    double target_train = 0;
    double output_train = 0;
    double target_test = 0;
    double output_test = 0;

    for(int i1 = 0; i1<vtree.size(); i1++)
    {
        TTree *tree = new TTree();
        tree = vtree.at(i1);
        cout<<"Name of the tree : "<<tree->GetName()<<endl;
        vbranch = BFT->Read_branch(tree);
		if(i1==0)
        {
            tree->SetBranchAddress("target_train",&target_train);
            tree->SetBranchAddress("output_train",&output_train);
		}
        else if(i1==1)
		{
			tree->SetBranchAddress("target_test",&target_test);
			tree->SetBranchAddress("output_test",&output_test);
		}

        cout<<"Entry number of this tree is : "<<tree->GetEntries()<<endl;   
        for(int i3=0; i3 < tree->GetEntries(); i3++)
        {
                tree->GetEntry(i3);
				if(i1==0)
				{
					HistoOutput_Train->Fill(output_train);
					HistoTarget_Train->Fill(target_train);
				}
				else if(i1==1)
				{
					HistoOutput_Test->Fill(output_test);
					HistoTarget_Test->Fill(target_test);
				}

        }
        cout<<(i1+1)<<"th tree just finished to read in"<<endl<<endl;
    }
	Double_t norm = 1.0;
    double scale = 1.0;
    scale = norm/(HistoOutput_Train->Integral());
    HistoOutput_Train->Scale(scale);
    scale = norm/(HistoOutput_Test->Integral());
    HistoOutput_Test->Scale(scale);
    scale = norm/(HistoTarget_Train->Integral());
    HistoTarget_Train->Scale(scale);
    scale = norm/(HistoTarget_Test->Integral());
    HistoTarget_Test->Scale(scale);

    HistoOutput_Train->Draw("ep");    HistoOutput_Train->SetStats(0);
    HistoOutput_Test->Draw("same ep");
	HistoTarget_Train->Draw("HIST same");
	HistoTarget_Test->Draw("HIST same");
//    HistoOutput_Train->Draw("same ep");
//    HistoOutput_Test->Draw("same ep");

  TLegend* legend = new TLegend(0.7, 0.7, 0.9, 0.9, "");
  legend->SetFillColor(kWhite);
  legend->AddEntry(HistoTarget_Train->GetName(), "Target, train", "l");
  legend->AddEntry(HistoOutput_Train->GetName(), "Output, train", "p");
  legend->AddEntry(HistoTarget_Test->GetName(), "Target, test", "l");
  legend->AddEntry(HistoOutput_Test->GetName(), "Output, test", "p");
  legend->Draw();



}
