void self_Pull_mem()
{
  
  gROOT->ProcessLine(".x setTDRStyle.C");
  TFile* f = new TFile("./self_trees_GEN/4_Gen_NeutTopLep_Phi/self_tree_kin_EN280000_LN2_E1000_NN30_B500_adam_L1e-05_DR0.0.root","READ");
	
  Double_t norm = 1.0;
  TH1F* HistoOutput_Train = (TH1F*)f->Get("OutputDataTrain");
  double Enum_Otrain = 0;
  Enum_Otrain = HistoOutput_Train->GetEntries();
  cout<<Enum_Otrain<<endl;
//  Double_t scale = norm/(HistoOutput_Train->Integral());
//  HistoOutput_Train->Scale(scale);

  TH1F* HistoOutput_Test = (TH1F*)f->Get("OutputDataTest");
//  Double_t scale = norm/(HistoOutput_Test->Integral());
///  HistoOutput_Test->Scale(scale);

  TH1F* HistoTarget_Train = (TH1F*)f->Get("TrainData");
  double Enum_Ttrain = 0;
  Enum_Ttrain = HistoTarget_Train->GetEntries();
  cout<<Enum_Ttrain<<endl;
//  Double_t scale = norm/(HistoTarget_Train->Integral());
//  HistoTarget_Train->Scale(scale);

  TH1F* HistoTarget_Test = (TH1F*)f->Get("TestData");
//  Double_t scale = norm/(HistoTarget_Test->Integral());
//  HistoTarget_Test->Scale(scale);

  TCanvas* Canvas1 = new TCanvas("Canvas1","Canvas1");
  Double_t YMAX;
  YMAX = HistoOutput_Test->GetMaximum();

  
  TTree *tree1 = (TTree*)f->Get("tree_train");
  TTree *tree2 = (TTree*)f->Get("tree_test");
  TH1F *htrain_pull = new TH1F("htrain_pull","htrain_pull",200,-0.5,0.5);
  TH1F *htest_pull  = new TH1F("htest_pull","htest_pull",50,-0.5,0.5);		htest_pull->SetLineColor(kRed);
  Double_t target_train;
  Double_t output_train;  
  Double_t target_test;
  Double_t output_test;
  tree1->SetBranchAddress("target_train",&target_train);
  tree1->SetBranchAddress("output_train",&output_train);
  tree2->SetBranchAddress("target_test",&target_test);
  tree2->SetBranchAddress("output_test",&output_test);
  Long64_t train_num = tree1->GetEntries();
  Long64_t test_num = tree2->GetEntries();
  cout<<"Event number for training : "<<train_num<<endl;
  cout<<"Event number for testing : "<<test_num<<endl;
  Double_t train_pull = 0;
  Double_t test_pull = 0;
  for(int i = 0; i<train_num; i++)
  {
	tree1->GetEntry(i);
	train_pull = (target_train - output_train)/target_train;
        if(i%10000 == 0) cout<<train_pull<<endl;
	htrain_pull->Fill(train_pull);
  }
  Double_t scale = norm/(htrain_pull->Integral());
  htrain_pull->Scale(scale);
  htrain_pull->SetXTitle("(target-output)/target");
  htrain_pull->Draw();  

  for(int i = 0; i<test_num; i++)
  {
	tree2->GetEntry(i);
	test_pull = (target_test - output_test)/target_test;
	if(i%10000 == 0) cout<<test_pull<<endl;
	htest_pull->Fill(test_pull);
  }
  scale = norm/(htest_pull->Integral());
  htest_pull->Scale(scale);
  htest_pull->Draw("same");

  TLegend* legend = new TLegend(0.2, 0.3, 0.4, 0.5, "");
  legend->SetFillColor(kWhite);
  legend->AddEntry(htrain_pull->GetName(), "mem_train", "l");
  legend->AddEntry(htest_pull->GetName(), "mem_test", "l");
  legend->Draw();



}
