TH1D* GetHistoWeight(TTree* t, string variable, int nbins, double xmin, double xmax, string cut, string weight, string name){
        string sxmin, sxmax, snbins;
        stringstream ss[3];

        ss[0] << xmin;
        ss[0] >> sxmin;
        ss[1] << xmax;
        ss[1] >> sxmax;
        ss[2] << nbins;
        ss[2] >> snbins;

        string variablenew = variable + " >> h(" + snbins + "," + sxmin + "," + sxmax + ")";

        string cutnew = weight + " * (" + cut + ")";

        t->Draw(variablenew.c_str(), cutnew.c_str());
        TH1D *histo = (TH1D*)gDirectory->Get("h");

        if (histo->GetEntries()==0) return histo;

        double underflow = histo->GetBinContent(0);
        double val = 0;
        if (underflow>0) {
                val = histo->GetBinContent(1);
                histo->SetBinContent(1, val+underflow);
                histo->SetBinContent(0, 0);
        }
        double overflow = histo->GetBinContent(nbins+1);
        if (overflow>0) {
                val = histo->GetBinContent(nbins);
                histo->SetBinContent(nbins+1, 0);
                histo->SetBinContent(nbins, val+overflow);
        }

	double area = histo->Integral();
	histo->Scale(1./area);

        histo->SetName(name.c_str());
        histo->SetTitle(name.c_str());

        return histo;
}


void Plot(){

  gROOT->ProcessLine(".x setTDRStyle.C");

  TFile* f = new TFile("generative_tree_1D.root","READ");

  TH1F* HistoOutput_Train = (TH1F*)f->Get("OutputDataTrain");
  TH1F* HistoOutput_Test = (TH1F*)f->Get("OutputDataTest");

  TH1F* HistoTarget_Train = (TH1F*)f->Get("TrainData");
  TH1F* HistoTarget_Test = (TH1F*)f->Get("TestData");

  TCanvas* Canvas1 = new TCanvas("Canvas1","Canvas1");

  HistoTarget_Train->SetXTitle("x");
  HistoTarget_Train->SetLineColor(kBlack);
  HistoTarget_Train->Draw("HIST");

  HistoOutput_Train->SetLineColor(kRed);
  HistoOutput_Train->SetMarkerStyle(20);
  HistoOutput_Train->SetLineColor(kRed);
  HistoOutput_Train->Draw("EPsame");

  HistoTarget_Test->SetMarkerStyle(20);
  HistoTarget_Test->SetMarkerColor(kBlack);
  HistoTarget_Test->Draw("HISTsame");

  HistoOutput_Test->SetLineColor(kRed);
  HistoOutput_Test->SetMarkerStyle(20);
  HistoOutput_Test->SetMarkerColor(kRed);
  HistoOutput_Test->Draw("EPsame");


  TLegend* legend = new TLegend(0.5, 0.5, 0.8, 0.7, "");
  legend->SetFillColor(kWhite);
  legend->AddEntry(HistoTarget_Train->GetName(), "Target, train", "l");
  legend->AddEntry(HistoOutput_Train->GetName(), "Output, train", "p");
  legend->AddEntry(HistoTarget_Test->GetName(), "Target, test", "l");
  legend->AddEntry(HistoOutput_Test->GetName(), "Output, test", "p");
  legend->Draw();

  //Canvas1->SetLogy(1);

  string PicName = "Camel2D_projX.pdf";
  Canvas1->Print(PicName.c_str());

  return;

}
