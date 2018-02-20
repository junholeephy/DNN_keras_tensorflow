#include "TROOT.h"
#include "TKey.h"
#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"

vector<string> read_tree_name()
{
    vector<string> Vstr;
    TFile *file = new TFile("generative_tree_1D.root","READ");
    cout<<"input file name : "<<file->GetName()<<endl;
/*
    TH1F* hist2 = (TH1F*)file->Get(file->GetListOfKeys()->At(2)->GetName());
    cout<<typeid(hist2).name()<<endl;
    file->GetListOfKeys()->Print();    
    cout<<endl<<endl;
    file->GetListOfKeys()->ls();
//    TTree* Tree_Data2 = (TTree*)file->Get(file->GetListOfKeys()->At(2)->GetName());
    TTree* Tree_Data5 = (TTree*)file->Get(file->GetListOfKeys()->At(5)->GetName());
    string str = Tree_Data5->GetName();
    cout<<str<<endl;
    string classname = typeid(Tree_Data5).name();
    if(classname=="P5TTree") cout<<typeid(Tree_Data5).name()<<endl;
    else ;
*/        

    TKey *key;
    TIter nextkey(file->GetListOfKeys());
    while ((key = (TKey*)nextkey()))
    {
        string classname = key->GetClassName();
        if(classname != "TTree") continue;
//        cout<<classname<<endl;
        cout<<key->GetName()<<endl;
        Vstr.push_back(key->GetName());

    }
    file->Close();
    return Vstr;
}
