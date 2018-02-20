#include "TROOT.h"
#include "TKey.h"
#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"
#include "read_tree_name.h"

void read_branch_name()
{
    TFile *file = new TFile("generative_tree_1D.root","READ");
    cout<<"input file name : "<<file->GetName()<<endl;

    read_tree_name *Name = new read_tree_name();
    vector<string> VName;
    vector<string> VBName;
    string stemp = "generative_tree_1D.root";
    VName = Name->NameOfTree(&stemp);

    for(int i1=0; i1<VName.size(); i1++)
    {
        TTree *t = new TTree();
        string stemp2 = VName.at(i1);
//        cout<<"stemp2:"<<stemp2<<endl;
        const char* ctemp2 = stemp2.data();
        t = (TTree*)file->Get(ctemp2);   

//        t->Print();
        TBranch *branch;
        TIter nextbr(t->GetListOfBranches());
        while((branch = (TBranch*)nextbr()))
        {
            string branchname = branch->GetName();
            cout<<branchname<<endl;
            VBName.push_back(branchname);

        }
        cout<<t->GetEntries()<<endl;

    }


}
