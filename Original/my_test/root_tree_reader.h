#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include "TTree.h"
#include "TFile.h"
#include "read_tree_name.h"

class root_tree_reader
{
    public:
        ifstream infile;
        root_tree_reader(string *str);
        vector<TTree*> Vtree;
        vector<TTree*> Read_tree(string *str);
//        TTree *t1 =new TTree();
//        TTree *t2 =new TTree();
//        TTree *t = new TTree();
};

root_tree_reader::root_tree_reader(string *str)
{
    cout<<"The input root file : "<<str->data()<<endl;
}

vector<TTree*> root_tree_reader::Read_tree(string *str)
{
    TFile *file = new TFile(str->data(),"READ");
	read_tree_name *Name = new read_tree_name();
    vector<string> VName;
    string stemp = str->data();
    VName = Name->NameOfTree(&stemp);
    cout<<"Number of trees included in the file : "<<VName.size()<<endl;
    
    for(int i1=0; i1<VName.size(); i1++)
    {
        TTree *t = new TTree();
	    string stemp2 = VName.at(i1);
        cout<<"stemp2:"<<stemp2<<endl;
        const char* ctemp2 = stemp2.data();
        t = (TTree*)file->Get(ctemp2);
        Vtree.push_back(t);
    }

    return Vtree;
	Vtree.clear();
}




