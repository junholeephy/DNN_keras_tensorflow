#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include "TTree.h"
#include "TFile.h"

class root_tree_reader
{
    public:
        ifstream infile;
        root_tree_reader(string *str);
        vector<TTree*> Vtree;
        vector<TTree*> Read_tree(string *str);
        TTree *t1 =new TTree();
        TTree *t2 =new TTree();
};

root_tree_reader::root_tree_reader(string *str)
{
    cout<<"The input root file : "<<str->data()<<endl;
}

vector<TTree*> root_tree_reader::Read_tree(string *str)
{
    TFile *file = new TFile(str->data(),"READ");
	t1 =(TTree*)file->Get("tree_train");
    t2 =(TTree*)file->Get("tree_test");

    Vtree.push_back(t1);
    Vtree.push_back(t2);
    return Vtree;
	Vtree.clear();
}

