#ifndef _READ_BRANCH_NAME_H_
#define _READ_BRANCH_NAME_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include "TTree.h"
#include "TFile.h"
#include "TBranch.h"

class read_branch_name
{
    public:
        read_branch_name();
        vector<string> NameOfBranch(TTree *tree);
        vector<string> Vstr;
};

read_branch_name::read_branch_name()
{
//    cout<<"!!!read branch name invoked!!!"<<endl;
}

vector<string> read_branch_name::NameOfBranch(TTree *tree)
{
//    const char* ctemp = str->data();
    TBranch *branch;
    TIter nextbr(tree->GetListOfBranches());
    while((branch = (TBranch*)nextbr()))
    {
        string branchname = branch->GetName();
//        cout<<branchname<<endl;
        Vstr.push_back(branchname);
    }
    return Vstr;
    Vstr.clear();
}

#endif

