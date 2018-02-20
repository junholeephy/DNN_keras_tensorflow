#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include "TTree.h"
#include "TFile.h"
#include "TBranch.h"
#include "read_branch_name.h"

class branch_from_tree
{
    public:
        
        branch_from_tree();
        vector<TBranch*> vbranch;
        vector<TBranch*> Read_branch(TTree *tree);
        TBranch* target = new TBranch();
        TBranch* output = new TBranch();
        void Reset();
};

branch_from_tree::branch_from_tree()
{
//    cout<<"!!branch from tree invoked!!"<<endl;
}

vector<TBranch*> branch_from_tree::Read_branch(TTree *tree)
{
//    target = (TBranch*)tree->GetBranch("target_train");
//    output = (TBranch*)tree->GetBranch("output_train");
//    vbranch.push_back(target);
//    vbranch.push_back(output);
    read_branch_name *Name = new read_branch_name();
    vector<string> VName = Name->NameOfBranch(tree);
    for(int i1=0; i1<VName.size(); i1++)
    {
        cout<<"branch name : "<<VName.at(i1)<<endl;
    }    


    return vbranch;
    vbranch.clear();

}

void branch_from_tree::Reset()
{
    vbranch.clear();
    
}

