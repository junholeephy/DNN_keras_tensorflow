#include <iostream>
#include <vector>
#include "TTree.h"
#include "TFile.h"
#include "root_tree_reader.h"
#include "branch_from_tree.h"
#include "branch_to_histo.h"

void Plot_MEM()
{
    string str = "../self_trees/self_tree_mem_EN90000_LN1_E5000_NN50_B500_adam_LR1e-05_DR0.0.root"; 

    root_tree_reader *RTR = new root_tree_reader(&str);
    branch_from_tree *BFT1 = new branch_from_tree();
    branch_from_tree *BFT2 = new branch_from_tree();
    vector<TTree*> vtree = RTR->Read_tree(&str);
    vector<TBranch*> vbranch1, vbranch2;
    cout<<"vector tree size : "<<vtree.size()<<endl;

    cout<<endl;
    TTree *Tree = new TTree("Tree","Tree");
    TBranch *Branch = new TBranch();

    Tree = vtree.at(0);
    cout<<"Name of the tree : "<<Tree->GetName()<<endl;
    vbranch1 = BFT1->Read_branch(Tree);
    Tree = vtree.at(1);    
    vbranch2 = BFT2->Read_branch(Tree);
    cout<<"vector branch size : "<<vbranch1.size()<<"!!!!"<<endl;
    cout<<"vector branch size : "<<vbranch2.size()<<"!!!!"<<endl;

    cout<<vbranch1.at(0)->GetName()<<endl;
    cout<<vbranch1.at(1)->GetName()<<endl;
    cout<<vbranch2.at(0)->GetName()<<endl;
    cout<<vbranch2.at(1)->GetName()<<endl;

}
