#include <iostream>
#include <vector>
#include "TTree.h"
#include "TFile.h"
#include "root_tree_reader.h"
#include "branch_from_tree.h"
//#include "read_branch_name.h"
//#include "branch_to_histo.h"

void Read_tree_branch()
{
//    string str = "../../self_trees/self_tree_mem_EN90000_LN1_E5000_NN50_B500_adam_LR1e-05_DR0.0.root"; 
    string str = "../../input_TTZ_DelphesEvalGen_5275k.root";
//    string str = "generative_tree_1D.root";

    root_tree_reader *RTR = new root_tree_reader(&str);
    vector<TTree*> vtree = RTR->Read_tree(&str);

    branch_from_tree *BFT = new branch_from_tree();

    for(int i1 = 0; i1<vtree.size(); i1++)
    {
        TTree *tree = new TTree();
        tree = vtree.at(i1);
        cout<<"Name of the tree : "<<tree->GetName()<<endl;
        BFT->Read_branch(tree);
    

        cout<<(i1+1)<<"th tree just finished to read in"<<endl<<endl;
    }



}
