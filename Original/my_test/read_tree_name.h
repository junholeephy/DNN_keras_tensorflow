#ifndef _READ_TREE_NAME_H_
#define _READ_TREE_NAME_H_

#include "TROOT.h"
#include "TKey.h"
#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"
#include <vector>


class read_tree_name
{
    public:
        read_tree_name();
        vector<string> NameOfTree(string *str);
        vector<string> Vstr;
};

read_tree_name::read_tree_name()
{
//    cout<<"!!!read tree name invoked!!!"<<endl;
}

vector<string> read_tree_name::NameOfTree(string *str)
{
    TFile *file = new TFile(str->data(),"READ");
//    cout<<"input file name : "<<file->GetName()<<endl;

    TKey *key;
    TIter nextkey(file->GetListOfKeys());
    while ((key = (TKey*)nextkey()))
    {
        string classname = key->GetClassName();
        if(classname != "TTree") continue;
//        cout<<classname<<endl;
//        cout<<key->GetName()<<endl;
        Vstr.push_back(key->GetName());

    }
    file->Close();
    return Vstr;
    Vstr.clear();
}


#endif
