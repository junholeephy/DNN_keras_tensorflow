void test_string_char()
{
    string str1 = "../aaa.root";
//    string *str2 = "../bbb.root/";  //error!!!!!!!!
//    string &str = "../ccc.root";  //error!!!!!!!
//    char ch1 = "../ddd.root";   //error!!!!!!!!
    char *ch2 = "../eee/.root"; //warning 
//    char &ch3 = "../fff.root"; //error!!!!!!!!
    cout<<str1<<endl;
//    cout<<(*str1)<<endl;  //error!!!!!!!!!
    cout<<(&str1)<<endl;    //adress
    cout<<ch2<<endl;
    cout<<*ch2<<endl;   // not what I expected
    cout<<&ch2<<endl;   // address

}
