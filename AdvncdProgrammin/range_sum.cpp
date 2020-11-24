#include <iostream>
using namespace std;

int main  (){
    int i, j;
    cout << "Enter value of a" << endl;
    cin >> i;
    cout << "Enter value of b" << endl;
    cin >> j;
    if(i > j){
        cout << "b should be smaller than a" << endl;
        return 0;
    };

    int sum = 0;
    for(int x=i; x < j; x++){
        sum += x;
    };

    cout << "Sum of a and b is " << sum << endl;
    return sum;
}