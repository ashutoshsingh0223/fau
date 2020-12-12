#include <iostream>
#include <typeinfo>
using namespace std;

double fact (double i) {

    if(i == 0){
        return 1;
    };

    return i * fact( i-1 );
};


int main (){
    double i;
    double facto = 1;
    cout << "Enter number you want to calculate factorial for: " << endl;
    cin >> i;
    facto = fact(i);
    cout << "Factorial: " << facto << endl;
}