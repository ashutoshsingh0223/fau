#include <iostream>
#include <vector>
#include <initializer_list>
#include <iterator>


using namespace std;

// int main(){
//     // Copy initialization
//     int i = 1;
//     cout << i << endl;
//     // Direct initialization
//     int j(1); 
//     cout << j<< endl;
//     // List initialization
//     int k{1};
//     cout << k << endl;
//     // Value initialization
//     int l{};
//     cout << l << endl;
//     return 0;
// }

struct X{
    X(int i){ 
        cout << "X(int i)" << endl;
    }
    X(X&){
        cout << "X(X&)" << endl;
    }
    // Explicit copy constructor. Prevents copy contructor being called implicitly when not desired by user
    // explicit X(X&){
    //     cout << "X(X&)" << endl;
    // }
    X(initializer_list<int>){
        cout << "X(initializer_list)" << endl;
    }

    X& operator=(X&) {
        cout << "X& opertaor = (X&)" << endl;
        return *this;
    }
};

// int main(){
//     X x1(1);

//     X x2{1};

//     X x3{1,2,3};

//     // Direct Initiatization - Calls copy constructor
//     X x4(x1);

//     // Copy Initiatization - Calls copy constructor
//     X x5 = x1;

//     x1 = x3;

//     return 0;
// }

template <class T>
struct S{
    std::vector<T> v;
    S(std::initializer_list<T> l) : v(l){
        std::cout << "constructed with a " << l.size() << "element list\n";
    }

    void append(std::initializer_list<T> l){
        v.insert(v.end(), l.begin(), l.end());
    }

    // Const function: Cannot change the state of internal members 
    std::pair<const T*, std::size_t> c_arr() const {
        // Copy list-initialization in return statement
        // This is not a use of std::initializer_list
        return {&v[0], v.size()};
    }
};



int main(){

    // copy list-initialization
    S<int> s = {1,2,3,4,5};
    // list-initialization in function call
    s.append({6,7,8});

    cout << "The vector is now" << s.c_arr().second << "ints:\n";
    copy(s.v.begin(), s.v.end(), ostream_iterator<int>(cout, " "));
    cout << endl;

    // The rulle for auto makes this ranged for work
    for (int x: {-1, -2, -3}){
        cout << x << " ";
    }
    cout << '\n';

    auto al = {10, 11, 12};
    cout << "This list bound to auto has size() = " << al.size() << "\n";
    return 0;
}