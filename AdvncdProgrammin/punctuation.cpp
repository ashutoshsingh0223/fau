#include <cctype>
#include <iostream>
#include <string>
using namespace std;



int main(){
    for (string line; getline(cin, line);) {
        // cout << line.length() << endl;
        while (line.length() == 0 ) {
            // cout << "In while" << line.length() << endl;
            getline(cin, line); 
        };

    for (int i = 0, len = line.size(); i < len; i++) 
    { 
        // check whether parsing character is punctuation or not 
        if (ispunct(line[i])) 
        { 
            line.erase(i--, 1); 
            len = line.size(); 
        } 
    } 
        
        cout << line << endl;
    }

}


