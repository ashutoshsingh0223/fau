#pragma once

// A TestCase is an Object that prints a message both in its constructor
// and in its destructor.  This way, we can simply create a TestCase object
// in a scope to have the messages printed at the beginning and the end
// of that scope.
//
// We also define a TestSuite preprocessor macro to create such TestCase
// objects conveniently.

class TestCase {
public:
    explicit TestCase(std::string name) : m_name(name) {
        std::cout << "Test Case START:\t" << m_name << std::endl;
    }

    ~TestCase() {
        std::cout << "Test Case END:  \t" << m_name << std::endl;
    }
private:
    std::string m_name;
};

#define TESTCASE(name) TestCase _testcase(#name)

// The THROWS macro takes a piece of code and assert that this code throws
// an exception of the specified class.

#define THROWS(code, exception) do {bool _throws = false; try {code} catch (const exception& _e) {_throws = true;}; assert(_throws);} while(0)
