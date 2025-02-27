#include "Except.h"
#include "Tests.h"

int main()
{
    try {
        neural_network::run_all_tests();
    } catch (const std::exception& e) {
        exept::react();
    }

    return 0;
}
