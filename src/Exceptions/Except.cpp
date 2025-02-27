#include "Except.h"

namespace exept
{

void react()
{
    try {
        throw;
    } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception caught" << std::endl;
    }
}

} // namespace exept
