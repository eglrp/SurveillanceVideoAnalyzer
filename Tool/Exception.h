#include <string>
#include <exception>

namespace ztool
{
class Exception : public std::exception
{
public:
    Exception(const std::string& info, const char* function, const char* file, const int line);
    virtual ~Exception(void) throw ();
    virtual const char* what(void) const throw ();
private:
    Exception& operator=(const Exception&) throw ();
    std::string message;
};

#define THROW_EXCEPT(x) throw (ztool::Exception((x), __FUNCTION__, __FILE__, __LINE__))
}