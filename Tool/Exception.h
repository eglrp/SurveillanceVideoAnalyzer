#include <string>
#include <exception>

namespace ztool
{
class Exception : public std::exception
{
public:
    Exception(const std::string& info, const char* function, const char* file, const int line);
    virtual ~Exception(void);
    virtual const char* what(void) const;
private:
    std::string message;
};

#define THROW_EXCEPT(x) throw (ztool::Exception((x), __FUNCTION__, __FILE__, __LINE__))
}