#include "Exception.h"

#include <sstream>
static std::string getIntStr(const int val)
{
    std::stringstream strm;
    strm << val;
    std::string ret;
    strm >> ret;
    return ret;
}

namespace ztool
{
Exception::Exception(const std::string& info, const char* function, const char* file, const int line)
{
    message.reserve(512);
    message.append("ERROR in function: ")
        .append(function)
        .append("(), file: ")
        .append(file)
        .append(", line: ")
        .append(getIntStr(line))
        .append(", ")
        .append(info);
};

Exception::~Exception(void) 
{

}

const char* Exception::what(void) const
{
    return message.c_str();
};
}