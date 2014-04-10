#pragma once

#if WIN32 || _WIN32
#	include <direct.h>  
#	include <io.h>  
//#	include <hiredis_win/hiredis.h>
#else
#	include <stdarg.h>  
#	include <sys/stat.h> 
#   include <unistd.h>
//#	include <hiredis/hiredis.h>
#endif
#ifdef WIN32 || _WIN32  
#define ACCESS _access  
#define MKDIR(a) _mkdir((a))  
#else
#define ACCESS access  
#define MKDIR(a) mkdir((a),0755)  
#endif
#include <cstdio>
#include <string>

namespace ztool
{

struct CreateDirectoryResult
{
    enum
    {
        FAIL = 0,
        SUCCESS = 1,
        EXISTS = 2
    };
};

inline int createDirectoryBase(const std::string& dir)
{
    if (ACCESS(dir.c_str(), 0) == 0)
        return CreateDirectoryResult::EXISTS;
	if (MKDIR(dir.c_str()) == 0)
        return CreateDirectoryResult::SUCCESS;
    return CreateDirectoryResult::FAIL;
}

inline int createDirectory(const std::string& dir)
{
    if (dir.empty())
        return CreateDirectoryResult::EXISTS;

    unsigned int len = dir.size();
    int retVal = CreateDirectoryResult::FAIL;
    for (unsigned int pos = 0; pos < len; pos++)
    {
        if (dir[pos] == '\\' || dir[pos] == '/')
        {
            int result = createDirectoryBase(dir.substr(0, pos).c_str());
            if (result == CreateDirectoryResult::FAIL)
                return CreateDirectoryResult::FAIL;
            else if (result == CreateDirectoryResult::SUCCESS)
                retVal = CreateDirectoryResult::SUCCESS;
        }
    }
    int result = createDirectoryBase(dir);
    if (result == CreateDirectoryResult::FAIL)
        return CreateDirectoryResult::FAIL;
    else
        return retVal;
}

}
