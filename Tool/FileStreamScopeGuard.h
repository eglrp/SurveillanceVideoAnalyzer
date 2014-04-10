#pragma once

namespace ztool
{

template<typename FileStrmType>
class FileStreamScopeGuard
{
public:
    FileStreamScopeGuard(FileStrmType& fstrm)
        : ptrF(&fstrm)
    {}
    ~FileStreamScopeGuard(void)
    {
        if (ptrF && ptrF->is_open())
            ptrF->close();
    }
private:
    FileStrmType* ptrF;
};

}