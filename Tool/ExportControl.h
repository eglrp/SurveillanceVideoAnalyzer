#pragma once

#define Z_LIB_EXPORT_WIN_DLL 1
#if (_WIN32 || WIN32) && Z_LIB_EXPORT_WIN_DLL
#define Z_LIB_EXPORT __declspec(dllexport)
#else
#define Z_LIB_EXPORT
#endif