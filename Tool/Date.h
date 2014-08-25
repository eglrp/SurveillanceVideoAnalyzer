#pragma once
#include <ctime>
#include <climits>
#include <string>

namespace ztool
{

struct Date
{
    Date(void) : year(0), month(0), day(0) {};
    Date(int year_, int month_, int day_) : year(year_), month(month_), day(day_) {};
    int year, month, day;
};

inline bool isLeapYear(int year)
{
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0); 
}

inline bool isValid(const Date& date)
{
    if (date.year <= 0)
        return false;
    if (date.month <= 0 || date.month > 12)
        return false;
    if (date.day <= 0 || date.day > 31)
        return false;
    switch (date.month)
    {
    case 1:
        return date.day <= 31;
    case 2:
        return isLeapYear(date.year) ? (date.day <= 29) : (date.day <= 28);
    case 3:
        return date.day <= 31;    
    case 4:
        return date.day <= 30;    
    case 5:
        return date.day <= 31;    
    case 6:
        return date.day <= 30;    
    case 7:
        return date.day <= 31;    
    case 8:
        return date.day <= 31;    
    case 9:
        return date.day <= 30;    
    case 10:
        return date.day <= 31;    
    case 11:
        return date.day <= 30;    
    case 12:
        return date.day <= 31;
    default:
        return false;
    }
}

inline int yearDay(const Date& date)
{    
    int day = 0;
    if (isLeapYear(date.year))
    {
        int monthDaysLeapYear[] = {0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        for (int i = 1; i < date.month; i++)
            day += monthDaysLeapYear[i];
        day += date.day;
    }
    else
    {
        int monthDaysCommYear[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        for (int i = 1; i < date.month; i++)
            day += monthDaysCommYear[i];
        day += date.day;
    }
    return day;
}

inline int dateDiff(const Date& from, const Date& to)
{
    if (!isValid(from) || !isValid(to))
        return INT_MAX;
    Date smallDate, bigDate;
    int sign = 1;
    if (from.year < to.year ||
        from.year == to.year && from.month < to.month ||
        from.year == to.year && from.month == to.month && from.day < to.day)
    {
        smallDate = from;
        bigDate = to;
        sign = 1;
    }
    else
    {
        smallDate = to;
        bigDate = from;
        sign = -1;
    }
    int smallYearDay = yearDay(smallDate);
    int bigYearDay = yearDay(bigDate);
    if (smallDate.year == bigDate.year)
        return sign * (bigYearDay - smallYearDay);
    int diff = isLeapYear(smallDate.year) ? 366 - smallYearDay : 365 - smallYearDay;
    for (int i = smallDate.year + 1; i < bigDate.year; i++)
        diff += isLeapYear(i) ? 366 : 365;
    diff += bigYearDay;
    return sign * diff;
}

inline bool allowRun(int numDaysAllowed)
{
    std::string cmplTimeStr = __DATE__;
    if (cmplTimeStr.size() != 11)
        return false;
    
    Date to;
    time_t timeVal;
    time(&timeVal);
    tm* localTime = localtime(&timeVal);
    to.year = localTime->tm_year + 1900;
    to.month = localTime->tm_mon + 1;
    to.day = localTime->tm_mday;

    Date from;
    from.year = atoi(cmplTimeStr.substr(7, 4).c_str());
    std::string mon = cmplTimeStr.substr(0, 3);
    if (mon == "Jan")
        from.month = 1;
    else if (mon == "Feb")
        from.month = 2;
    else if (mon == "Mar")
        from.month = 3;
    else if (mon == "Apr")
        from.month = 4;
    else if (mon == "May")
        from.month = 5;
    else if (mon == "Jun")
        from.month = 6;
    else if (mon == "Jul")
        from.month = 7;
    else if (mon == "Aug")
        from.month = 8;
    else if (mon == "Sep")
        from.month = 9;
    else if (mon == "Oct")
        from.month = 10;
    else if (mon == "Nov")
        from.month = 11;
    else if (mon == "Dec")
        from.month = 12;
    from.day = atoi(cmplTimeStr.substr(4, 2).c_str());

    int diff = dateDiff(from, to);
    return diff >= 0 && diff < numDaysAllowed;
}

}