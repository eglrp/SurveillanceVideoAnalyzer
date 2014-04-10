#include <iostream>
#include <sstream>
#include "ConfigFileReader.h"
#include "FileStreamScopeGuard.h"
#include "StringProcessor.h"
#include "Exception.h"

using namespace std;

namespace ztool
{

ConfigFileReader::ConfigFileReader(const string& cfgFileName)
    : fileName(cfgFileName), readMode(ReadMode::UNREAD), setItr(false)
{
    ifstream fileReader(fileName.c_str());
    if (fileExists = fileReader.is_open())
        fileReader.close();
}

bool ConfigFileReader::canBeOpened(void) const
{
    return fileExists;
}

bool ConfigFileReader::read(const std::string& label)
{    
    ifstream fileReader(fileName.c_str());
    FileStreamScopeGuard<ifstream> guard(fileReader);
    if (!fileReader.is_open())
        return false;
    labelContent.clear();
    bool hasContent = readLabelContent(label, fileReader);
    fileReader.close();
    if (hasContent)
        readMode = ReadMode::READ_LABEL;
    return hasContent;
}

bool ConfigFileReader::readAll(void)
{
    ifstream fileReader(fileName.c_str());
    FileStreamScopeGuard<ifstream> guard(fileReader);
    if (!fileReader.is_open())
        return false;
    string currLabel, nextLabel;
    labelContent.clear();
    while (true)
    {
        readLabelContent(currLabel, fileReader, nextLabel);
        if (nextLabel.empty())
            break;
        else
            currLabel = nextLabel;
    }
    fileReader.close();
    bool hasContent = !labelContent.empty();
    if (hasContent)
        readMode = ReadMode::READ_ALL;
    return hasContent;
}

void ConfigFileReader::print(void) const
{
    for (LabelContent::const_iterator itr = labelContent.begin(), itrEnd = labelContent.end(); itr != itrEnd; ++itr)
    {
        if (!itr->first.empty())
            cout << itr->first << "\n";
        for (SubLabelContent::const_iterator subItr = itr->second.begin(), subItrEnd = itr->second.end(); subItr != subItrEnd; ++subItr)
        {
            if (!subItr->first.empty())
                cout << "  " << subItr->first << "\n";
            for (RawKeyValues::const_iterator kvItr = subItr->second.begin(), kvItrEnd = subItr->second.end(); kvItr != kvItrEnd; ++kvItr)
            {
                cout << "    " << kvItr->first << " " << kvItr->second << "\n";
            }
        }
        printf("\n");
    }
}

bool ConfigFileReader::seek(const std::string& label, const std::string& subLabel)
{
    for (LabelContent::const_iterator itr = labelContent.begin(), itrEnd = labelContent.end(); itr != itrEnd; ++itr)
    {
        if (itr->first == label)
        {
            for (SubLabelContent::const_iterator subItr = itr->second.begin(), subItrEnd = itr->second.end(); subItr != subItrEnd; ++subItr)
            {
                if (subItr->first == subLabel)
                {
                    setItr = true;
                    subLabelContentItr = subItr;
                    return true;
                }
            }
        }
    }
    return false;
}

bool ConfigFileReader::seek(const std::string& subLabel)
{
    if (readMode != ReadMode::READ_LABEL || labelContent.empty())
        return false;
    for (SubLabelContent::const_iterator subItr = labelContent.begin()->second.begin(), subItrEnd = labelContent.begin()->second.end(); 
        subItr != subItrEnd; ++subItr)
    {
        if (subItr->first == subLabel)
        {            
            setItr = true;
            subLabelContentItr = subItr;
            return true;
        }
    }
    return false;
}

bool ConfigFileReader::getValueString(const string& key, string& val) const
{
    val.clear();
    if (!setItr) return false;
    for (RawKeyValues::const_iterator itr = subLabelContentItr->second.begin(), itrEnd = subLabelContentItr->second.end(); 
        itr != itrEnd; ++itr)
    {
        if (itr->first == key)
        {
            if (itr->second.empty()) return false;
            val = itr->second;
            return true;
        }
    }
    return false;
}

bool ConfigFileReader::getValueStrings(const string& key, vector<string>& vals) const
{
    vals.clear();
    if (!setItr) return false;
    bool retVal = false;
    for (RawKeyValues::const_iterator itr = subLabelContentItr->second.lower_bound(key), itrEnd = subLabelContentItr->second.upper_bound(key); 
        itr != itrEnd; ++itr)
    {
        if (itr->second.empty()) continue;
        vals.push_back(itr->second);
        retVal = true;
    }
    return retVal;
}

bool ConfigFileReader::getSingleKeySingleVal(const string& key, bool& val) const
{
    string str;
    if (getValueString(key, str))
    {
        val = cvtStringToBool(str);
        return true;
    }
    return false;
}

bool ConfigFileReader::getSingleKeySingleVal(const string& key, int& val) const
{
    string str;
    if (getValueString(key, str))
    {
        val = cvtStringToInt(str);
        return true;
    }
    return false;
}

bool ConfigFileReader::getSingleKeySingleVal(const string& key, double& val) const
{
    string str;
    if (getValueString(key, str))
    {
        val = cvtStringToDouble(str);
        return true;
    }
    return false;
}

bool ConfigFileReader::getSingleKeySingleVal(const string& key, string& val) const
{
    return getValueString(key, val);
}

bool ConfigFileReader::getSingleKeyMultiVal(const string& key, vector<int>& vals) const
{
    vals.clear();
    string str;
    if (getValueString(key, str))
    {
        cvtStringToInts(str, vals);
        return true;
    }
    return false;
}

bool ConfigFileReader::getMultiKeySingleVal(const string& key, vector<int>& vals) const
{
    vals.clear();
    vector<string> strs;
    if (getValueStrings(key, strs))
    {
        int size = strs.size();
        vals.resize(size);
        for (int i = 0; i < size; i++)
            vals[i] = cvtStringToInt(strs[i]);
        return true;
    }
    return false;
}

bool ConfigFileReader::getMultiKeyMultiVal(const string& key, vector<vector<int> >& vals) const
{
    vals.clear();
    vector<string> strs;
    if (getValueStrings(key, strs))
    {
        int size = strs.size();
        vals.resize(size);
        for (int i = 0; i < size; i++)
            cvtStringToInts(strs[i], vals[i]);
        return true;
    }
    return false;
}

bool ConfigFileReader::readLabelContent(const string& label, ifstream& fileReader)
{
    char buf[1000];
    bool hasFound = false;
    if (label.empty())
    {
        bool hasFoundBracket = false;
        while (!fileReader.eof())
        {
            fileReader >> buf;
            if (buf[0] == '(' || buf[0] == '#')
            {
                if (!hasFoundBracket)
                    hasFound = true;
                break;
            }
            else if (buf[0] == '[')
                hasFoundBracket = true;
        }
    }
    else
    {
        while (!fileReader.eof())
        {
            fileReader >> buf;
            if (label == string(buf))
            {
                hasFound = true;
                break;
            }
        }
    }
    if (!hasFound)
    {
        return false;
    }

    if (label.empty())
        fileReader.seekg(0);
    string nextLabel;
    bool retVal = readLabelContent(label, fileReader, nextLabel);
    
    if (!label.empty() && label == nextLabel)
    {
        //throw string("ERROR in function ") + __FUNCTION__ + "(), " +
        //    "label " + label + " occurs more than once, not allowed";
        THROW_EXCEPT("label " + label + " occurs more than once, not allowed");
    }
    while (!fileReader.eof())
    {
        fileReader >> buf;
        if (!label.empty() && label == string(buf))
        {
            /*throw string("ERROR in function ") + __FUNCTION__ + "(), " +
                "label " + label + " occurs more than once, not allowed";*/
            THROW_EXCEPT("label " + label + " occurs more than once, not allowed");
        }
    }
    return retVal;
}

bool ConfigFileReader::readLabelContent(const std::string& currLabel, std::ifstream& fileReader, std::string& nextLabel)
{
    nextLabel.clear();
    
    char buf[1000];
    struct State
    {
        enum {LABEL, SUBLABEL, KEY, VALUE};
    };
    int state;

    string rawContentString;
    vector<string> rawContent;
    state = State::LABEL;
    // 从 fileReader 的指针的当前位置读入内容 放到 rawContent 当中
    // 直到遇到 label 的标识符 [ 或者文件结束为止
    while (true)
    {
        if (fileReader.eof())
        {
            if (state == State::LABEL)
            {
                // 如果文件为空文件 那么到达此处时 currLabel 是空字符串 不应该抛出异常
                // 仅当 currLabel 不是空字符串 并且到达文件末尾 才抛出异常
                /*if (!currLabel.empty())
                {
                    fileReader.close();
                    throw string("ERROR in function ") + __FUNCTION__ + "(), " +
                        "file ends with label " + currLabel + ", not allowed";
                }*/
            }
            else if (state == State::SUBLABEL)
            {
                /*fileReader.close();
                throw string("ERROR in function ") + __FUNCTION__ + "(), " +
                    "file ends with sub label " + rawContent.back() + ", not allowed";*/
            }
            else if (state == State::KEY)
            {
                /*fileReader.close();
                throw string("ERROR in function ") + __FUNCTION__ + "(), " +
                    "file ends with key " + rawContent.back() + ", not allowed";*/
            }
            else
            {
                rawContent.push_back(rawContentString);
                rawContentString.clear();
            }
            break;
        }

        fileReader >> buf;
        // 如果最后一个 value 后面是换行符 上面一行操作最后一次会读入一个空字符串 
        // 需要用下面的操作过滤掉这个空字符串
        if (buf[0] == 0)
            continue;

        // 读到 label 的开头 [ 可以结束本 while 循环
        if (buf[0] == '[')
        {
            if (state == State::LABEL)
            {
                /*fileReader.close();
                throw string("ERROR in function ") + __FUNCTION__ + "(), " +
                    "label " + currLabel + " followed by label " + buf + ", not allowed";*/
            }
            else if (state == State::SUBLABEL)
            {
                /*fileReader.close();
                throw string("ERROR in function ") + __FUNCTION__ + "(), " +
                    "sub label " + rawContent.back() + " followed by label " + buf + ", not allowed";*/
            }
            else if (state == State::KEY)
            {
                /*fileReader.close();
                throw string("ERROR in function ") + __FUNCTION__ + "(), " +
                    "key " + rawContent.back() + " followed by label " + buf + ", not allowed";*/
            }
            else
            {
                rawContent.push_back(rawContentString);
                rawContentString.clear();
            }
            nextLabel = buf;
            break;
        }

        switch (state)
        {
        case State::LABEL :
            if (buf[0] != '(' && buf[0] != '#')
            {
                fileReader.close();
                //throw string("ERROR in function ") + __FUNCTION__ + "(), " +
                //    "value " + buf + " not follow any key";
                THROW_EXCEPT(string("value ") + buf + " not follow any key");
            }
            else if (buf[0] == '(')
            {
                state = State::SUBLABEL;
                rawContent.push_back(buf);
            }
            else if (buf[0] == '#')
            {
                state = State::KEY;
                rawContent.push_back(buf);
            }
            break;
        case State::SUBLABEL :            
            if (buf[0] == '(')
            {
                /*fileReader.close();
                throw string("ERROR in function ") + __FUNCTION__ + "(), " +
                    "sub label " + rawContent.back() + " followed by sub label " + buf + ", " + 
                    "not allowed";*/
                state = State::SUBLABEL;
                rawContent.push_back(buf);
            }
            else if (buf[0] == '#')
            {
                state = State::KEY;
                rawContent.push_back(buf);
            }
            else
            {
                fileReader.close();
                /*throw string("ERROR in function ") + __FUNCTION__ + "(), " +
                    "sub label " + rawContent.back() + " followed by value " + buf + ", " + 
                    "not allowed";*/
                THROW_EXCEPT("sub label " + rawContent.back() + " followed by value " + buf + ", " + 
                    "not allowed");
            }
            break;
        case State::KEY :
            if (buf[0] == '(')
            {
                /*fileReader.close();
                throw string("ERROR in function ") + __FUNCTION__ + "(), " +
                    "key " + rawContent.back() + " followed by sub label " + buf + ", " + 
                    "not allowed";*/
                state = State::SUBLABEL;
                rawContent.push_back(buf);
            }
            else if (buf[0] == '#')
            {
                /*fileReader.close();
                throw string("ERROR in function ") + __FUNCTION__ + "(), " +
                    "key " + rawContent.back() + " followed by key " + buf + ", " + 
                    "not allowed";*/
                state = State::KEY;
                rawContent.push_back(buf);
            }
            else
            {
                state = State::VALUE;
                rawContentString = buf;
            }
            break;
        case State::VALUE :
            if (buf[0] == '(')
            {
                rawContent.push_back(rawContentString);
                rawContentString.clear();
                state = State::SUBLABEL;
                rawContent.push_back(buf);
            }
            else if (buf[0] == '#')
            {
                rawContent.push_back(rawContentString);
                rawContentString.clear();
                state = State::KEY;
                rawContent.push_back(buf);
            }
            else
            {
                rawContentString += " ";
                rawContentString += buf;
            }
            break;
        default:
            break;
        }
    }

    // 空文件
    if (rawContent.empty() && currLabel.empty() && nextLabel.empty())
        return false;
    // 空标签空内容
    if (rawContent.empty() && currLabel.empty() && !nextLabel.empty())
        return true;

    // 上面处理完之后 rawContent 中的内容都是符合规范的
    // 下面部分不会出现异常状况 导致部分条件语句后面 {} 中的内容为空
    int numOfRaw = rawContent.size();
    string subLabel, key;    
    RawKeyValues rawKeyValues;
    SubLabelContent subLabelContent;
    if (!numOfRaw)
    {

    }
    else if (rawContent[0][0] == '#')
    {
        state = State::KEY;
        key = rawContent[0];
    }
    else if (rawContent[0][0] == '(')
    {
        state = State::SUBLABEL;
        subLabel = rawContent[0];
    }
    else
    {
        
    }

    for (int i = 1; i < numOfRaw; i++)
    {
        switch (state)
        {
        case State::KEY :
            if (rawContent[i][0] == '(')
            {
                rawKeyValues.insert(make_pair(key, string()));
                subLabelContent.insert(make_pair(subLabel, rawKeyValues));
                rawKeyValues.clear();
                state = State::SUBLABEL;
                subLabel = rawContent[i];
                if (subLabelContent.find(subLabel) != subLabelContent.end())
                {
                    //throw string("ERROR in function ") + __FUNCTION__ + "(), " +
                    //    "sub label " + subLabel + " occurs more than once, not allowed";
                    THROW_EXCEPT("sub label " + subLabel + " occurs more than once, not allowed");
                }
            }
            else if (rawContent[i][0] == '#')
            {
                rawKeyValues.insert(make_pair(key, string()));
                state = State::KEY;
                key = rawContent[0];
            }
            else
            {
                state = State::VALUE;
                rawKeyValues.insert(make_pair(key, rawContent[i]));
            }
            break;
        case State::VALUE :            
            if (rawContent[i][0] == '(')
            {   
                subLabelContent.insert(make_pair(subLabel, rawKeyValues));
                rawKeyValues.clear();
                state = State::SUBLABEL;
                subLabel = rawContent[i];
                if (subLabelContent.find(subLabel) != subLabelContent.end())
                {
                    //throw string("ERROR in function ") + __FUNCTION__ + "(), " +
                    //    "sub label " + subLabel + " occurs more than once, not allowed";
                    THROW_EXCEPT("sub label " + subLabel + " occurs more than once, not allowed");
                }
            }
            else if (rawContent[i][0] == '#')
            {
                state = State::KEY;
                key = rawContent[i];
            }
            else
            {
                
            }
            break;
        case State::SUBLABEL :
            if (rawContent[i][0] == '(')
            {
                subLabelContent.insert(make_pair(subLabel, rawKeyValues));
                rawKeyValues.clear();
                state = State::SUBLABEL;
                subLabel = rawContent[i];
                if (subLabelContent.find(subLabel) != subLabelContent.end())
                {
                    //throw string("ERROR in function ") + __FUNCTION__ + "(), " +
                    //    "sub label " + subLabel + " occurs more than once, not allowed";
                    THROW_EXCEPT("sub label " + subLabel + " occurs more than once, not allowed");
                }
            }
            else if (rawContent[i][0] == '#')
            {
                state = State::KEY;
                key = rawContent[i];
            }
            else
            {
                
            }
            break;
        default:
            break;
        }
    }
    // 对最后的 key 和 value 对进行以下操作
    subLabelContent.insert(make_pair(subLabel, rawKeyValues));
    // 给 labelContent 添加新内容
    labelContent.insert(make_pair(currLabel, subLabelContent));
    return true;
}

}