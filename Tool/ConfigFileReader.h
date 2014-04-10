#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <utility>

namespace ztool
{
/*!
    解析配置文件的类

    配置文件由 标签 LABEL 子标签 SUBLABEL 键 KEY 和值 VALUE 构成
    LABEL 由 [ 开头
    SUBLABEL 由 ( 开头
    KEY 由 # 开头
    VALUE 可以是除了上述字符之外的任意字符组成 
    例如 
    [Config] 是一个 LABEL
    (Config) 是一个 SUBLABEL
    #Config yes 中 #Config 是 KEY, yes 是 VALUE
    
    同一个配置文件中的标签必须互不相同
    同一个标签下属的子标签必须互不相同
    同一个子标签下属的键可以重复
    一个键后面可以跟多个并列的值直到文件末尾, 或者直到下一个键, 或者直到下一个子标签, 或者直到下一个标签

    标签, 子标签, 和键的内容可以为空
    下面的配置文件是合法的 配置文件之间用横线隔开
    -----------------------------------
    [LabelA]
    [LabelB]
      (SubLabelBA)
        #KeyBAA valBAAA
        #KeyBAB valBABA valBABB valBABC
    -----------------------------------
    [LabelA]
      (SubLabelAA)
        #KeyAAA valAAAA valAAAB
        #KeyAAB valAABA valAABB valAABC
      (SubLabelAB)
      (SubLabelAC)
        #KeyACA valACAA
      (SubLabelAD)
        #KeyADA
    -----------------------------------

    值不能独立存在
    下面的配置文件是非法的 配置文件之间用横线隔开
    -----------------------------------
    val
    -----------------------------------
    [LabelA]
        val
    [LabelB]
        val
      (SubLabelBA)
        val
    -----------------------------------

    可以有未指定标签和子标签的键-值组合
    这些匿名标签和匿名子标签的键-值组合必须位于配置文件最开头的部分, 他们的标签和子标签都被定义成空字符串 ""
    也可以有未指定标签但是指定了子标签的键-值组合
    这些有名标签并且匿名子标签的键-值组合必须跟在标签后面, 标签下属的第一个子标签之前, 他们的子标签是空字符串 ""
    下面的配置文件是合法的 配置文件之间用横线隔开
    -----------------------------------
        #KeyA valA
        #KeyB valB
      (SubLabelA)
        #KeyAA valAAA
      (SubLabelB)
        #KeyBA valBAA valBAB
    [LabelA]
        #KeyA valAA valAB
      (SubLabelA)
        #KeyAA valAAA
        #KeyAB valABA
      (SubLabelB)
        #KeyBA valBAA
    [LabelB]
        #keyB valBA
    -----------------------------------
 */
class ConfigFileReader
{
public:
    /*!
        构造函数, 指定需要处理的配置文件名 cfgFileName
     */
    ConfigFileReader(const std::string& cfgFileName);
    /*!
        文件能打开, 返回 true, 文件不能打开, 返回 false
     */
    bool canBeOpened(void) const;
    /*!
        读取配置文件中指定标签 label 的内容
        如果读取成功, 返回 true, 如果配置文件不能打开或者 label 不存在, 返回 false
        如果配置文件格式不合规范, 则会抛出 std::string 类型的异常
     */
    bool read(const std::string& label);
    /*!
        读取配置文件中的所有内容
        如果配置文件不为空, 返回 true, 如果配置文件不能打开或者为空, 返回 false
        如果配置文件格式不合规范, 则会抛出 std::string 类型的异常
     */
    bool readAll(void);
    /*!
        调用 read 或者 readAll 函数之后, 打印读到的内容
     */
    void print(void) const;
    /*!
        调用 read 或者 readAll 之后, 查找标签等于 label 并且子标签等于 subLabel 的下属的键-值组合
        如果找到, 返回 true, 没找到则返回 false
        匿名标签或者匿名子标签写空字符串 "" 或者 std::string()
     */
    bool seek(const std::string& label, const std::string& subLabel);
    /*!
        调用 read 之后, 查找子标签 subLabel 下属的键-值组合
        如果找到, 返回 true, 没找到则返回 false
        如果调用 readAll 之后调用这个函数, 返回 false
        匿名部分写空字符串 "" 或者 std::string()
     */
    bool seek(const std::string& subLabel);
    /*!
        seek 之后, 读取第一个键等于 key 的键-值组合的值 val, 读出的是 int 型的值
        如果有多个相同的 key, 则读出的 val 是私有成员变量 labelContent 中
        存储的多个键等于 key 的键-值组合中的第一个键-值组合的值, 不一定是原配置文件中的第一个键-值组合的值
     */
    bool getSingleKeySingleVal(const std::string& key, bool& val) const;
    /*!
        seek 之后, 读取第一个键等于 key 的键-值组合的值 val, 读出的是 int 型的值
        如果有多个相同的 key, 则读出的 val 是私有成员变量 labelContent 中
        存储的多个键等于 key 的键-值组合中的第一个键-值组合的值, 不一定是原配置文件中的第一个键-值组合的值
     */
    bool getSingleKeySingleVal(const std::string& key, int& val) const;
    /*!
        seek 之后, 读取第一个键等于 key 的键-值组合的值 val, 读出的是 double 型的值
        如果有多个相同的 key, 则读出的 val 是私有成员变量 labelContent 中
        存储的多个键等于 key 的键-值组合中的第一个键-值组合的值, 不一定是原配置文件中的第一个键-值组合的值
        读取成功则返回 true, 没有读到 key 对应的值, 返回 false
     */
    bool getSingleKeySingleVal(const std::string& key, double& val) const;
    /*!
        seek 之后, 读取第一个键等于 key 的键-值组合的值 val, 读出的是 int 型的值
        如果有多个相同的 key, 则读出的 val 是私有成员变量 labelContent 中
        存储的多个键等于 key 的键-值组合中的第一个键-值组合的值, 不一定是原配置文件中的第一个键-值组合的值
        读取成功则返回 true, 没有读到 key 对应的值, 返回 false
     */
    bool getSingleKeySingleVal(const std::string& key, std::string& val) const;
    /*!
        seek 之后, 读取第一个键等于 key 的键-值组合的值 vals, 读出的是 int 型的值
        vals.size() 等于第一个键等于 key 的键-值组合的值部分 int 型数字的个数
        如果有多个相同的 key, 则读出的 vals 是私有成员变量 labelContent 中
        存储的多个键等于 key 的键-值组合中的第一个键-值组合的值, 不一定是原配置文件中的第一个键-值组合的值
        读取成功则返回 true, 没有读到 key 对应的值, 返回 false
     */
    bool getSingleKeyMultiVal(const std::string& key, std::vector<int>& vals) const;
    /*!
        seek 之后, 读取键等于 key 的所有键-值组合的值 vals, 读出的是 int 型的值
        vals.size() 等于键等于 key 的键值组合的数量
        读取成功则返回 true, 没有读到 key 对应的值, 返回 false
     */
    bool getMultiKeySingleVal(const std::string& key, std::vector<int>& vals) const;
    /*!
        seek 之后, 读取键等于 key 的所有键-值组合的值 vals, 读出的是 int 型的值
        vals.size() 等于键等于 key 的键-值组合的数量
        vals[i].size() 等于第 i 个键等于 key 的键-值组合中 int 型数字的个数,  i >= 0 && i < vals.size()
        读取成功则返回 true, 没有读到 key 对应的值, 返回 false
     */
    bool getMultiKeyMultiVal(const std::string& key, std::vector<std::vector<int> >& vals) const;

private:
    //! 存储一个子标签下属的键-值对的类型
    typedef std::multimap<std::string, std::string> RawKeyValues;
    //! 存储一个标签下属的子标签和子标签下属的键-值对的类型
    typedef std::map<std::string, RawKeyValues> SubLabelContent;
    //! 存储一个配置文件中所有标签和它下属的内容的类型
    typedef std::map<std::string, SubLabelContent> LabelContent;
    //! 读取模式
    struct ReadMode
    {
        enum 
        {
            UNREAD,    ///< 没有读任何内容, 或者读取失败
            READ_ALL,  ///< 调用了 readAll 函数, 成功读取了配置文件的所有内容
            READ_LABEL ///< 调用了 read 函数, 成功读取了配置文件中某个标签下属的所有内容
        };
    };

    //! 禁用拷贝构造函数
    ConfigFileReader(const ConfigFileReader&);
    //! 禁用赋值等号
    ConfigFileReader& operator=(const ConfigFileReader&);

    /*!
        fileReader 刚刚读出一个标签 currLabel, 从文件中读取出 currLabel 的所有内容, 保存到私有成员变量 labelContent 中
        如果遇到下一个标签, 则赋值给 nextLabel, 如果读到文件末尾, 则 nextLabel 等于空字符串
        如果标签 currLabel 没有内容, 或者配置文件不和规范, 则抛出 std::string 类型的异常, 抛出异常前会关闭文件 fileReader
        如果遇到文件末尾, 返回 false, 如果操作成功, 则返回 true
        本函数被 bool readAll() 调用
     */
    bool readLabelContent(const std::string& currLabel, std::ifstream& fileReader, std::string& nextLabel);
    /*!
        fileReader 刚刚读出一个标签 label, 从文件中读取出 label 的所有内容, 保存到私有成员变量 labelContent 中
        如果遇到文件末尾, 返回 false
        如果标签 label 没有内容, 或者配置文件不合规范, 则抛出 std::string 类型的异常, 抛出异常前会关闭文件 fileReader
        如果操作成功, 则返回 true
        本函数被 bool read(const std::string& label) 调用
     */
    bool readLabelContent(const std::string& label, std::ifstream& fileReader);
    /*!
        找到第一个键等于 key 的键-值组合, 取出值字符串, 赋值给 val
     */
    bool getValueString(const std::string& key, std::string& val) const;
    /*!
        找出键等于 key 的所有键-值组合, 取出所有字符串, 赋值给 vals
        vals.size() 等于键等于 key 的键-值组合的数量
     */
    bool getValueStrings(const std::string& key, std::vector<std::string>& vals) const;

    std::string fileName;         ///< 文件名
    bool fileExists;              ///< 文件是否存在
    int readMode;                 ///< 读取模式
    LabelContent labelContent;    ///< 读取到的内容
    bool setItr;                  ///< 是否已经设置了迭代器
    //! seek 函数运行成功之后, 设置指向特定子标签的迭代器
    SubLabelContent::const_iterator subLabelContentItr;
};

}