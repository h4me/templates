#include <iostream>
#include <type_traits>
#include <vector>
#include <map>

#ifndef __code 
#define __code

enum { EnabledCrc64 = 0 };


using RootKey= std::conditional< EnabledCrc64, unsigned long long, std::string>::type;
using RootKeyPass = std::conditional<EnabledCrc64, RootKey, const RootKey& >::type;


class Context {
private:
     std::map<RootKey,std::string> map;
public:
      void setBlob(RootKeyPass key) const;         
};

#endif

