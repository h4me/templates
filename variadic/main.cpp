#include "code.hpp"


std::string Vec2String(const std::vector<int>& v)
{
         std::string s;
          for(auto item : v)
               s+= std::to_string(item) + "-";
         return s;
}


RootKey Vec2SInt(const std::vector<int>& v)
{
         RootKey s;
          for(auto item : v)
               s+= item;
         return s;
}


/*
std::string GetHash(const std::vector<int>& v, int c, int b) {

   std::string s;

       s = Vec2String(v) + std::to_string(c) + std::to_string(b);  
    
   return s;

}
*/
/*

inline std::string GetHash(int head)
{            
   return std::to_string(head);
}

inline std::string GetHash(const std::vector<int>& head)
{            
   return Vec2String(head);
}

template<class...T>
inline std::string GetHash(int head, T&&...args)
{
   std::string s;
   s = std::to_string(head) + GetHash(std::forward<T>(args)...);                
   return s;
}

template<class...T>
inline std::string GetHash(const std::vector<int>& head, T&&...args)
{
   std::string s;
   s = Vec2String(head) + GetHash(std::forward<T>(args)...);                
   return s;
}
*/


template<class R>
inline R Converter(int first)
{
     return first;
}

template<class R>
inline R Converter(int first, int sec)
{
     return first ^ sec;    
}

template<class R>
inline R Converter(const std::vector<int>& first, int sec)
{
     return first ^= sec;    
}




std::string Converter(T first)
{
     return first;
}


T Converter(T first, T sec)
{
     return first ^ sec;    
}




template< bool B, class T = void >
using enable_if_t = typename std::enable_if<B,T>::type;

namespace one {

inline RootKey GetHash(int head)
{            
   return std::to_string(head);
}

inline RootKey GetHash(const std::vector<int>& head)
{            
   return Vec2String(head);
}

template<class...T>
inline RootKey GetHash(int head, T&&...args)
{
   return std::to_string(head) + GetHash(std::forward<T>(args)...);                
}

template<class...T>
inline RootKey GetHash(const std::vector<int>& head, T&&...args)
{
   return Vec2String(head) + GetHash(std::forward<T>(args)...);                
}

}




 using namespace  one;

int main(int argc, char **argv) {

   std::vector<int> v = { 1 ,2 , 3 };

   auto key = GetHash(v,4,5,6,7,4,5); 
 
   //std::cout << GetHash(v) << std::endl;

   Context ctx;

   ctx.setBlob(key);

    return 0;
}