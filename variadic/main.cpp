#include <unordered_map>
#include <algorithm>
#include <set>
#include <map>

#include <iomanip>
#include "code.hpp"

#include "cpu_profiler.hpp"


namespace mkldnn {
     namespace memory {
          typedef std::vector<int> dims;
     }
}

 static std::string dims2str(const mkldnn::memory::dims& operand_dims) {
    std::string dstr = "";
    for (size_t i = 0; i < operand_dims.size(); ++i) {
      dstr += std::to_string(operand_dims[i]) + "-";
    }
    return dstr;
  }

static std::string GetHashJacek(const mkldnn::memory::dims& input_dims,    // NOLINT
                             const mkldnn::memory::dims& weights_dims,  // NOLINT
                             const std::vector<int>& strides,           // NOLINT
                             const std::vector<int>& paddings,          // NOLINT
                             const std::vector<int>& dilations,         // NOLINT
                             int groups) {

    

    return dims2str(input_dims) + dims2str(weights_dims) + dims2str(strides) +
           dims2str(paddings) + dims2str(dilations) + std::to_string(groups) ;
           
  }






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

/*
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

*/

typedef unsigned long long u64;
#define CONST64(n) n##ull
static const u64 CRC64_Table[256] = {
	CONST64(0x0000000000000000), CONST64(0x42f0e1eba9ea3693),
	CONST64(0x85e1c3d753d46d26), CONST64(0xc711223cfa3e5bb5),
	CONST64(0x493366450e42ecdf), CONST64(0x0bc387aea7a8da4c),
	CONST64(0xccd2a5925d9681f9), CONST64(0x8e224479f47cb76a),
	CONST64(0x9266cc8a1c85d9be), CONST64(0xd0962d61b56fef2d),
	CONST64(0x17870f5d4f51b498), CONST64(0x5577eeb6e6bb820b),
	CONST64(0xdb55aacf12c73561), CONST64(0x99a54b24bb2d03f2),
	CONST64(0x5eb4691841135847), CONST64(0x1c4488f3e8f96ed4),
	CONST64(0x663d78ff90e185ef), CONST64(0x24cd9914390bb37c),
	CONST64(0xe3dcbb28c335e8c9), CONST64(0xa12c5ac36adfde5a),
	CONST64(0x2f0e1eba9ea36930), CONST64(0x6dfeff5137495fa3),
	CONST64(0xaaefdd6dcd770416), CONST64(0xe81f3c86649d3285),
	CONST64(0xf45bb4758c645c51), CONST64(0xb6ab559e258e6ac2),
	CONST64(0x71ba77a2dfb03177), CONST64(0x334a9649765a07e4),
	CONST64(0xbd68d2308226b08e), CONST64(0xff9833db2bcc861d),
	CONST64(0x388911e7d1f2dda8), CONST64(0x7a79f00c7818eb3b),
	CONST64(0xcc7af1ff21c30bde), CONST64(0x8e8a101488293d4d),
	CONST64(0x499b3228721766f8), CONST64(0x0b6bd3c3dbfd506b),
	CONST64(0x854997ba2f81e701), CONST64(0xc7b97651866bd192),
	CONST64(0x00a8546d7c558a27), CONST64(0x4258b586d5bfbcb4),
	CONST64(0x5e1c3d753d46d260), CONST64(0x1cecdc9e94ace4f3),
	CONST64(0xdbfdfea26e92bf46), CONST64(0x990d1f49c77889d5),
	CONST64(0x172f5b3033043ebf), CONST64(0x55dfbadb9aee082c),
	CONST64(0x92ce98e760d05399), CONST64(0xd03e790cc93a650a),
	CONST64(0xaa478900b1228e31), CONST64(0xe8b768eb18c8b8a2),
	CONST64(0x2fa64ad7e2f6e317), CONST64(0x6d56ab3c4b1cd584),
	CONST64(0xe374ef45bf6062ee), CONST64(0xa1840eae168a547d),
	CONST64(0x66952c92ecb40fc8), CONST64(0x2465cd79455e395b),
	CONST64(0x3821458aada7578f), CONST64(0x7ad1a461044d611c),
	CONST64(0xbdc0865dfe733aa9), CONST64(0xff3067b657990c3a),
	CONST64(0x711223cfa3e5bb50), CONST64(0x33e2c2240a0f8dc3),
	CONST64(0xf4f3e018f031d676), CONST64(0xb60301f359dbe0e5),
	CONST64(0xda050215ea6c212f), CONST64(0x98f5e3fe438617bc),
	CONST64(0x5fe4c1c2b9b84c09), CONST64(0x1d14202910527a9a),
	CONST64(0x93366450e42ecdf0), CONST64(0xd1c685bb4dc4fb63),
	CONST64(0x16d7a787b7faa0d6), CONST64(0x5427466c1e109645),
	CONST64(0x4863ce9ff6e9f891), CONST64(0x0a932f745f03ce02),
	CONST64(0xcd820d48a53d95b7), CONST64(0x8f72eca30cd7a324),
	CONST64(0x0150a8daf8ab144e), CONST64(0x43a04931514122dd),
	CONST64(0x84b16b0dab7f7968), CONST64(0xc6418ae602954ffb),
	CONST64(0xbc387aea7a8da4c0), CONST64(0xfec89b01d3679253),
	CONST64(0x39d9b93d2959c9e6), CONST64(0x7b2958d680b3ff75),
	CONST64(0xf50b1caf74cf481f), CONST64(0xb7fbfd44dd257e8c),
	CONST64(0x70eadf78271b2539), CONST64(0x321a3e938ef113aa),
	CONST64(0x2e5eb66066087d7e), CONST64(0x6cae578bcfe24bed),
	CONST64(0xabbf75b735dc1058), CONST64(0xe94f945c9c3626cb),
	CONST64(0x676dd025684a91a1), CONST64(0x259d31cec1a0a732),
	CONST64(0xe28c13f23b9efc87), CONST64(0xa07cf2199274ca14),
	CONST64(0x167ff3eacbaf2af1), CONST64(0x548f120162451c62),
	CONST64(0x939e303d987b47d7), CONST64(0xd16ed1d631917144),
	CONST64(0x5f4c95afc5edc62e), CONST64(0x1dbc74446c07f0bd),
	CONST64(0xdaad56789639ab08), CONST64(0x985db7933fd39d9b),
	CONST64(0x84193f60d72af34f), CONST64(0xc6e9de8b7ec0c5dc),
	CONST64(0x01f8fcb784fe9e69), CONST64(0x43081d5c2d14a8fa),
	CONST64(0xcd2a5925d9681f90), CONST64(0x8fdab8ce70822903),
	CONST64(0x48cb9af28abc72b6), CONST64(0x0a3b7b1923564425),
	CONST64(0x70428b155b4eaf1e), CONST64(0x32b26afef2a4998d),
	CONST64(0xf5a348c2089ac238), CONST64(0xb753a929a170f4ab),
	CONST64(0x3971ed50550c43c1), CONST64(0x7b810cbbfce67552),
	CONST64(0xbc902e8706d82ee7), CONST64(0xfe60cf6caf321874),
	CONST64(0xe224479f47cb76a0), CONST64(0xa0d4a674ee214033),
	CONST64(0x67c58448141f1b86), CONST64(0x253565a3bdf52d15),
	CONST64(0xab1721da49899a7f), CONST64(0xe9e7c031e063acec),
	CONST64(0x2ef6e20d1a5df759), CONST64(0x6c0603e6b3b7c1ca),
	CONST64(0xf6fae5c07d3274cd), CONST64(0xb40a042bd4d8425e),
	CONST64(0x731b26172ee619eb), CONST64(0x31ebc7fc870c2f78),
	CONST64(0xbfc9838573709812), CONST64(0xfd39626eda9aae81),
	CONST64(0x3a28405220a4f534), CONST64(0x78d8a1b9894ec3a7),
	CONST64(0x649c294a61b7ad73), CONST64(0x266cc8a1c85d9be0),
	CONST64(0xe17dea9d3263c055), CONST64(0xa38d0b769b89f6c6),
	CONST64(0x2daf4f0f6ff541ac), CONST64(0x6f5faee4c61f773f),
	CONST64(0xa84e8cd83c212c8a), CONST64(0xeabe6d3395cb1a19),
	CONST64(0x90c79d3fedd3f122), CONST64(0xd2377cd44439c7b1),
	CONST64(0x15265ee8be079c04), CONST64(0x57d6bf0317edaa97),
	CONST64(0xd9f4fb7ae3911dfd), CONST64(0x9b041a914a7b2b6e),
	CONST64(0x5c1538adb04570db), CONST64(0x1ee5d94619af4648),
	CONST64(0x02a151b5f156289c), CONST64(0x4051b05e58bc1e0f),
	CONST64(0x87409262a28245ba), CONST64(0xc5b073890b687329),
	CONST64(0x4b9237f0ff14c443), CONST64(0x0962d61b56fef2d0),
	CONST64(0xce73f427acc0a965), CONST64(0x8c8315cc052a9ff6),
	CONST64(0x3a80143f5cf17f13), CONST64(0x7870f5d4f51b4980),
	CONST64(0xbf61d7e80f251235), CONST64(0xfd913603a6cf24a6),
	CONST64(0x73b3727a52b393cc), CONST64(0x31439391fb59a55f),
	CONST64(0xf652b1ad0167feea), CONST64(0xb4a25046a88dc879),
	CONST64(0xa8e6d8b54074a6ad), CONST64(0xea16395ee99e903e),
	CONST64(0x2d071b6213a0cb8b), CONST64(0x6ff7fa89ba4afd18),
	CONST64(0xe1d5bef04e364a72), CONST64(0xa3255f1be7dc7ce1),
	CONST64(0x64347d271de22754), CONST64(0x26c49cccb40811c7),
	CONST64(0x5cbd6cc0cc10fafc), CONST64(0x1e4d8d2b65facc6f),
	CONST64(0xd95caf179fc497da), CONST64(0x9bac4efc362ea149),
	CONST64(0x158e0a85c2521623), CONST64(0x577eeb6e6bb820b0),
	CONST64(0x906fc95291867b05), CONST64(0xd29f28b9386c4d96),
	CONST64(0xcedba04ad0952342), CONST64(0x8c2b41a1797f15d1),
	CONST64(0x4b3a639d83414e64), CONST64(0x09ca82762aab78f7),
	CONST64(0x87e8c60fded7cf9d), CONST64(0xc51827e4773df90e),
	CONST64(0x020905d88d03a2bb), CONST64(0x40f9e43324e99428),
	CONST64(0x2cffe7d5975e55e2), CONST64(0x6e0f063e3eb46371),
	CONST64(0xa91e2402c48a38c4), CONST64(0xebeec5e96d600e57),
	CONST64(0x65cc8190991cb93d), CONST64(0x273c607b30f68fae),
	CONST64(0xe02d4247cac8d41b), CONST64(0xa2dda3ac6322e288),
	CONST64(0xbe992b5f8bdb8c5c), CONST64(0xfc69cab42231bacf),
	CONST64(0x3b78e888d80fe17a), CONST64(0x7988096371e5d7e9),
	CONST64(0xf7aa4d1a85996083), CONST64(0xb55aacf12c735610),
	CONST64(0x724b8ecdd64d0da5), CONST64(0x30bb6f267fa73b36),
	CONST64(0x4ac29f2a07bfd00d), CONST64(0x08327ec1ae55e69e),
	CONST64(0xcf235cfd546bbd2b), CONST64(0x8dd3bd16fd818bb8),
	CONST64(0x03f1f96f09fd3cd2), CONST64(0x41011884a0170a41),
	CONST64(0x86103ab85a2951f4), CONST64(0xc4e0db53f3c36767),
	CONST64(0xd8a453a01b3a09b3), CONST64(0x9a54b24bb2d03f20),
	CONST64(0x5d45907748ee6495), CONST64(0x1fb5719ce1045206),
	CONST64(0x919735e51578e56c), CONST64(0xd367d40ebc92d3ff),
	CONST64(0x1476f63246ac884a), CONST64(0x568617d9ef46bed9),
	CONST64(0xe085162ab69d5e3c), CONST64(0xa275f7c11f7768af),
	CONST64(0x6564d5fde549331a), CONST64(0x279434164ca30589),
	CONST64(0xa9b6706fb8dfb2e3), CONST64(0xeb46918411358470),
	CONST64(0x2c57b3b8eb0bdfc5), CONST64(0x6ea7525342e1e956),
	CONST64(0x72e3daa0aa188782), CONST64(0x30133b4b03f2b111),
	CONST64(0xf7021977f9cceaa4), CONST64(0xb5f2f89c5026dc37),
	CONST64(0x3bd0bce5a45a6b5d), CONST64(0x79205d0e0db05dce),
	CONST64(0xbe317f32f78e067b), CONST64(0xfcc19ed95e6430e8),
	CONST64(0x86b86ed5267cdbd3), CONST64(0xc4488f3e8f96ed40),
	CONST64(0x0359ad0275a8b6f5), CONST64(0x41a94ce9dc428066),
	CONST64(0xcf8b0890283e370c), CONST64(0x8d7be97b81d4019f),
	CONST64(0x4a6acb477bea5a2a), CONST64(0x089a2aacd2006cb9),
	CONST64(0x14dea25f3af9026d), CONST64(0x562e43b4931334fe),
	CONST64(0x913f6188692d6f4b), CONST64(0xd3cf8063c0c759d8),
	CONST64(0x5dedc41a34bbeeb2), CONST64(0x1f1d25f19d51d821),
	CONST64(0xd80c07cd676f8394), CONST64(0x9afce626ce85b507)
};

typedef unsigned int unint;
typedef unsigned char u8;

u64 update_crc64 (const void *buf, unint len, u64 crc)
{
	const u8	*b = reinterpret_cast<const u8*>(buf);
	unint		i;
	for (i = 0; i < len; i++) {
		crc = CRC64_Table[(u8)(crc >> 56) ^ *b++] ^ (crc << 8);
	}
	return ~crc;
}

//#define diagme(x) std::cout << x << std::endl;
#define diagme(x) ;


namespace debug { 
template<int N=0>
struct Converter {
   typedef Converter<N+1> next_one;
   enum {  value = N };

   static std::string convert(int input) {
         return std::to_string(input);
   }

   static std::string convert(const std::vector<int>& input) {
          std::string s;
          for(auto item : input)
               s+= std::to_string(item) + "-";
          return s;
   }

   static std::string convert(const std::vector<int>& input, const std::string& left) {
          return convert(input) + left;
   }

   static std::string convert(int input, const std::string& left) {
          return convert(input) + left;
   }

};

}

template<int N=0>
struct Converter {
   typedef Converter<N+1> next_one;
  // enum {  value = N, mod_index = N % 255 };
   enum {  value = N, mod_index = 0 };

   union CastMe {
         int value;
         unsigned char bytes[4];   
   };
   
   static u64 convert_algo(int input, u64 crc) {
          diagme( " u64 convert_algo(int input="<< std::hex << input <<", u64 crc=" << crc << " ) " << value )   
          static_assert(sizeof(int) == sizeof(CastMe), " Is not ok");
         
            // auto* Cast = reinterpret_cast<CastMe*>(&input);
            for(int i=0;i<4;++i)
             
             crc = CRC64_Table[(u8) (crc >> 56 ) ^ reinterpret_cast<CastMe*>(&input)->bytes[i]   ] ^ (crc << 8);
         /*
             crc = CRC64_Table[(u8) (crc >> 56 ) ^ reinterpret_cast<CastMe*>(&input)->bytes[1]   ] ^ (crc << 8);
         
             crc = CRC64_Table[(u8) (crc >> 56 ) ^ reinterpret_cast<CastMe*>(&input)->bytes[2]   ] ^ (crc << 8);
         
             crc = CRC64_Table[(u8) (crc >> 56 ) ^ reinterpret_cast<CastMe*>(&input)->bytes[3]   ] ^ (crc << 8);
         */
         return ~crc;
         /*
          for(int i=0;i<4;++i)
          {
            unsigned char byte =  *(reinterpret_cast<unsigned char*>(&input) + i);  
            diagme( " loop["<<i<<"]  " << ((int) byte )  )   
             
          //   std::cout << (char) (*(reinterpret_cast<u8*>(&input)) + i) ;
             
             crc = CRC64_Table[(u8) (crc >> 56 ) ^ byte   ] ^ (crc << 8);
          }
         // std::cout << std::endl;
          return ~crc;
          */
   }

   
   static u64 convert_algo(const std::vector<int>& input, u64 crc) {
         diagme( " u64 convert_algo(const std::vector<int>& input_size="<< input.size() <<", u64 crc ) " << value )   
   
          for(size_t i=0;i<input.size();++i)
              crc = convert_algo(input[i],crc);
          return ~crc;
       
   }


   static u64 convert_algo(const std::string& input, u64 crc) {
         
          for(size_t i=0;i<input.size();++i)
              crc = convert_algo( (int) input[i] ,crc);
          return ~crc;
       
   }



     static u64 convert(int input) {
             diagme( " u64 convert(int input="<< input <<") " << value )   
   
             return convert_algo(input,CRC64_Table[(u8)mod_index]);
     }

     static u64 convert(const std::string& input) {
             return convert_algo(input,CRC64_Table[(u8)mod_index]);
     }

/*
   static u64 convert(int input) {
         

         auto crc = CRC64_Table[(u8)mod_index]; 
       //  std::cout << "Mod index" << mod_index << std::endl;
         for(int i=0;i<4;++i)
          // crc = CRC64_Table[(u8) ((mod_index + i) % 255) ^ (*(reinterpret_cast<u8*>(&input)) + i)   ] ^ (crc << 8);
           crc = CRC64_Table[(u8) (crc >> 56) ^ (*(reinterpret_cast<u8*>(&input)) + i)   ] ^ (crc << 8);
          
          // crc ^= CRC64_Table[(u8) ((mod_index + i) % 255)  ] ;

          // CRC64_Table[(u8)(crc >> 56) ^ *b++] ^ (crc << 8);
          return ~crc;
        // return std::to_string(input);
   }
*/
   static u64 convert(const std::vector<int>& input) {
         
        //  u64 s = CRC64_Table[(u8)mod_index];
        //  for(auto item : input)
         //         s = convert_algo(item,s) ;
    
          diagme( " u64 convert(const std::vector<int>& input_size="<< input.size() <<") " << value )   
    
          return  convert_algo(input, CRC64_Table[(u8)mod_index] );
   }

   static u64 convert(const std::vector<int>& input, u64 left) {
           diagme( " u64 convert(const std::vector<int>& input_size="<< input.size() <<", u64 left="<< std::hex<< left << ") " << value )   
          return convert_algo(input,left);
   }

   static u64 convert(int input, u64 left) {
           diagme( " u64 convert(int input="<< std::hex<< input <<", u64 left="<< left << ") " << value )   
          return convert_algo(input,left) ;
   }

};






template< bool B, class T = void >
using enable_if_t = typename std::enable_if<B,T>::type;

namespace one {


template<class S=Converter<>>
inline auto GetHash(int head) -> decltype(S::convert(head))
{  
   diagme( " GetHash(int head)  " << S::value )               
   return S::convert(head);
}
template<class S=Converter<>>
inline auto GetHash(const std::vector<int>& head) -> decltype(S::convert(head))
{  
      diagme( "GetHash(const std::vector<int>& head) " << S::value )            
   return S::convert(head);
}


template<class S=Converter<>>
inline auto GetHash(const std::string& head) -> decltype(S::convert(head))
{  
    diagme( "GetHash(const std::string& head) " << S::value )           
   return S::convert(head);
}


template<class S=Converter<>, class...T>
inline auto GetHash(const std::string& head, T&&...args) -> decltype(S::convert(head))
{
  // return S::convert(head) + GetHash<typename S::type>(std::forward<T>(args)...);
  diagme( "GetHash(const std::string& head, T&&...args) " << S::value )
   return S::convert(head, GetHash<typename S::next_one>(std::forward<T>(args)...));                
}


template<class S=Converter<>, class...T>
inline auto GetHash(int head, T&&...args) -> decltype(S::convert(head))
{
      diagme( "GetHash(int head, T&&...args) " << S::value )
  // return S::convert(head) + GetHash<typename S::type>(std::forward<T>(args)...);
   return S::convert(head, GetHash<typename S::next_one>(std::forward<T>(args)...));                
}

template<class S=Converter<>, class...T>
inline auto GetHash(const std::vector<int>& head, T&&...args) -> decltype(S::convert(head))
{
  // return S::convert(head) + GetHash<typename S::type>(std::forward<T>(args)...);
   diagme( "GetHash(const std::vector<int>& head, T&&...args) " << S::value )
   return S::convert(head, GetHash<typename S::next_one>(std::forward<T>(args)...));                
}

}

static int counter=0;

namespace perf {

   template<class K, class V=int>
   //using MAP = std::unordered_map<K,V>;
  using MAP = std::map<K,V>;
  
   MAP<std::string> map_string;
   MAP<u64> map_u64;
   std::vector<u64> vec_u64;
   
}

INIT_PERF(); 


template<class ... T>
void UnitTest(std::map<u64,std::string>& s, T&&...args) {
       
     int error = 0;
       
MAKE_PERF_VAR(); 

   BEGIN(); 
       auto value = one::GetHash<>(std::forward<T>(args)...);
   END("CRC64");    

   BEGIN();

           perf::map_u64[value]=1;
   END("CRC64-MAP-INSERT");
   BEGIN();
    
           if(perf::map_u64.find(value)!=perf::map_u64.end()  )  
           {
               // std::cout << "error" << std::endl;
                error++; 
           }   

   END("CRC64-MAP-FIND");
   BEGIN();
            perf::vec_u64.push_back(value);
   END("CRC64-vec-insert");


   BEGIN();
          if( std::find(perf::vec_u64.begin(),perf::vec_u64.end(),value) !=  perf::vec_u64.end() )
           {
             //    std::cout << "error" << std::endl;
                 error++;
           }          

   END("CRC64-vec-find");



   BEGIN();
     std::hash<u64> hh_u64;
     std::size_t hash_u64 = hh_u64(value); 
   END("std::hash<u64>");
          std::cout << "hash_u64 " << std::hex << hash_u64 << std::endl;

   BEGIN(); 
          std::string my = one::GetHash<debug::Converter<>>(std::forward<T>(args)... );
   END("CRC64-MyString");

   BEGIN(); 
          std::string jacek = GetHashJacek(std::forward<T>(args)...);
   END("Jacek");

   BEGIN(); 
            perf::map_string[jacek]=1;

   END("JacekMap-INSERT");
  BEGIN(); 
           if(perf::map_string.find(jacek)!=perf::map_string.end())
           {
                error++;
              //  std::cout << "error" << std::endl;
           }   


   END("JacekMap-FIND"); 
   BEGIN();
       std::hash<std::string> hh;
       std::size_t hash_hh = hh(jacek); 
  
   END("std::hash<std::string>");
  
         std::cout << "std::Hash-str :" << hash_hh << std::endl; 
        // std::string big_key = "12-3-4-5 " + jacek;       

   BEGIN();

        std::size_t kkkk_my= one::GetHash<>(jacek);

   END("big-my");

   BEGIN();

         std::size_t kkkk_std= hh(jacek);

   END("big-hash<std::string>");


    std::cout << "kkkk_my=" << kkkk_my  << " " <<  kkkk_std << " error" << error << std::endl;

       counter++;
       auto it = s.find(value);
       if(it!=s.end())
       {
            std::cout << "\n\n\nUnit Test Fails! "<<  std::hex << value << " test=" << counter << "my :" << my << 
            " has conflict with " << it->second << "\n\n\n ###############FAIL####" << std::endl;
            exit (1);
       } 

       s[value]= my;
       std::cout << "[TEST OK] :Uniq Key " << std::hex << value << std::endl;           

       std::cout << jacek << std::endl;

 

 

 

}

 using namespace  one;

 

int main(int argc, char **argv) {
 
 
 BEGIN_OVERALL();
 

 


 

   std::map<u64,std::string> check_set;
   

    for(int i=0;i<1400;++i)
    {

    std::vector<int> v1{1,2,5,6}; 
    std::vector<int> v2{1,2,5,6};
    std::vector<int> v3{1,2,5,6};
    std::vector<int> v4{1,2,5,6};
    std::vector<int> v5{i,2,5,6};

   // std::vector<int> v6{i+40,2,5,6};
    
    UnitTest(check_set, v1,v2,v3,v4,v5, 5);
    
    }
 /*
   UnitTest(check_set, std::vector<int>{1,4,5,6 },4,5,6 );
   UnitTest(check_set, std::vector<int>{0,4,5,6},4,5,6 );
   UnitTest(check_set, std::vector<int>{1,2,5,6},4,5,6 );
   UnitTest(check_set, std::vector<int>{1,2,5,6},5,5,6 );

   for(int i=0;i<30;++i)
         UnitTest(check_set, std::vector<int>{1,2,5,6},5,5,3,i);
 
   for(int i=0;i<30;++i)
         UnitTest(check_set, std::vector<int>{1,2,5,6},i,i,i);
 
  for(int i=3;i<30;++i)
         UnitTest(check_set, std::vector<int>{1,i,5,6},i,i,i);
 
 for(int i=3;i<30;++i)
         UnitTest(check_set, std::vector<int>{1,i,5,i+6},i,i,i);
 

 for(int i=3;i<30;++i)
         UnitTest(check_set, std::vector<int>{1,i,5,i+6},i,i+1,i+2);
 

 for(int i=3;i<30;++i)
         UnitTest(check_set, std::vector<int>{1+i,i,5,i+6},i,i+1,i+2);
 
 for(int i=3;i<30;++i)
         UnitTest(check_set, std::vector<int>{1+i,i,5,i+6},i,i+1,i+2,i);
 
 for(int i=3;i<30;++i)
         UnitTest(check_set, i,i+1,i+2,i,i, std::vector<int>{3,4});
 
*/




END_OVERALL();




  // ctx.setBlob(key);

    return 0;
}