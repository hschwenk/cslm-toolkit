
#include <lm.h>
#include <dict.h>
#ifdef __cplusplus
    extern "C" {
#endif
      void  finishLRU(void);
      int * enregistrerLRU(unsigned int *w,int tempo);
      int lireLRU (unsigned int *w) ;
      void initCacheLRU( int tailleCache, int ordre, lm_t *lm,dict_t *dict , char const * nomMach);
#ifdef __cplusplus
    }
#endif
