#include <tr1/unordered_map>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <lru.h>
#include <string.h>
#include <lm.h>
#include <wid.h>
#include <logs3.h>
#include <TrainerNgramSlist.h>
#include <WordList.h>

using namespace std::tr1;



//tempo to accelerate, but not mandatory (sometimes it's bigger but not faster)
typedef struct { unsigned int *w ; int older ;int newer ; int  val; int tempo;} Gram;
static float **resuCSLM=NULL;
static Gram * ram=NULL;
static int newer=0,older=1;
static int cardinal=1;
static int ordreMax=2;
static size_t ordreHash=2;
static int  TAILLEMAX=6;
static  lm_t *lm =NULL;
typedef Gram* Pgram;
static TrainerNgramSlist ** cslm=NULL;
static WordList * cslm_wlist=NULL;
static int nbTrainer=0;
static  double *poids=NULL; 
#define mix64(a,b,c) \
{ \
  a -= b; a -= c; a ^= (c>>43); \
  b -= c; b -= a; b ^= (a<<9); \
  c -= a; c -= b; c ^= (b>>8); \
  a -= b; a -= c; a ^= (c>>38); \
  b -= c; b -= a; b ^= (a<<23); \
  c -= a; c -= b; c ^= (b>>5); \
  a -= b; a -= c; a ^= (c>>35); \
  b -= c; b -= a; b ^= (a<<49); \
  c -= a; c -= b; c ^= (b>>11); \
  a -= b; a -= c; a ^= (c>>12); \
  b -= c; b -= a; b ^= (a<<18); \
  c -= a; c -= b; c ^= (b>>22); \
}

static size_t  hash2( size_t *k)
       /* the key */
   /* the length of the key */
    /* the previous hash, or an arbitrary value */
{
  size_t  a,b,c,len,length;

  /* Set up the internal state */
  length=len = ordreHash;
  a = b = 99000221LL;                         /* the previous hash value */
  c = 0x9e3779b97f4a7c13LL; /* the golden ratio; an arbitrary value */

  /*---------------------------------------- handle most of the key */
  while (len >= 3)
  {
    a += k[0];
    b += k[1];
    c += k[2];
    mix64(a,b,c);
    k += 3; len -= 3;
  }

  /*-------------------------------------- handle the last 2 ub8's */
  c += (length<<3);
  switch(len)              /* all the case statements fall through */
  {
    /* c is reserved for the length */
  case  2: b+=k[1];
  case  1: a+=k[0];
    /* case 0: nothing left to add */
  }
  mix64(a,b,c);
  /*-------------------------------------------- report the result */
  return c;
}


static int premier[]={ 37, 229, 409,547,601,907};
  
struct hash_X {
  size_t operator()(const  Pgram  &x) const {
    return hash2((size_t*) (x->w));
    size_t res = x->w[0];
    for(int i = 1; i < ordreMax; i++) {
      res ^= (size_t)(x->w[i] *premier[i]) << 3*ordreMax;
    }
    //fprintf(stderr," h ici : %d %lu\n", x->w[0],res&0xFFFFFFFF);
    return res;
  }


  bool operator ()(const  Pgram & x1, const  Pgram &x2) const {
    // fprintf(stderr," eq %d %d \n",x1->w[0],x2->w[0]);
    for(int i = 0; i < ordreMax; i++) {
      if(x1->w[i] != x2->w[i]) return false;
    }
    //fprintf(stderr," true \n");
    return true;
  }
};
  
static   unordered_map< Gram  *  , int,hash_X ,hash_X > cache;

static void dump (unordered_map< Gram  *  , int,hash_X ,hash_X > &cachou, const char *p="main") {
 unordered_map <Pgram,int,hash_X ,hash_X>::iterator j;
 return ;
std::cerr << "debut dump "<<p<<std::endl;
 
 for (j=cachou.begin();j!=cachou.end(); j++)
   std::cerr << j->first << " sec   " << j->second << " case" << ram[j->second].w[0] <<std::endl;
 std::cerr << "find dump "<<p<<std::endl;

}  
 


 static int lru_count=0;
static int hit=0;
static int  enregistrerLRUinterne(unsigned int *w,int tempo) {
  static int vcardinal=0;
  static int atempo=-1;

  if (atempo!= tempo) {
    vcardinal=cardinal;
   fprintf(stderr,"LRU cache taille :%d sentence:%d  card:%d h:%d     old:%d new%d \n",lru_count,atempo,cardinal,hit,older,newer);
   lru_count=0;
   hit=0;
   atempo=tempo;
 }
 hit++;
  ram[0].w=w;
 unordered_map <Pgram,int,hash_X ,hash_X>::iterator j =cache.find(ram);
 //  dump(cache,"in passe1"); 
  if (j!= cache.end()) { 
    
    int p=j->second;
    if (ram[p].tempo>=tempo) return -1;
    ram[p].tempo=tempo;
    if (p==newer) return -1;
    if (older!=p) {
      ram[ram[p].older].newer = ram[p].newer;
      ram[ram[p].newer].older=ram[p].older;}
    else {
      ram[ram[p].newer].older=-1;
      older=ram[p].newer;
    }

    assert (older>0);
    //assert (older< vcardinal || vcardinal <1000);
    ram[p].newer=-1;
    ram[p].older=newer;
    ram[newer].newer=p;
    newer=p;
    return  -1;
}
  lru_count++;
  if (cardinal <TAILLEMAX) {
    memcpy( ram[cardinal].w,w,ordreMax*sizeof(int));
    ram[cardinal].older=newer;
    ram[newer].newer=cardinal;
    newer=cardinal;
    ram[cardinal].tempo=tempo;
    ram[cardinal].newer=-1;
    cache[ram+cardinal]=cardinal;
    cardinal ++;
    ram[cardinal-1].val= 3*cardinal;
    return   cardinal-1;
  }

  dump(cache,"in pass1 av er"); 
  assert(ram[older].tempo<tempo);

  cache.erase(ram+older);
  dump(cache,"in passe1 apr er"); 
  memcpy( ram[older].w,w,ordreMax*sizeof(int));
  ram[older].tempo=tempo;
  ram[older].older=newer;
  ram[newer].newer=older;
  newer=older;
  cache[ram+older]=older;
  int tempOlder=older;
  
  older=ram[older].newer;
  assert(older>0);
  //assert (older< vcardinal || vcardinal <1000);
  ram[older].older=-1;
  dump(cache,"in passe1 apr ins"); 
  return  tempOlder;
}

 int * enregistrerLRU(unsigned int *w,int tempo) { 
   int res=   enregistrerLRUinterne( w,tempo);
   
     
   if (-1 ==res) return NULL;
   int tab[ordreMax+1];
   int cslmOK= cslm!=NULL;
   for (int i =0 ;i<ordreMax+1;i++) tab[i]=-1;
   if (cslmOK)
     for (int i =0 ; i<ordreMax; i++){ 
       tab[i]=( w[i]!= BAD_LMWID(lm))  ? lm->lmwid2cslm[w[i]] :  -1;
       if (tab[i]==-1) cslmOK=0;
       assert(cslmOK==0 || i==0 ||i>=3||  tab[i]>3);
 }
     cslmOK=cslmOK && lm->cslmshort[w[ordreMax-1]];
     int score =lm_ng_score(lm,ordreMax,w,0);//it's strange to set 0 instead of last word id
     assert(score <0);
     if (!cslmOK){
       ram[res].val=score;
       return &ram[res].val;
     }
ram[res].val=-score;


if (0) fprintf(stderr," (%d,%d)", w[ordreMax-1],tab[ordreMax-1]);
 for (int i =0 ; i<nbTrainer; i++){
   resuCSLM[i][res]=1.0;
   cslm[i]->BlockEval(tab,ordreMax,&(resuCSLM[i][res]));
   if (0&& resuCSLM[i][res]!=1.0) {
     fprintf (stderr, "-- %d %f:",res,exp(resuCSLM[i][res]));
     for (int iw=0 ;iw<ordreMax ; iw++) {
       fprintf(stderr,"%s:(%d,%d) " ,lm->wordstr[w[iw]], w[iw],tab[iw]);
     }
     fprintf(stderr,"\n");
   }
 }
  return &ram[res].val;
}


void  finishLRU(void){
  fprintf(stderr," finish\n");
  for (int i=0 ; i<nbTrainer ; i++)
    cslm[i]->BlockFinish();
}

int lireLRU (unsigned int *w) {
  
  ram[0].w=w;

  unordered_map <Pgram,int,hash_X ,hash_X>::iterator j;
  j =cache.find(ram);
  if   (j!= cache.end()){
    if (  ram[j->second].val<0)  return ram[j->second].val;
    int pos= j->second;
    double proba=0.0;
    int probleme=0;
    if (cslm) {
      for (int i =0 ; i<nbTrainer ; i++) {
	double prCSLM = exp(resuCSLM[i][pos]);
	if ((prCSLM<0) || (prCSLM>1.0))
	  probleme=1;
      }
    }
    if (probleme) {
      fprintf(stderr,"--------------------->%6d %5d %f %i %s\n",pos,ram[pos].w[ordreMax-1],exp(resuCSLM[0][pos]),ram[j->second].val,lm->wordstr[ram[pos].w[ordreMax-1]]);

      unsigned int *w=ram[pos].w;
      for (int i =0 ; i<ordreMax; i++){
	int j;
       j=( w[i]!= BAD_LMWID(lm))  ? lm->lmwid2cslm[w[i]] :  -1;
       fprintf (stderr,"%s:(%d,%d) ",lm->wordstr[w[i]], w[i],j);
      }
      fprintf(stderr,"\n");

      return -ram[j->second].val;
    }


      for (int i=0 ;i<nbTrainer ; i++)
      proba += poids[i]*exp(resuCSLM[i][pos]);
    proba= (1-lm->poidscslm)* logs3_to_p(LM_RAWSCORE(lm,-ram[pos].val))+lm->poidscslm*proba;
    ram[pos].val=(int32)(logs3(proba)*lm->lw)+lm->wip;
    assert(ram[pos].val <0);
    return ram[pos].val;
     }
  fprintf(stderr,"LRU cache taille :%d   card:%d h:%d\n",lru_count,cardinal,hit);
    assert(0);
  return 10000;

}






void  lmMakeIndexCSLM(dict_t *dict ,lm_t * lm,WordList * cslm_wlist) {
  s3wid_t dictid;
  WordList::const_iterator cslm_cur, cslm_end = cslm_wlist->End();
  WordList::WordIndex index;
  char * mot;
  s3lmwid32_t lmid;
  int i;
lm->lmwid2cslm= (int32 *) malloc( sizeof (int32)*lm->n_ng[0]);
  lm->cslmshort=(int*) calloc(lm->n_ng[0],sizeof(int));
  for (i =0 ; i<lm->n_ng[0] ; i++)
    lm->lmwid2cslm[i]=-1;
  for (cslm_cur = cslm_wlist->Begin() ; cslm_cur != cslm_end ; cslm_cur++) {
    index = cslm_cur->id;
    mot = cslm_cur->word;
    dictid=dict_wordid(dict,mot);
    if (! IS_S3WID(dictid)) {
      E_ERROR("Word '%s' found in CSLM but not in dictionary\n", mot);
      continue;
    }else
      if (dict_filler_word(dict, dictid)){
	E_ERROR("Filler dictionary word '%s' found in CSLM\n",mot);
	continue;}

    lmid=lm->dict2lmwid[dictid];
    if (lmid==BAD_LMWID(lm)) {
      	E_INFO("Word '%s' found in CSLM but not in LM\n",mot);
	continue;
    }
    lm->lmwid2cslm[lmid]=index;
    lm->cslmshort[lmid]=cslm_wlist->InShortList(cslm_wlist->MapIndex(index));
  }
  E_INFO("%d words in CSLM vocabulary\n",cslm_wlist->GetSize());
}

void initCacheLRU( int tailleCache, int ordre, lm_t *lmlocal , dict_t *dict ,  char const  * nomMach) {
  FILE *f=NULL;
 
  char nom1[1024], nomVocabPondere[1024];
 
 lm=lmlocal;
  if (nomMach !=NULL) {
    f=fopen(nomMach,"r");
    if (f==NULL) {
      fprintf(stderr,"can't open %s\n",nomMach);
      exit(1);
    }
    fscanf(f,"%s",nomVocabPondere);
    fscanf(f,"%i",&nbTrainer); //machine number
    cslm= new TrainerNgramSlist *[nbTrainer];
    poids= new double[nbTrainer];
    cslm_wlist = new WordList;
    if (cslm_wlist != NULL)
      cslm_wlist->Read(nomVocabPondere); /* shortlist length will be set with TrainerNgramSlist class */
    for (int i =0; i<nbTrainer; i++)
      {
	if (fscanf(f,"%s%lf",nom1,poids+i)!=2) {fprintf(stderr,"error reading machine %i\n",i);exit(1);}

	std::ifstream ifs;
	ifs.open(nom1,ios::binary);
	CHECK_FILE(ifs,nom1);
	Mach *m = Mach::Read(ifs);
	ifs.close();
	m->Info();
	cslm[i] = new  TrainerNgramSlist(m, cslm_wlist, (char*)"");
      }
    fclose(f);
    lmMakeIndexCSLM(dict,lm,cslm_wlist);
    
  }
  TAILLEMAX=tailleCache+1;
  resuCSLM=(float **) malloc(nbTrainer*sizeof(float*));
  resuCSLM[0]=(float *) malloc(nbTrainer*sizeof(float)*TAILLEMAX);
  for (int i=1 ; i<nbTrainer ;i++)
    resuCSLM[i]=resuCSLM[i-1]+TAILLEMAX;

  
  ordreMax=ordre;
  if (ordre%2 ==1) ordre++;
  ordreHash=ordre/2;
  ram=(Gram * )malloc(sizeof( Gram) * TAILLEMAX);
  ram->w=(unsigned int *) calloc(TAILLEMAX*ordre, sizeof(unsigned int) );
 
  for (int i =1 ;i<TAILLEMAX; i++) ram[i].w=ram[i-1].w+ordre;
}  
