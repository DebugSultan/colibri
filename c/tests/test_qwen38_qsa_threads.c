/* QSA decode threading: when S==1 the outer loop (positions) is switched off
 * by its `if(S>1)` clause and the block-ranking and head loops carry the
 * parallelism instead. The claim is STRONGER than "same within tolerance":
 * the per-b and per-h FP order is untouched by the pragmas (writes disjoint,
 * scratch per iteration, shared data read-only), so decode must be
 * BIT-IDENTICAL to one thread, at any team size, token for token -- and the
 * prefill path must not have moved at all.
 *
 * Strategy: run the REAL q38_attention over synthetic state shaped like the
 * 4k-token decode position (24 heads, 2 KV heads, D=256, 2048 selected),
 * twice -- once with OMP_NUM_THREADS=2 and once with =1, which degrades the
 * same code to serial through the same if-clauses -- and diff the FNV hash of
 * the output. A shared write or a cross-thread FP addition would part from the
 * serial run on at least one bit. Equal hashes alone could also mean the
 * guards never fire and everything stayed serial, so case 4 times the real
 * call inside each child and requires a clear speedup.
 *
 * The children are this same binary re-executed with the env set; nothing
 * touches CUDA or the model files -- safe on a machine whose cards are busy. */
#define _GNU_SOURCE
#define QWEN38_NO_MAIN
#include "../qwen38.c"

#include <fcntl.h>
#include <time.h>
#include <sys/wait.h>
#include <unistd.h>

#define QSA_QH 24
#define QSA_KVH 2
#define QSA_D 256
#define QSA_IQ 4
#define QSA_ID 128
#define QSA_R 4
#define QSA_BUDGET 2048
#define QSA_POS 8192          /* decode at the end of a 4k context */
#define QSA_H 2560

static Model m;
static Layer l;

static uint32_t rs=12345;
static float rf(void){rs^=rs<<13;rs^=rs>>17;rs^=rs<<5;return (int32_t)(rs>>8)/8388608.f;}

static void build_state(void){
    Cfg *c=&m.c;
    c->hidden=QSA_H;c->q_heads=QSA_QH;c->kv_heads=QSA_KVH;c->head_dim=QSA_D;
    c->idx_qheads=QSA_IQ;c->idx_kheads=1;c->idx_dim=QSA_ID;
    c->idx_budget=QSA_BUDGET;c->idx_ratio=QSA_R;
    c->eps=1e-6f;c->rotary_dim=64;c->theta=10000.f;
    m.kv_cap=QSA_POS+8;
    m.K=calloc(1,sizeof(float*));m.V=calloc(1,sizeof(float*));
    m.IK=calloc(1,sizeof(float*));
    m.K[0]=malloc(sizeof(float)*(size_t)QSA_KVH*m.kv_cap*QSA_D);
    m.V[0]=malloc(sizeof(float)*(size_t)QSA_KVH*m.kv_cap*QSA_D);
    m.IK[0]=malloc(sizeof(float)*(size_t)m.kv_cap*QSA_ID);
    for(size_t i=0;i<(size_t)QSA_KVH*m.kv_cap*QSA_D;i++){m.K[0][i]=rf();m.V[0][i]=rf();}
    for(size_t i=0;i<(size_t)m.kv_cap*QSA_ID;i++)m.IK[0][i]=rf();
    memset(&l,0,sizeof l);
    #define WL(field,nrows,ncols) do{ \
        l.field.owns_data=1;l.field.kind=Q38_WEIGHT_F32; \
        l.field.rows=(nrows);l.field.cols=(ncols); \
        float *d=malloc(sizeof(float)*(size_t)(nrows)*(ncols)); \
        for(size_t i=0;i<(size_t)(nrows)*(ncols);i++)d[i]=rf()*0.05f; \
        l.field.data=d; \
    }while(0)
    #define VL(field,n) do{ \
        float *v=malloc(sizeof(float)*(size_t)(n)); \
        for(size_t i=0;i<(size_t)(n);i++)v[i]=rf()*0.2f+0.8f; \
        l.field=v; \
    }while(0)
    WL(q,QSA_QH*2*QSA_D,QSA_H);
    WL(k,QSA_KVH*QSA_D,QSA_H);
    WL(v,QSA_KVH*QSA_D,QSA_H);
    WL(o,QSA_H,QSA_QH*QSA_D);
    WL(idx_qk,(QSA_IQ+1)*QSA_ID,QSA_H);
    VL(qn,QSA_D);VL(kn,QSA_D);
    VL(idx_qn,QSA_ID);VL(idx_kn,QSA_ID);
    #undef WL
    #undef VL
}

static int run_child(int S){
    build_state();
    float *x=malloc(sizeof(float)*(size_t)S*QSA_H);
    for(size_t i=0;i<(size_t)S*QSA_H;i++)x[i]=rf()*0.1f;
    float *out=malloc(sizeof(float)*(size_t)S*QSA_H);
    memset(out,0,sizeof(float)*(size_t)S*QSA_H);
    struct timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    q38_attention(&m,&l,0,x,S,QSA_POS-S,out);   /* the real function */
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double ms=(t1.tv_sec-t0.tv_sec)*1e3+(t1.tv_nsec-t0.tv_nsec)/1e6;
    uint64_t h=1469598103934665603ULL;
    unsigned char *b=(unsigned char*)out;
    for(size_t i=0;i<sizeof(float)*(size_t)S*QSA_H;i++){h^=b[i];h*=1099511628211ULL;}
    printf("%016llx %.2f\n",(unsigned long long)h,ms);
    fflush(stdout);
    return 0;
}

/* one child: this binary, S rows, T OMP threads, stdout captured to file */
static int spawn(int threads,int S,char *line,size_t cap){
    pid_t p=fork();
    if(p<0)return -1;
    if(p==0){
        char tb[8],sb[8];
        snprintf(tb,sizeof tb,"%d",threads);
        snprintf(sb,sizeof sb,"%d",S);
        setenv("OMP_NUM_THREADS",tb,1);
        int fd=open("/tmp/qsa_threads_hash.txt",O_WRONLY|O_CREAT|O_TRUNC,0644);
        if(fd<0)_exit(3);
        dup2(fd,1);
        char path[512];
        ssize_t n=readlink("/proc/self/exe",path,sizeof path-1);
        if(n<0)_exit(2);
        path[n]=0;
        char *av[4]={"self","run",sb,NULL};
        execv(path,av);
        _exit(4);
    }
    int st=0;
    if(waitpid(p,&st,0)<0||!WIFEXITED(st)||WEXITSTATUS(st))return -1;
    FILE *f=fopen("/tmp/qsa_threads_hash.txt","r");
    if(!f)return -1;
    int ok=fgets(line,(int)cap,f)!=NULL;
    fclose(f);
    return ok?0:-1;
}

static const char *hash_of(char *line){static char h[17];sscanf(line,"%16s",h);return h;}

int main(int argc,char **argv){
    if(argc>2&&!strcmp(argv[1],"run"))return run_child(atoi(argv[2]));

    char a[64],b[64];
    int rc=0;

    if(spawn(1,1,a,sizeof a)||spawn(2,1,b,sizeof b)){fprintf(stderr,"child failed\n");return 1;}
    printf("%-6s decode S=1, 1 vs 2 threads: %s vs %s\n",
           strcmp(hash_of(a),hash_of(b))?"FAIL":"ok",a,b);
    if(strcmp(hash_of(a),hash_of(b)))rc=1;

    if(spawn(16,1,b,sizeof b)){fprintf(stderr,"child failed\n");return 1;}
    printf("%-6s decode S=1, 1 vs 16 threads: %s vs %s\n",
           strcmp(hash_of(a),hash_of(b))?"FAIL":"ok",a,b);
    if(strcmp(hash_of(a),hash_of(b)))rc=1;

    if(spawn(1,8,a,sizeof a)||spawn(16,8,b,sizeof b)){fprintf(stderr,"child failed\n");return 1;}
    printf("%-6s prefill S=8, 1 vs 16 threads: %s vs %s\n",
           strcmp(hash_of(a),hash_of(b))?"FAIL":"ok",a,b);
    if(strcmp(hash_of(a),hash_of(b)))rc=1;

    /* The guards must actually fire: decode at 16 threads has to be clearly
     * faster than at 1, or the "threading" is decoration. Several reps, best
     * of each, because the child pays one cold-state pass. */
    double best1=1e9,best16=1e9;
    for(int rep=0;rep<3;rep++){
        double t;
        if(spawn(1,1,a,sizeof a))return 1;
        sscanf(a,"%*s %lf",&t); if(t<best1)best1=t;
        if(spawn(16,1,b,sizeof b))return 1;
        sscanf(b,"%*s %lf",&t); if(t<best16)best16=t;
    }
    printf("%-6s decode attention call: serial %.2f ms vs x16 %.2f ms (want >2x)\n",
           best1>2*best16?"ok":"FAIL",best1,best16);
    if(!(best1>2*best16))rc=1;

    printf("%s\n",rc?"FAILED":"PASSED");
    return rc;
}
