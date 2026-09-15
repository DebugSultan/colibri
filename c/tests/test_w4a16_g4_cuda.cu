/* W4A16 Tensor Core over grouped-int4 scales (fmt=4) — routing + parity oracle.
 *
 * The TC_W4A16 branch of coli_cuda_expert_group was per-row scales only
 * (fmt=2); the VRAM expert tiers upload int4-gs64 experts (fmt=4), so the
 * branch never engaged for them and prefill rode the g4 dual kernels at
 * ~40 us/row. This test pins the extension: fmt=4 members (gs=64, tail
 * groups included) and a fmt=2 rider in the SAME group must go through the
 * W4A16 kernels and match the CPU reference within fp16 noise.
 *
 * Three claims, each independently falsifiable:
 *   1. ROUTING — the branch's row counter advances by exactly the rows the
 *      group served while COLI_CUDA_TC_W4A16=1, and does not move with the
 *      env off (the g4 dual takes over). Without this, a silently-skipped
 *      branch would make every parity number fiction.
 *   2. PARITY — TC path vs the double CPU reference (cpu_gemv_g4, same
 *      oracle as test_grouped_g4) within fp16 rounding of the folded scales.
 *   3. CROSS-PATH — TC and g4-dual outputs on identical data agree with each
 *      other, and the async decode path (issue/take, no TC branch) stays
 *      bitwise equal to the sync g4-dual result.
 *
 * Build: nvcc -O2 -std=c++17 -arch=native tests/test_w4a16_g4_cuda.cu -o tests/test_w4a16_g4
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#include "../backend_cuda.cu"

static void cpu_gemv_g4(const uint8_t *q,const float *sc,int K,int O,int gs,
                        const float *x,float *y){
    int rb=(K+1)/2, ng=gs>0?(K+gs-1)/gs:1, egs=gs>0?gs:K;
    for(int o=0;o<O;o++){
        const uint8_t *row=q+(size_t)o*rb; const float *scl=sc+(size_t)o*ng;
        double a=0;
        for(int g=0; g*egs<K; g++){
            int base=g*egs, glen=egs; if(base+glen>K) glen=K-base;
            double p=0;
            for(int i=base;i<base+glen;i++){
                uint8_t v=row[i>>1]; int n=(i&1)?(v>>4):(v&15);
                p+=(double)x[i]*(n-8);
            }
            a+=p*scl[g];
        }
        y[o]=(float)a;
    }
}
static int close_enough(float a,float b){
    /* fp16 A/B rounding in the TC path: per-term rel err ~1e-3 over K terms
     * walks to ~1.5e-3 absolute on unit-scale data (measured, worst 1.7e-3).
     * A wrong group/scale index would move an output by O(0.1..1) — two
     * orders above this bound. */
    return fabsf(a-b)<=3e-3f*fabsf(b)+5e-3f;
}

int main(void){
    srand(11);
    const int D=200, I=96, gs=64;            /* tail groups: 200%64=8, 96%64=32 */
    const int COUNT=3;                       /* experts 0,1: fmt4 gs=64; expert 2: fmt2 */
    const int rbD=(D+1)/2, rbI=(I+1)/2;
    const int ngD=(D+gs-1)/gs, ngI=(I+gs-1)/gs;
    int bad=0;
    if(!coli_cuda_init((int[]){0},1)){ printf("FAIL cuda init\n"); return 1; }

    ColiCudaTensor *tg[COUNT]={},*tu[COUNT]={},*td[COUNT]={};
    uint8_t *hg[COUNT],*hu[COUNT],*hd[COUNT]; float *hgs[COUNT],*hus[COUNT],*hds[COUNT];
    for(int c=0;c<COUNT;c++){
        int cgs = c==2 ? 0 : gs;             /* expert 2 rides along as fmt=2 */
        int cngD = cgs? ngD:1, cngI = cgs? ngI:1, fmt = cgs?4:2;
        hg[c]=(uint8_t*)malloc((size_t)I*rbD); hu[c]=(uint8_t*)malloc((size_t)I*rbD);
        hd[c]=(uint8_t*)malloc((size_t)D*rbI);
        hgs[c]=(float*)malloc((size_t)I*cngD*4); hus[c]=(float*)malloc((size_t)I*cngD*4);
        hds[c]=(float*)malloc((size_t)D*cngI*4);
        for(size_t i=0;i<(size_t)I*rbD;i++){ hg[c][i]=rand()&255; hu[c][i]=rand()&255; }
        for(size_t i=0;i<(size_t)D*rbI;i++) hd[c][i]=rand()&255;
        for(size_t i=0;i<(size_t)I*cngD;i++){ hgs[c][i]=.01f+.05f*(rand()/(float)RAND_MAX);
                                              hus[c][i]=.01f+.05f*(rand()/(float)RAND_MAX); }
        for(size_t i=0;i<(size_t)D*cngI;i++) hds[c][i]=.01f+.05f*(rand()/(float)RAND_MAX);
        if(!coli_cuda_tensor_upload_g(&tg[c],hg[c],hgs[c],fmt,D,I,0,cgs)||
           !coli_cuda_tensor_upload_g(&tu[c],hu[c],hus[c],fmt,D,I,0,cgs)||
           !coli_cuda_tensor_upload_g(&td[c],hd[c],hds[c],fmt,I,D,0,cgs)){
            printf("FAIL upload_g\n"); return 1; }
    }

    const int rows[COUNT]={17,33,16};        /* all >= one TC tile; 17/33 leave partial tiles */
    const int total=rows[0]+rows[1]+rows[2]; /* 66 */
    float *x=(float*)malloc((size_t)total*D*4);
    float *ytc=(float*)malloc((size_t)total*D*4);
    float *yg4=(float*)malloc((size_t)total*D*4);
    for(size_t i=0;i<(size_t)total*D;i++) x[i]=(rand()/(float)RAND_MAX-.5f)*2.f;

    /* ---- claim 1+2: env on -> TC branch takes all rows, output matches CPU ---- */
    setenv("COLI_CUDA_TC_W4A16","1",1);
    uint64_t tc_before=0,tc_after=0;
    coli_cuda_tc_w4a16_rows(&tc_before);
    if(!coli_cuda_expert_group(tg,tu,td,rows,COUNT,ytc,x)){ printf("FAIL sync group (tc)\n"); return 1; }
    coli_cuda_tc_w4a16_rows(&tc_after);
    if(tc_after-tc_before!=(uint64_t)total){
        printf("FAIL routing: TC rows +%llu, expected %d (branch did not engage)\n",
               (unsigned long long)(tc_after-tc_before),total); bad++; }
    /* full check, every (expert,row) pair against its own x row */
    {
        int off=0;
        for(int c=0;c<COUNT;c++){
            for(int s=0;s<rows[c];s++){
                float rg[512],ru[512],rh[512],ry[512];
                const float *xr=x+(size_t)(off+s)*D;
                cpu_gemv_g4(hg[c],hgs[c],D,I,c==2?0:gs,xr,rg);
                cpu_gemv_g4(hu[c],hus[c],D,I,c==2?0:gs,xr,ru);
                for(int o=0;o<I;o++) rh[o]=(rg[o]/(1.f+expf(-rg[o])))*ru[o];
                cpu_gemv_g4(hd[c],hds[c],I,D,c==2?0:gs,rh,ry);
                for(int o=0;o<D;o++){
                    float got=ytc[(size_t)(off+s)*D+o];
                    if(!close_enough(got,ry[o])){
                        if(bad<5) printf("  tc mismatch c=%d s=%d o=%d: got %g want %g\n",
                                         c,s,o,got,ry[o]);
                        bad++;
                    }
                }
            }
            off+=rows[c];
        }
    }

    /* ---- claim 1 (negative) + 3: env off -> g4 dual, counter must not move ---- */
    setenv("COLI_CUDA_TC_W4A16","0",1);
    uint64_t tc_mid=0;
    coli_cuda_tc_w4a16_rows(&tc_mid);
    if(!coli_cuda_expert_group(tg,tu,td,rows,COUNT,yg4,x)){ printf("FAIL sync group (g4)\n"); return 1; }
    coli_cuda_tc_w4a16_rows(&tc_after);
    if(tc_after!=tc_mid){ printf("FAIL routing: counter moved with the env off\n"); bad++; }
    for(size_t i=0;i<(size_t)total*D;i++)
        if(!close_enough(ytc[i],yg4[i])){
            if(bad<8) printf("  tc-vs-g4 mismatch at %zu: tc %g g4 %g\n",i,ytc[i],yg4[i]);
            bad++;
        }

    /* ---- claim 3b: async decode path (issue/take) unchanged, bitwise == sync ---- */
    {
        const int drows[COUNT]={1,2,1}, dtotal=4;
        float *dx=(float*)malloc((size_t)dtotal*D*4);
        float *dys=(float*)malloc((size_t)dtotal*D*4);
        for(size_t i=0;i<(size_t)dtotal*D;i++) dx[i]=(rand()/(float)RAND_MAX-.5f)*2.f;
        if(!coli_cuda_expert_group(tg,tu,td,drows,COUNT,dys,dx)){ printf("FAIL decode sync\n"); return 1; }
        if(!coli_cuda_expert_group_issue(tg,tu,td,drows,COUNT,dx)){ printf("FAIL decode issue\n"); return 1; }
        const float *dya=coli_cuda_expert_group_take(0);
        if(!dya){ printf("FAIL decode take\n"); return 1; }
        for(size_t i=0;i<(size_t)dtotal*D;i++) if(dya[i]!=dys[i]) bad++;
        free(dx); free(dys);
    }

    printf("w4a16-g4 oracle: %d experts (rows 17/33/16, fmt4 gs=64 + fmt2 rider), %d mismatches\n",
           COUNT,bad);
    if(bad){ printf("FAIL\n"); return 1; }
    printf("ok\n");
    return 0;
}
