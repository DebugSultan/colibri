/* test_qwen38_i4_matmul.c - discriminating checks for q38_matmul_i4_avx2, the
 * pinned AVX2 kernel of the int4 gs64 container.
 *
 * The properties worth pinning down:
 *  1. bit-for-bit identity between a row computed on its own and the same row
 *     inside a batch - including batches that STRADDLE Q38_MV_ROWS (64):
 *     S=100 and S=130 cross the boundary, and an off-by-r0 defect in the row
 *     bank is unreachable at S<=32 by construction;
 *  2. the vector path is closer to the truth (float64) than the scalar
 *     matmul_i4_grouped, measured as a BACKWARD error (sum |w.x| in the
 *     denominator, not |result|: with random activations a 2560-term dot
 *     product cancels, and dividing by the result would measure that
 *     cancellation rather than the kernel);
 *  3. agreement with the reference path (rel << 1);
 *  4. a NaN scale poisons its own output column and no other;
 *  5. counterproofs: a two's-complement nibble convention and transposed
 *     scales must both be CAUGHT by the agreement gate.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../qwen38_matmul.h"

static uint64_t rng=0x9e3779b97f4a7c15ull;
static double frand(void){
    rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;
    return (double)(rng>>11)/9007199254740992.0;
}

/* quantize per int4 group, binary offset: q=clamp(round(w/s)+8,0,15),
 * low nibble = even index. */
static void quantize_i4(uint8_t *q4,float *scale,const float *w,int I,int O,
                        int gs){
    int ng=(I+gs-1)/gs;
    memset(q4,0,(size_t)O*((I+1)/2));
    for(int o=0;o<O;o++){
        const float *row=w+(int64_t)o*I;
        uint8_t *q=q4+(int64_t)o*((I+1)/2);
        float *scl=scale+(int64_t)o*ng;
        for(int g=0;g*gs<I;g++){
            int base=g*gs,glen=gs;if(base+glen>I)glen=I-base;
            double amax=0;
            for(int k=0;k<glen;k++){double a=fabs(row[base+k]);if(a>amax)amax=a;}
            float s=amax>0?(float)(amax/7.0):1e-8f;
            scl[g]=s;
            for(int k=0;k<glen;k++){
                int v=(int)lround((double)row[base+k]/s)+8;
                if(v<0)v=0;if(v>15)v=15;
                int idx=base+k;
                if(idx&1)q[idx>>1]|=(uint8_t)(v<<4);
                else q[idx>>1]=(uint8_t)v;
            }
        }
    }
}

/* float64 reference: binary-offset decode, per-group scale */
static double ref_dot_f64(const uint8_t *q4,const float *scale,const float *x,
                          int I,int gs){
    double a=0;
    for(int g=0;g*gs<I;g++){
        int base=g*gs,glen=gs;if(base+glen>I)glen=I-base;
        double d=0;
        for(int k=0;k<glen;k++){
            uint8_t byte=q4[(base+k)>>1];
            int q=(base+k)&1?byte>>4:byte&0xF;
            d+=(double)x[base+k]*(double)(q-8);
        }
        a+=d*(double)scale[g];
    }
    return a;
}

/* deliberately wrong variant: two's-complement decode (XOR 0x8) */
static double ref_dot_f64_twos(const uint8_t *q4,const float *scale,
                               const float *x,int I,int gs){
    double a=0;
    for(int g=0;g*gs<I;g++){
        int base=g*gs,glen=gs;if(base+glen>I)glen=I-base;
        double d=0;
        for(int k=0;k<glen;k++){
            uint8_t byte=q4[(base+k)>>1];
            int q=(base+k)&1?byte>>4:byte&0xF;
            d+=(double)x[base+k]*(double)((q^8)-8);
        }
        a+=d*(double)scale[g];
    }
    return a;
}

/* backward error: max |y - yref| / sum |w.x| in f64 */
static double backward_error(const uint8_t *q4,const float *scale,
                             const float *x,const float *y,int S,int I,int O,
                             int gs){
    double worst=0;
    for(int o=0;o<O;o++){
        const uint8_t *w=q4+(int64_t)o*((I+1)/2);
        const float *scl=scale+(int64_t)o*((I+gs-1)/gs);
        for(int r=0;r<S;r++){
            double denom=0;
            for(int g=0;g*gs<I;g++){
                int base=g*gs,glen=gs;if(base+glen>I)glen=I-base;
                for(int k=0;k<glen;k++){
                    uint8_t byte=w[(base+k)>>1];
                    int q=(base+k)&1?byte>>4:byte&0xF;
                    denom+=fabs((double)(q-8)*scl[g]*x[(int64_t)r*I+base+k]);
                }
            }
            double num=fabs((double)y[(int64_t)r*O+o]-
                            ref_dot_f64(w,scl,x+(int64_t)r*I,I,gs));
            double e=denom>0?num/denom:num;
            if(e>worst)worst=e;
        }
    }
    return worst;
}

static int fails=0;
static void check(int cond,const char *what){
    printf("%s %s\n",cond?"ok":"FAIL",what);
    if(!cond)fails++;
}

static void run_shape(int I,int O,int gs,const char *name){
    int S=130;                       /* crosses Q38_MV_ROWS twice */
    uint8_t *q4=malloc((size_t)O*((I+1)/2)+64);
    float *scale=malloc(sizeof(float)*(size_t)O*((I+gs-1)/gs)+64);
    float *x=malloc(sizeof(float)*(size_t)S*I+64);
    float *yv=malloc(sizeof(float)*(size_t)S*O+64);
    float *yr=malloc(sizeof(float)*(size_t)S*O+64);
    float *y1=malloc(sizeof(float)*O+64);
    if(!q4||!scale||!x||!yv||!yr||!y1){fprintf(stderr,"OOM\n");exit(1);}
    for(int64_t i=0;i<(int64_t)S*I;i++)x[i]=(float)((frand()*2-1)*1.5);
    float *w=malloc(sizeof(float)*(size_t)O*I);
    if(!w){fprintf(stderr,"OOM\n");exit(1);}
    for(int64_t i=0;i<(int64_t)O*I;i++)w[i]=(float)((frand()*2-1)*3.0);
    quantize_i4(q4,scale,w,I,O,gs);
    free(w);

    q38_matmul_i4_avx2(yv,x,q4,scale,gs,S,I,O);
    matmul_i4_grouped(yr,x,q4,scale,S,I,O,gs);

    printf("== %s I=%d O=%d gs=%d\n",name,I,O,gs);

    /* 1. bit-for-bit identity under split: every row of yv (batch 130) must
     *    match the same row computed alone and inside a batch of 100. */
    int ident=1;
    for(int r=0;r<S&&ident;r++){
        q38_matmul_i4_avx2(y1,x+(int64_t)r*I,q4,scale,gs,1,I,O);
        if(memcmp(y1,yv+(int64_t)r*O,(size_t)O*sizeof(float)))ident=0;
    }
    check(ident,"row-alone bit-identical to row-in-batch-130");
    int ident100=1;
    {
        float *y100=malloc(sizeof(float)*(size_t)100*O);
        if(!y100){fprintf(stderr,"OOM\n");exit(1);}
        q38_matmul_i4_avx2(y100,x,q4,scale,gs,100,I,O);
        for(int r=0;r<100&&ident100;r++)
            if(memcmp(y100+(int64_t)r*O,yv+(int64_t)r*O,
                      (size_t)O*sizeof(float)))ident100=0;
        free(y100);
    }
    check(ident100,"batch-100 prefix bit-identical to batch-130 prefix");

    /* 2. backward accuracy: the vector path beats the scalar one */
    double ev=backward_error(q4,scale,x,yv,S<8?S:8,I,O,gs);
    double er=backward_error(q4,scale,x,yr,S<8?S:8,I,O,gs);
    printf("   backward err: vec %.3e  scalar %.3e\n",ev,er);
    check(ev<er,"vector is closer to f64 than the scalar reference");

    /* 3. agreement with the reference (rel over rms) */
    double num=0,rms=0;
    for(int64_t i=0;i<(int64_t)S*O;i++){
        double d=yv[i]-yr[i];num+=d*d;rms+=yr[i]*yr[i];
    }
    double rel=sqrt(num/rms);
    printf("   vs scalar reference: rel %.3e\n",rel);
    check(rel<1e-6,"agrees with matmul_i4_grouped");

    /* 4. a NaN scale poisons only its own column */
    {
        int ng=(I+gs-1)/gs;
        scale[ng+0]=NAN;             /* column o=1, group 0 */
        q38_matmul_i4_avx2(yv,x,q4,scale,gs,3,I,O);
        scale[ng+0]=1.0f;
        int poison=1,clean=1;
        for(int r=0;r<3;r++){
            if(!isnan(yv[(int64_t)r*O+1]))poison=0;
            if(isnan(yv[(int64_t)r*O+0]))clean=0;
        }
        check(poison,"NaN scale poisons its own output column");
        check(clean,"neighbouring columns stay clean");
        q38_matmul_i4_avx2(yv,x,q4,scale,gs,S,I,O);   /* regenerate a clean yv */
    }

    /* 5. counterproofs: two wrong conventions must be caught */
    {
        double num2=0;
        for(int r=0;r<3;r++)
            for(int o=0;o<O;o++)
                num2+=fabs((double)yv[(int64_t)r*O+o]-
                           ref_dot_f64_twos(q4+(int64_t)o*((I+1)/2),
                                            scale+(int64_t)o*((I+gs-1)/gs),
                                            x+(int64_t)r*I,I,gs));
        double denom=0;
        for(int64_t i=0;i<(int64_t)3*O;i++)denom+=fabs(yv[i]);
        if(denom<=0)denom=1;
        double rel2=num2/denom;
        printf("   counterproof two's complement: rel %.3e\n",rel2);
        check(rel2>1e-3,"wrong nibble convention is actually caught");

        /* transposed scales: s[g*O+o] instead of s[o*ng+g], same bytes */
        float *tscale=malloc(sizeof(float)*(size_t)O*((I+gs-1)/gs));
        int ng=(I+gs-1)/gs;
        if(!tscale){fprintf(stderr,"OOM\n");exit(1);}
        for(int o=0;o<O;o++)for(int g=0;g<ng;g++)tscale[(size_t)g*O+o]=scale[(size_t)o*ng+g];
        float *yt=malloc(sizeof(float)*(size_t)3*O);
        if(!yt){fprintf(stderr,"OOM\n");exit(1);}
        q38_matmul_i4_avx2(yt,x,q4,tscale,gs,3,I,O);
        double num3=0;
        for(int64_t i=0;i<(int64_t)3*O;i++){double d=yt[i]-yv[i];num3+=d*d;}
        double rel3=sqrt(num3/rms);
        printf("   counterproof transposed scales: rel %.3e\n",rel3);
        check(rel3>1e-3,"transposed scale map is actually caught");
        free(tscale);free(yt);
    }

    free(q4);free(scale);free(x);free(yv);free(yr);free(y1);
}

int main(void){
    run_shape(2560,640,64,"gate/up");
    run_shape(640,2560,64,"down");
    run_shape(100,37,64,"tail-I");
    run_shape(130,64,64,"tail-I-2");
    printf("%s\n",fails?"FAILED":"PASS");
    return fails?1:0;
}
