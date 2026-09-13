/* qwen38_tier.c -- CUDA VRAM expert tier for the qwen38 engine. See header.
 *
 * Parente stretto di qwen36_tier.c, ma con il vincolo invertito che l'header
 * spiega: qui il tier non puo' leggere il disco e non conserva mai puntatori
 * alla RAM, quindi promuove solo cio' che il motore gli OFFRE. Da questo
 * discende la differenza di struttura piu' visibile rispetto all'altro file:
 * non c'e' nessun tick LFRU periodico che va a caccia di candidati (li'
 * poteva, perche' i byte erano sempre raggiungibili). La decisione di
 * promozione si prende dentro q38t_offer, cioe' nell'unico istante in cui i
 * byte esistono davvero -- ed e' anche l'istante giusto, perche' un'offerta
 * nasce da un miss appena servito: l'esperto ha appena dimostrato di servire. */
#ifdef COLI_CUDA
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "qwen38_tier.h"
#include "backend_cuda.h"
#include "tier.h"
#include "quant.h"            /* E4M3_LUT */

#define Q38T_MAX_DEV  8
#define Q38T_QCAP     16      /* staging ~4,7 MB/voce -> ~75 MB di tetto */
#define Q38T_MAX_ROWS 8       /* backend_cuda.cu:2091, "decode-scale only" */

typedef struct {
    ColiCudaTensor *tg, *tu, *td;
    uint32_t heat;
    uint8_t resident, queued, planned;
} Q38TSlot;

static struct {
    int on, nl, ne, D, Ih, topk, ndev;
    size_t sc;                            /* float di scale per matrice */
    size_t mat_bytes;                     /* byte e4m3 per matrice */
    size_t exp_bytes;                     /* stima VRAM per esperto */
    int dev[Q38T_MAX_DEV];
    size_t budget[Q38T_MAX_DEV], used[Q38T_MAX_DEV];
    Q38TSlot *slot;                       /* [nl*ne] */
    pthread_mutex_t mx;
    pthread_t th;
    int th_stop;
    /* coda di upload con copie di staging */
    struct { int layer, eid; uint8_t *w; float *s; int v_layer, v_eid; } q[Q38T_QCAP];
    int qh, qt_, qn;
    pthread_cond_t cv;
    pthread_cond_t cv_take;               /* spazio in coda + qt_take fatto */
    /* statistiche */
    uint64_t hits[Q38T_MAX_DEV], miss, uploads, upload_fail;
    uint64_t offers, promotions, swaps, q_full_skips, overflow_rows, take_fails;
    uint64_t tick;
    /* stato di issue del (singolo) thread di decode */
    int is_cnt[Q38T_MAX_DEV];
    int is_k[Q38T_MAX_DEV][Q38T_MAX_ROWS];
    float *is_x; size_t is_x_floats;
    int issue_open;                       /* nessun free mentre un gruppo vola */
    /* warmstart */
    int *fill_order; int fill_n, fill_cur;
    uint32_t *heat0;
} G;

static Q38TSlot *qs(int layer, int eid){ return &G.slot[(size_t)layer*G.ne + eid]; }
static int home(int eid){ return eid % G.ndev; }

/* Staging. In qwen36 questa funzione doveva riportare i nibble int4 da
 * complemento a due a binario sfalsato; qui il formato in RAM E' gia' quello
 * del backend (e4m3 grezzo, scale per blocco 128x128), quindi sono tre copie e
 * nient'altro. Le tre matrici NON si assumono contigue: con COLI_MAP_EXPERTS=1
 * lo slot punta a tre mappature distinte del file. */
static void stage(uint8_t *dw, float *dsc,
                  const uint8_t *gate, const uint8_t *up, const uint8_t *down,
                  const float *scales){
    memcpy(dw,                  gate, G.mat_bytes);
    memcpy(dw+G.mat_bytes,      up,   G.mat_bytes);
    memcpy(dw+2*G.mat_bytes,    down, G.mat_bytes);
    memcpy(dsc, scales, 3*G.sc*sizeof(float));
}

static void *uploader(void *arg){
    (void)arg;
    for(;;){
        pthread_mutex_lock(&G.mx);
        while(G.qn==0 && !G.th_stop) pthread_cond_wait(&G.cv,&G.mx);
        if(G.th_stop && G.qn==0){ pthread_mutex_unlock(&G.mx); return NULL; }
        int layer=G.q[G.qh].layer, eid=G.q[G.qh].eid;
        int vl=G.q[G.qh].v_layer, ve=G.q[G.qh].v_eid;
        uint8_t *w=G.q[G.qh].w; float *sc=G.q[G.qh].s;
        G.qh=(G.qh+1)%Q38T_QCAP; G.qn--;
        pthread_cond_broadcast(&G.cv_take);          /* spazio in coda */
        if(ve>=0){
            /* swap: la vittima si libera solo quando nessun gruppo e' in volo */
            while(G.issue_open && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
            Q38TSlot *v=qs(vl,ve);
            if(G.th_stop && G.issue_open){
                /* Chiusura con un gruppo ancora aperto: q38t_take, l'unica cosa
                 * che azzera issue_open, non arrivera' mai. Si abbandona lo
                 * swap invece di liberare un tensore che il gruppo in volo puo'
                 * ancora leggere; il flag resident era gia' stato spento da chi
                 * ha accodato, quindi va rimesso com'era. */
                v->resident=1; qs(layer,eid)->queued=0;
                pthread_mutex_unlock(&G.mx); free(w); free(sc); continue;
            }
            ColiCudaTensor *a=v->tg,*b=v->tu,*c=v->td;
            v->tg=v->tu=v->td=NULL;
            pthread_mutex_unlock(&G.mx);
            if(a)coli_cuda_tensor_free(a);
            if(b)coli_cuda_tensor_free(b);
            if(c)coli_cuda_tensor_free(c);
        } else pthread_mutex_unlock(&G.mx);

        int dv=G.dev[home(eid)];
        /* fmt=8: gate/up sono [inter,hidden], down e' [hidden,inter]; la firma
         * vuole (I=ingresso, O=uscita), non (righe, colonne). */
        ColiCudaTensor *tg=NULL,*tu=NULL,*td=NULL;
        int ok = coli_cuda_tensor_upload(&tg, w,                 sc,          8, G.D,  G.Ih, dv)
              && coli_cuda_tensor_upload(&tu, w+G.mat_bytes,     sc+G.sc,     8, G.D,  G.Ih, dv)
              && coli_cuda_tensor_upload(&td, w+2*G.mat_bytes,   sc+2*G.sc,   8, G.Ih, G.D,  dv);
        free(w); free(sc);
        pthread_mutex_lock(&G.mx);
        Q38TSlot *s=qs(layer,eid);
        if(ok){ s->tg=tg; s->tu=tu; s->td=td; s->resident=1; G.uploads++; }
        else  { int hd=home(eid);
                G.upload_fail++;
                if(G.used[hd]>=G.exp_bytes) G.used[hd]-=G.exp_bytes;
                G.budget[hd]=G.used[hd];   /* scheda davvero piena: smettere */
                if(tg)coli_cuda_tensor_free(tg);
                if(tu)coli_cuda_tensor_free(tu);
                if(td)coli_cuda_tensor_free(td); }
        s->queued=0; s->planned=0;
        pthread_mutex_unlock(&G.mx);
    }
}

/* --- init ---------------------------------------------------------------- */

int q38t_init(int nl, int ne, int D, int Ih, int topk, int scale_count,
              int native_fp8){
    const char *e=getenv("COLI_CUDA");
    if(!(e && *e=='1')) return 0;
    if(G.on){
        /* Singleton di processo: gli slot sono dimensionati su UNA geometria.
         * Un secondo modello (adapter Segment) non puo' condividerli, quindi
         * si sente dire di no e resta su CPU. Per questo il motore ricorda il
         * ritorno di questa funzione invece di fidarsi di q38t_ready(). */
        fprintf(stderr,"[q38tier] already attached to another model in this "
                       "process -> CPU path for this one\n");
        return 0;
    }
    if(!native_fp8){
        fprintf(stderr,"[q38tier] native FP8 disabled: the RAM slots hold expanded "
                       "F32, which no fmt=8 kernel reads -> CPU path\n");
        return 0;
    }
    if(topk>Q38T_MAX_ROWS*Q38T_MAX_DEV){
        fprintf(stderr,"[q38tier] topk=%d unsupported\n",topk); return 0;
    }
    if(nl<1||ne<1||D<1||Ih<1||scale_count<1) return 0;
    memset(&G,0,sizeof G);
    G.nl=nl; G.ne=ne; G.D=D; G.Ih=Ih; G.topk=topk; G.sc=(size_t)scale_count;
    G.mat_bytes=(size_t)D*(size_t)Ih;

    const char *gl=getenv("COLI_GPUS");
    if(gl && *gl){
        char buf[128]; snprintf(buf,sizeof buf,"%s",gl);
        for(char *t=strtok(buf,","); t && G.ndev<Q38T_MAX_DEV; t=strtok(NULL,","))
            G.dev[G.ndev++]=atoi(t);
    } else {
        int available=coli_cuda_available_device_count();
        int want=available<2?available:2;
        for(int i=0;i<want && i<Q38T_MAX_DEV;i++) G.dev[G.ndev++]=i;
        fprintf(stderr,"[q38tier] COLI_GPUS unset: selecting %d visible device(s)\n",G.ndev);
    }
    if(G.ndev<1){ fprintf(stderr,"[q38tier] no visible CUDA devices -> CPU path\n"); return 0; }
    if(!coli_cuda_init(G.dev,G.ndev)){ fprintf(stderr,"[q38tier] coli_cuda_init failed -> CPU path\n"); return 0; }
    int have=coli_cuda_device_count();
    if(have<G.ndev) G.ndev=have;
    if(G.ndev<1){ fprintf(stderr,"[q38tier] no CUDA devices -> CPU path\n"); return 0; }

    /* La LUT e4m3 va pubblicata PRIMA di qualunque upload fmt=8: senza, il
     * backend li rifiuta invece di decodificare contro una tabella di zeri. */
    if(!coli_cuda_fp8_set_lut(E4M3_LUT)){
        fprintf(stderr,"[q38tier] coli_cuda_fp8_set_lut failed -> CPU path\n");
        return 0;
    }

    G.exp_bytes = 3*G.mat_bytes + 3*G.sc*sizeof(float) + 4096; /* + slack */
    const char *bg=getenv("CUDA_EXPERT_GB");
    for(int i=0;i<G.ndev;i++){
        size_t freeb=0,totb=0; coli_cuda_mem_info(G.dev[i],&freeb,&totb);
        size_t b = (bg && strcmp(bg,"auto") && atof(bg)>0)
                   ? (size_t)(atof(bg)*1024.0*1024.0*1024.0)
                   : (freeb>(1ull<<30) ? freeb-(1ull<<30) : 0);
        G.budget[i]=b;
        fprintf(stderr,"[q38tier] dev %d: %.1f GB free, budget %.1f GB (~%zu experts)\n",
                G.dev[i], freeb/1073741824.0, b/1073741824.0, b/G.exp_bytes);
    }
    G.slot=calloc((size_t)nl*ne,sizeof(Q38TSlot));
    G.is_x_floats=(size_t)G.ndev*Q38T_MAX_ROWS*D;
    G.is_x=malloc(G.is_x_floats*sizeof(float));
    if(!G.slot||!G.is_x){ free(G.slot); free(G.is_x); return 0; }

    const char *hf=getenv("HEAT_FILE");
    if(hf){
        FILE *f=fopen(hf,"rb");
        if(f){
            uint32_t hdr[3]={0,0,0};
            if(fread(hdr,4,3,f)==3 && hdr[0]==0x51544831u &&
               hdr[1]==(uint32_t)nl && hdr[2]==(uint32_t)ne){
                G.heat0=malloc((size_t)nl*ne*4);
                if(G.heat0 && fread(G.heat0,4,(size_t)nl*ne,f)==(size_t)nl*ne){
                    for(size_t i=0;i<(size_t)nl*ne;i++)
                        G.slot[i].heat=tier_decay_value(G.heat0[i]);
                    fprintf(stderr,"[q38tier] HEAT_FILE loaded: %s\n",hf);
                } else { free(G.heat0); G.heat0=NULL; }
            }
            fclose(f);
        }
    }
    pthread_mutex_init(&G.mx,NULL);
    pthread_cond_init(&G.cv,NULL);
    pthread_cond_init(&G.cv_take,NULL);
    if(pthread_create(&G.th,NULL,uploader,NULL)!=0){ free(G.slot); free(G.is_x); return 0; }
    G.on=1;
    fprintf(stderr,"[q38tier] CUDA VRAM expert tier active: %d device(s), "
                   "%.2f MB/expert, %d experts total\n",
            G.ndev, G.exp_bytes/1048576.0, nl*ne);
    return 1;
}

int q38t_ready(void){ return G.on; }

int q38t_is_resident(int layer,int eid){
    if(!G.on||layer<0||layer>=G.nl||eid<0||eid>=G.ne) return 0;
    pthread_mutex_lock(&G.mx);
    int r=qs(layer,eid)->resident;
    pthread_mutex_unlock(&G.mx);
    return r;
}

/* --- calore -------------------------------------------------------------- */

void q38t_note(int layer,const int *eids,int K){
    if(!G.on||!eids||layer<0||layer>=G.nl) return;
    pthread_mutex_lock(&G.mx);
    if(layer==0){
        G.tick++;
        /* Decadimento periodico: un carico vecchio non deve possedere la VRAM
         * per sempre. Le promozioni, a differenza di qwen36_tier, non si
         * decidono qui -- servono i byte, e qui non ci sono. */
        if(!(G.tick%1024)){
            size_t n=(size_t)G.nl*G.ne;
            for(size_t i=0;i<n;i++) G.slot[i].heat=tier_decay_value(G.slot[i].heat);
        }
    }
    for(int k=0;k<K;k++){
        int e=eids[k];
        if(e<0||e>=G.ne) continue;
        Q38TSlot *s=qs(layer,e);
        if(s->heat<0xFFFFFFFFu) s->heat++;
    }
    pthread_mutex_unlock(&G.mx);
}

/* --- offerta e promozione ------------------------------------------------ */

/* Chiamata con il lock preso. Torna 1 se accodato. */
static int enqueue_locked(int layer,int eid,int v_layer,int v_eid,
                          const uint8_t *gate,const uint8_t *up,
                          const uint8_t *down,const float *scales){
    if(G.qn>=Q38T_QCAP){ G.q_full_skips++; return 0; }
    uint8_t *w=malloc(3*G.mat_bytes);
    float *sc=malloc(3*G.sc*sizeof(float));
    if(!w||!sc){ free(w); free(sc); G.q_full_skips++; return 0; }
    stage(w,sc,gate,up,down,scales);
    G.q[G.qt_].layer=layer; G.q[G.qt_].eid=eid;
    G.q[G.qt_].w=w;         G.q[G.qt_].s=sc;
    G.q[G.qt_].v_layer=v_layer; G.q[G.qt_].v_eid=v_eid;
    G.qt_=(G.qt_+1)%Q38T_QCAP; G.qn++;
    qs(layer,eid)->queued=1;
    pthread_cond_signal(&G.cv);
    return 1;
}

void q38t_offer(int layer,int eid,
                const uint8_t *gate,const uint8_t *up,const uint8_t *down,
                const float *scales,int planned){
    if(!G.on||!gate||!up||!down||!scales) return;
    if(layer<0||layer>=G.nl||eid<0||eid>=G.ne) return;
    pthread_mutex_lock(&G.mx);
    G.offers++;
    Q38TSlot *s=qs(layer,eid);
    if(s->resident||s->queued){ pthread_mutex_unlock(&G.mx); goto out; }

    int di=home(eid);
    if(planned){
        /* Il budget e' gia' stato riservato da q38t_plan_fill: si accoda e
         * basta. Se il piano nel frattempo e' stato annullato (planned=0), si
         * ricade sulla strada normale qui sotto. */
        if(s->planned){
            if(enqueue_locked(layer,eid,-1,-1,gate,up,down,scales)) G.promotions++;
            else { if(G.used[di]>=G.exp_bytes) G.used[di]-=G.exp_bytes; s->planned=0; }
            pthread_mutex_unlock(&G.mx); goto out;
        }
    }

    if(G.used[di]+G.exp_bytes<=G.budget[di]){
        G.used[di]+=G.exp_bytes;
        if(enqueue_locked(layer,eid,-1,-1,gate,up,down,scales)) G.promotions++;
        else G.used[di]-=G.exp_bytes;
        pthread_mutex_unlock(&G.mx); goto out;
    }

    /* Scheda piena: si entra solo scalzando il piu' freddo residente di questa
     * stessa scheda, e solo se il contratto condiviso (tier.h, con isteresi) lo
     * autorizza. La scansione e' lineare su nl*ne slot: a un miss per esperto e
     * un token da oltre un secondo e' rumore, e tenere una struttura ordinata
     * costerebbe piu' complessita' di quanta ne faccia risparmiare. */
    {
        size_t n=(size_t)G.nl*G.ne;
        long cold=-1; uint32_t ch=0;
        for(size_t i=0;i<n;i++){
            Q38TSlot *c=&G.slot[i];
            if(!c->resident||c->queued) continue;
            if(home((int)(i%G.ne))!=di) continue;
            if(cold<0||c->heat<ch){ cold=(long)i; ch=c->heat; }
        }
        if(cold>=0 && tier_should_promote(s->heat,ch)){
            Q38TSlot *v=&G.slot[cold];
            v->resident=0;                       /* da ora e' CPU fallback */
            if(enqueue_locked(layer,eid,(int)(cold/G.ne),(int)(cold%G.ne),
                              gate,up,down,scales)) G.swaps++;
            else v->resident=1;                  /* coda piena: si rimette */
        }
    }
    pthread_mutex_unlock(&G.mx);
out:
    return;
}

/* --- esecuzione ---------------------------------------------------------- */

uint32_t q38t_issue(int layer,const int *eids,int K,const float *x){
    if(!G.on||!eids||!x||K<1||K>32||layer<0||layer>=G.nl) return 0;
    uint32_t mask=0;
    ColiCudaTensor *tg[Q38T_MAX_DEV][Q38T_MAX_ROWS];
    ColiCudaTensor *tu[Q38T_MAX_DEV][Q38T_MAX_ROWS];
    ColiCudaTensor *td[Q38T_MAX_DEV][Q38T_MAX_ROWS];
    static const int rows[Q38T_MAX_ROWS]={1,1,1,1,1,1,1,1};
    for(int i=0;i<G.ndev;i++) G.is_cnt[i]=0;

    pthread_mutex_lock(&G.mx);
    G.issue_open=1;
    for(int k=0;k<K;k++){
        int e=eids[k];
        if(e<0||e>=G.ne){ G.miss++; continue; }
        Q38TSlot *s=qs(layer,e);
        if(!s->resident){ G.miss++; continue; }
        int di=home(e), c=G.is_cnt[di];
        /* Il backend tiene un solo issue in volo per device e ne rifiuta piu'
         * di Q38T_MAX_ROWS righe, quindi una scheda che si vede arrivare piu'
         * esperti del tetto non puo' spezzare il lancio: i sopranumerari
         * restano alla CPU via maschera. Con topk=10 e due schede la
         * spartizione eid%2 ne manda ~5 per parte e il caso non si presenta,
         * ma niente lo vieta e il conteggio lo dice (overflow_rows). */
        if(c>=Q38T_MAX_ROWS){ G.miss++; G.overflow_rows++; continue; }
        tg[di][c]=s->tg; tu[di][c]=s->tu; td[di][c]=s->td;
        G.is_k[di][c]=k; G.is_cnt[di]=c+1;
        mask|=1u<<k; G.hits[di]++;
    }
    pthread_mutex_unlock(&G.mx);

    for(int di=0;di<G.ndev;di++){
        int c=G.is_cnt[di];
        if(!c) continue;
        float *xr=G.is_x + (size_t)di*Q38T_MAX_ROWS*G.D;
        for(int j=0;j<c;j++) memcpy(xr+(size_t)j*G.D, x, (size_t)G.D*sizeof(float));
        if(!coli_cuda_expert_group_issue(tg[di],tu[di],td[di],rows,c,xr)){
            for(int j=0;j<c;j++) mask &= ~(1u<<G.is_k[di][j]);
            G.is_cnt[di]=0;
        }
    }
    if(!mask){
        pthread_mutex_lock(&G.mx);
        G.issue_open=0;
        pthread_cond_broadcast(&G.cv_take);
        pthread_mutex_unlock(&G.mx);
    }
    return mask;
}

void q38t_take(uint32_t mask,const float *val,int K,float *out){
    (void)K;
    if(!G.on) return;
    if(mask&&val&&out) for(int di=0;di<G.ndev;di++){
        int c=G.is_cnt[di];
        if(!c) continue;
        const float *y=coli_cuda_expert_group_take(G.dev[di]);
        if(!y){
            /* Questi k erano nella maschera: la CPU li ha saltati e adesso
             * nessuno li calcola, cioe' il token esce con un pezzo di MoE in
             * meno. Non e' recuperabile qui (i pesi in RAM sono gia' stati
             * sfrattati), ma non deve passare in silenzio. */
            G.take_fails++;
            if(G.take_fails==1)
                fprintf(stderr,"[q38tier] WARNING: expert_group_take failed on dev %d; "
                               "%d routed expert(s) dropped from this token\n",
                        G.dev[di], c);
            G.is_cnt[di]=0;
            continue;
        }
        for(int j=0;j<c;j++){
            float w=val[G.is_k[di][j]];
            const float *row=y+(size_t)j*G.D;
            for(int d=0;d<G.D;d++) out[d]+=w*row[d];
        }
        G.is_cnt[di]=0;
    }
    pthread_mutex_lock(&G.mx);
    G.issue_open=0;
    pthread_cond_broadcast(&G.cv_take);
    pthread_mutex_unlock(&G.mx);
}

/* --- warmstart ----------------------------------------------------------- */

static const uint32_t *g_sort_heat;
static int cmp_heat_desc(const void *a,const void *b){
    uint32_t ha=g_sort_heat[*(const int*)a], hb=g_sort_heat[*(const int*)b];
    return ha<hb ? 1 : (ha>hb ? -1 : 0);
}

int q38t_plan_fill(int *layers,int *eids,int max){
    if(!G.on||!layers||!eids||max<1) return 0;
    if(getenv("Q38T_NO_WARMSTART")) return 0;
    size_t n=(size_t)G.nl*G.ne;
    pthread_mutex_lock(&G.mx);
    if(!G.fill_order){
        /* Senza HEAT_FILE non c'e' un ordine sensato da inventare: si lascia
         * che sia il traffico a promuovere, che e' gia' la strada normale. */
        if(!G.heat0){ pthread_mutex_unlock(&G.mx); return 0; }
        G.fill_order=malloc(n*sizeof(int));
        if(!G.fill_order){ pthread_mutex_unlock(&G.mx); return 0; }
        for(size_t i=0;i<n;i++) G.fill_order[i]=(int)i;
        g_sort_heat=G.heat0;
        qsort(G.fill_order,n,sizeof(int),cmp_heat_desc);
        G.fill_n=(int)n; G.fill_cur=0;
    }
    int out=0;
    while(out<max && G.fill_cur<G.fill_n){
        int i=G.fill_order[G.fill_cur++];
        int layer=i/G.ne, eid=i%G.ne, di=home(eid);
        Q38TSlot *s=&G.slot[i];
        if(s->resident||s->queued||s->planned) continue;
        if(G.used[di]+G.exp_bytes>G.budget[di]) continue;
        G.used[di]+=G.exp_bytes;
        s->planned=1;
        layers[out]=layer; eids[out]=eid; out++;
    }
    pthread_mutex_unlock(&G.mx);
    return out;
}

void q38t_cancel_plan(int layer,int eid){
    if(!G.on||layer<0||layer>=G.nl||eid<0||eid>=G.ne) return;
    pthread_mutex_lock(&G.mx);
    Q38TSlot *s=qs(layer,eid);
    if(s->planned&&!s->queued&&!s->resident){
        int di=home(eid);
        if(G.used[di]>=G.exp_bytes) G.used[di]-=G.exp_bytes;
        s->planned=0;
    }
    pthread_mutex_unlock(&G.mx);
}

void q38t_fill_wait(void){
    if(!G.on) return;
    pthread_mutex_lock(&G.mx);
    while(G.qn>0 && !G.th_stop) pthread_cond_wait(&G.cv_take,&G.mx);
    pthread_mutex_unlock(&G.mx);
}

/* --- chiusura e telemetria ----------------------------------------------- */

void q38t_stats(void){
    if(!G.on) return;
    pthread_mutex_lock(&G.mx);
    size_t n=(size_t)G.nl*G.ne;
    uint64_t res=0, hit=0;
    for(size_t i=0;i<n;i++) if(G.slot[i].resident) res++;
    fprintf(stderr,"[q38tier] resident %llu/%zu experts (%.1f%%)",
            (unsigned long long)res, n, n? 100.0*res/(double)n : 0.0);
    for(int i=0;i<G.ndev;i++){
        hit+=G.hits[i];
        fprintf(stderr," | dev%d hits %llu, %.1f/%.1f GB",
                G.dev[i],(unsigned long long)G.hits[i],
                G.used[i]/1073741824.0, G.budget[i]/1073741824.0);
    }
    fprintf(stderr,"\n[q38tier] gpu %llu, cpu %llu (%.1f%% on GPU) | offers %llu,"
                   " promotions %llu, swaps %llu, uploads %llu, failed %llu,"
                   " queue-full %llu, row-overflow %llu, take-fail %llu\n",
            (unsigned long long)hit,(unsigned long long)G.miss,
            (hit+G.miss)? 100.0*hit/(double)(hit+G.miss) : 0.0,
            (unsigned long long)G.offers,(unsigned long long)G.promotions,
            (unsigned long long)G.swaps,(unsigned long long)G.uploads,
            (unsigned long long)G.upload_fail,
            (unsigned long long)G.q_full_skips,
            (unsigned long long)G.overflow_rows,
            (unsigned long long)G.take_fails);
    pthread_mutex_unlock(&G.mx);
}

/* Il calore imparato in questa sessione, per la prossima. Stesso formato
 * dell'altro tier (magia, layer, esperti, poi la matrice), cosi' un file scritto
 * qui e' leggibile da qui e basta: le geometrie non coincidono fra i due motori
 * e l'header le controlla prima di fidarsi. */
static void heat_save(void){
    const char *hf=getenv("HEAT_FILE");
    if(!hf) return;
    FILE *f=fopen(hf,"wb");
    if(!f){ fprintf(stderr,"[q38tier] cannot write HEAT_FILE %s\n",hf); return; }
    uint32_t hdr[3]={0x51544831u,(uint32_t)G.nl,(uint32_t)G.ne};
    size_t n=(size_t)G.nl*G.ne;
    uint32_t *h=malloc(n*4);
    int ok=0;
    if(h){
        for(size_t i=0;i<n;i++) h[i]=G.slot[i].heat;
        ok = fwrite(hdr,4,3,f)==3 && fwrite(h,4,n,f)==n;
        free(h);
    }
    if(fclose(f)!=0) ok=0;
    if(!ok) fprintf(stderr,"[q38tier] HEAT_FILE %s written incompletely\n",hf);
}

void q38t_shutdown(void){
    if(!G.on) return;
    heat_save();
    pthread_mutex_lock(&G.mx);
    G.th_stop=1;
    pthread_cond_broadcast(&G.cv);
    pthread_cond_broadcast(&G.cv_take);
    pthread_mutex_unlock(&G.mx);
    pthread_join(G.th,NULL);
    /* Quel che resta in coda non e' mai stato caricato: si libera lo staging. */
    while(G.qn>0){ free(G.q[G.qh].w); free(G.q[G.qh].s);
                   G.qh=(G.qh+1)%Q38T_QCAP; G.qn--; }
    size_t n=(size_t)G.nl*G.ne;
    for(size_t i=0;i<n;i++){
        Q38TSlot *s=&G.slot[i];
        if(s->tg) coli_cuda_tensor_free(s->tg);
        if(s->tu) coli_cuda_tensor_free(s->tu);
        if(s->td) coli_cuda_tensor_free(s->td);
        s->tg=s->tu=s->td=NULL; s->resident=0;
    }
    free(G.slot); free(G.is_x); free(G.fill_order); free(G.heat0);
    G.slot=NULL; G.is_x=NULL; G.fill_order=NULL; G.heat0=NULL;
    G.on=0;
}

#endif /* COLI_CUDA */
