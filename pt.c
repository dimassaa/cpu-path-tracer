/*
 * pt.c - a small multithreaded CPU path tracer.
 *
 * Why this file exists: written from scratch in one sitting as a self-contained
 * toy. No external dependencies beyond libc/pthreads: geometry, BSDFs,
 * direct-light sampling, tone mapping and a minimal PNG encoder all live here.
 *
 * Design notes:
 *  - Everything is a triangle or a sphere; the Cornell box walls are quads
 *    split into two triangles. That keeps the intersection code tiny.
 *  - Direct lighting uses next-event estimation against a single rectangular
 *    area light, which is what makes the box converge in a few hundred
 *    samples instead of many thousands. Emission is only added when the path has seen
 *    specular bounces (or on the primary ray) so it is not counted twice.
 *  - RNG is PCG32 seeded per (pixel, sample), so the image is deterministic
 *    and rows can be rendered in any order.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>

/* ------------------------------------------------------------------ vec3 */

typedef struct { float x, y, z; } vec3;

static inline vec3 v3(float x, float y, float z) { vec3 r = {x, y, z}; return r; }
static inline vec3 vadd(vec3 a, vec3 b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline vec3 vsub(vec3 a, vec3 b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline vec3 vmul(vec3 a, float s) { return v3(a.x * s, a.y * s, a.z * s); }
static inline vec3 vmulv(vec3 a, vec3 b) { return v3(a.x * b.x, a.y * b.y, a.z * b.z); }
static inline float vdot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline vec3 vcross(vec3 a, vec3 b) {
    return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline float vlen(vec3 a) { return sqrtf(vdot(a, a)); }
static inline vec3 vnorm(vec3 a) { float l = vlen(a); return l > 0 ? vmul(a, 1.0f / l) : a; }
static inline vec3 vneg(vec3 a) { return v3(-a.x, -a.y, -a.z); }
static inline vec3 vclamp01(vec3 a) {
    return v3(fminf(fmaxf(a.x, 0), 1), fminf(fmaxf(a.y, 0), 1), fminf(fmaxf(a.z, 0), 1));
}

/* ------------------------------------------------------------------- rng */

typedef struct { uint64_t state, inc; } pcg32;

static inline uint32_t pcg_next(pcg32 *r) {
    uint64_t old = r->state;
    r->state = old * 6364136223846793005ULL + r->inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static inline float rnd(pcg32 *r) { return (float)(pcg_next(r) >> 8) * (1.0f / 16777216.0f); }

static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static void pcg_seed(pcg32 *r, uint64_t seed) {
    r->state = 0;
    r->inc = (seed << 1u) | 1u;
    pcg_next(r);
    r->state += splitmix64(seed);
    pcg_next(r);
}

/* ----------------------------------------------------------------- scene */

enum { M_DIFFUSE, M_METAL, M_GLASS, M_EMIT };

typedef struct {
    int type;
    vec3 albedo;   /* base colour, also used as metal tint */
    vec3 emit;     /* emission, only meaningful for M_EMIT */
    float rough;   /* metal fuzz */
    float ior;     /* glass index of refraction */
    float checker; /* >0: multiply albedo by a checker pattern with this scale */
} material;

typedef struct { vec3 c; float r; int mat; } sphere;
typedef struct { vec3 a, b, c; vec3 n; int mat; } tri;

typedef struct {
    /* The area light is described analytically so it can be sampled for NEE. */
    vec3 center, u, v; /* center +/- u +/- v spans the rectangle */
    vec3 normal, emit;
    float area;
} rect_light;

#define MAX_SPHERES 64
#define MAX_TRIS 64
#define MAX_MATS 32

typedef struct {
    sphere sph[MAX_SPHERES]; int nsph;
    tri    tr[MAX_TRIS];     int ntri;
    material mat[MAX_MATS];  int nmat;
    rect_light light;
    int has_light;
    vec3 bg_top, bg_bot;      /* environment gradient (may be near black) */
    float bg_strength;
    /* Optional distance fade for the checker texture: keeps the pattern from
     * turning into moire where one cell is smaller than a pixel. */
    float checker_fade_start, checker_fade_end;
} scene;

/* Fixed-capacity scene storage: fail loudly rather than corrupt memory. */
static void die_full(const char *what) {
    fprintf(stderr, "scene capacity exceeded: %s\n", what);
    exit(1);
}
static void add_mat(scene *s, material m) {
    if (s->nmat >= MAX_MATS) die_full("materials");
    s->mat[s->nmat++] = m;
}
static void add_sphere(scene *s, vec3 c, float r, int mat) {
    if (s->nsph >= MAX_SPHERES) die_full("spheres");
    s->sph[s->nsph++] = (sphere){c, r, mat};
}
static void add_tri(scene *s, vec3 a, vec3 b, vec3 c, int mat) {
    if (s->ntri >= MAX_TRIS) die_full("triangles");
    vec3 n = vnorm(vcross(vsub(b, a), vsub(c, a)));
    s->tr[s->ntri++] = (tri){a, b, c, n, mat};
}
/* Adds a quad a-b-c-d (in order around the perimeter) as two triangles. */
static void add_quad(scene *s, vec3 a, vec3 b, vec3 c, vec3 d, int mat) {
    add_tri(s, a, b, c, mat);
    add_tri(s, a, c, d, mat);
}

/* ------------------------------------------------------------------ rays */

typedef struct { vec3 o, d; } ray;
typedef struct { float t; vec3 p, n; int mat; int valid; } hit;

static hit intersect(const scene *s, ray r, float tmin, float tmax) {
    hit h = {.valid = 0, .t = tmax};

    for (int i = 0; i < s->nsph; i++) {
        const sphere *sp = &s->sph[i];
        vec3 oc = vsub(r.o, sp->c);
        float b = vdot(oc, r.d);
        float c = vdot(oc, oc) - sp->r * sp->r;
        float disc = b * b - c;
        if (disc < 0) continue;
        float sq = sqrtf(disc);
        float t = -b - sq;
        if (t < tmin) t = -b + sq;
        if (t > tmin && t < h.t) {
            h.t = t; h.valid = 1; h.mat = sp->mat;
            h.p = vadd(r.o, vmul(r.d, t));
            h.n = vnorm(vsub(h.p, sp->c));
        }
    }

    for (int i = 0; i < s->ntri; i++) {
        const tri *tr = &s->tr[i];
        /* Moller-Trumbore, two-sided: the shading code orients the normal. */
        vec3 e1 = vsub(tr->b, tr->a), e2 = vsub(tr->c, tr->a);
        vec3 pv = vcross(r.d, e2);
        float det = vdot(e1, pv);
        if (fabsf(det) < 1e-12f) continue;
        float inv = 1.0f / det;
        vec3 tv = vsub(r.o, tr->a);
        float u = vdot(tv, pv) * inv;
        if (u < 0 || u > 1) continue;
        vec3 qv = vcross(tv, e1);
        float v = vdot(r.d, qv) * inv;
        if (v < 0 || u + v > 1) continue;
        float t = vdot(e2, qv) * inv;
        if (t > tmin && t < h.t) {
            h.t = t; h.valid = 1; h.mat = tr->mat;
            h.p = vadd(r.o, vmul(r.d, t));
            h.n = tr->n;
        }
    }
    return h;
}

/*
 * Shadow test for next-event estimation. Emissive geometry does not count as
 * an occluder: the light itself is a quad and must not shadow its own samples.
 */
static int occluded(const scene *s, vec3 o, vec3 d, float maxdist) {
    for (int i = 0; i < s->nsph; i++) {
        const sphere *sp = &s->sph[i];
        vec3 oc = vsub(o, sp->c);
        float b = vdot(oc, d);
        float c = vdot(oc, oc) - sp->r * sp->r;
        float disc = b * b - c;
        if (disc < 0) continue;
        /*
         * Entry root only. Shadow rays start just outside the surface and
         * head away, so the entry root is the only relevant crossing. Falling
         * back to the exit root for t < 0 was wrong here: -b + sqrt(b*b-c)
         * subtracts two nearly equal numbers when the ray points away from the
         * sphere, and the rounding error could flip it just above zero. That
         * produced periodic false self-shadowing on large spheres.
         */
        float t = -b - sqrtf(disc);
        if (t > 1e-4f && t < maxdist) return 1;
    }
    for (int i = 0; i < s->ntri; i++) {
        const tri *tr = &s->tr[i];
        if (s->mat[tr->mat].type == M_EMIT) continue;
        vec3 e1 = vsub(tr->b, tr->a), e2 = vsub(tr->c, tr->a);
        vec3 pv = vcross(d, e2);
        float det = vdot(e1, pv);
        if (fabsf(det) < 1e-12f) continue;
        float inv = 1.0f / det;
        vec3 tv = vsub(o, tr->a);
        float u = vdot(tv, pv) * inv;
        if (u < 0 || u > 1) continue;
        vec3 qv = vcross(tv, e1);
        float v = vdot(d, qv) * inv;
        if (v < 0 || u + v > 1) continue;
        float t = vdot(e2, qv) * inv;
        if (t > 1e-4f && t < maxdist) return 1;
    }
    return 0;
}

/* ------------------------------------------------------------- sampling */

static inline float clampf(float x, float a, float b) { return x < a ? a : (x > b ? b : x); }

/* Branchless orthonormal basis from a unit vector (Duff et al. 2017). */
static void onb(vec3 n, vec3 *u, vec3 *v) {
    float sign = copysignf(1.0f, n.z);
    float a = -1.0f / (sign + n.z);
    float b = n.x * n.y * a;
    *u = v3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    *v = v3(b, sign + n.y * n.y * a, -n.y);
}

static vec3 cosine_hemisphere(vec3 n, pcg32 *rng) {
    float r1 = rnd(rng), r2 = rnd(rng);
    float r = sqrtf(r1), phi = 6.28318530718f * r2;
    float x = r * cosf(phi), y = r * sinf(phi);
    float z = sqrtf(fmaxf(0.0f, 1.0f - r1));
    vec3 u, v;
    onb(n, &u, &v);
    return vnorm(vadd(vadd(vmul(u, x), vmul(v, y)), vmul(n, z)));
}

static vec3 random_in_sphere(pcg32 *rng) {
    for (;;) {
        vec3 p = v3(2 * rnd(rng) - 1, 2 * rnd(rng) - 1, 2 * rnd(rng) - 1);
        if (vdot(p, p) < 1.0f) return p;
    }
}

/* --------------------------------------------------------------- shading */

static inline vec3 albedo_at(const scene *s, const material *m, vec3 p) {
    if (m->checker <= 0) return m->albedo;
    /* Round to cell centres instead of cell edges. A floor lying exactly on
     * y=0 produces hit points with y = +/-epsilon, and floorf(y/scale) then
     * flips between -1 and 0 per sample, which turns the pattern into noise. */
    int x = (int)floorf(p.x / m->checker + 0.5f);
    int y = (int)floorf(p.y / m->checker + 0.5f);
    int z = (int)floorf(p.z / m->checker + 0.5f);
    float c = ((x + y + z) & 1) ? 0.18f : 0.85f;
    if (s->checker_fade_end > s->checker_fade_start) {
        float d = sqrtf(vdot(p, p));
        float t = clampf((d - s->checker_fade_start) / (s->checker_fade_end - s->checker_fade_start), 0, 1);
        c += (0.515f - c) * t;
    }
    return vmul(m->albedo, c);
}

/* Samples the area light and returns the direct diffuse contribution. */
static vec3 direct_light(const scene *s, vec3 p, vec3 n, vec3 albedo, pcg32 *rng) {
    if (!s->has_light) return v3(0, 0, 0);
    const rect_light *L = &s->light;
    float r1 = rnd(rng), r2 = rnd(rng);
    vec3 q = vadd(L->center, vadd(vmul(L->u, 2 * r1 - 1), vmul(L->v, 2 * r2 - 1)));
    vec3 d = vsub(q, p);
    float dist2 = vdot(d, d);
    float dist = sqrtf(dist2);
    vec3 wi = vmul(d, 1.0f / dist);

    float cos_s = vdot(n, wi);
    float cos_l = -vdot(L->normal, wi); /* light normal points into the room */
    if (cos_s <= 0 || cos_l <= 0) return v3(0, 0, 0);

    vec3 origin = vadd(p, vmul(n, 1e-3f));
    if (occluded(s, origin, wi, dist - 1e-3f)) return v3(0, 0, 0);

    /* f = albedo/pi, pdf = 1/area, G = cos_s*cos_l/dist2  */
    float scale = (cos_s * cos_l) * L->area / (3.14159265358979f * dist2);
    return vmul(vmulv(albedo, L->emit), scale);
}

/* Fresnel reflectance for a relative index ratio eta (n2/n1). */
static inline float schlick(float cos_theta, float eta) {
    float r0 = (1 - eta) / (1 + eta);
    r0 *= r0;
    return r0 + (1 - r0) * powf(1 - cos_theta, 5);
}

static vec3 sky(const scene *s, vec3 d) {
    /* Cheap smooth gradient: no importance sampling, only used as a soft
     * backdrop for the open scene. */
    float t = clampf(0.5f * (d.y + 1.0f), 0, 1);
    vec3 c = vadd(vmul(s->bg_bot, 1 - t), vmul(s->bg_top, t));
    return vmul(c, s->bg_strength);
}

static vec3 trace(const scene *s, ray r, pcg32 *rng, int max_depth) {
    vec3 radiance = v3(0, 0, 0);
    vec3 throughput = v3(1, 1, 1);
    int specular_path = 1; /* primary ray may see the light directly */

    for (int depth = 0; depth < max_depth; depth++) {
        hit h = intersect(s, r, 1e-3f, 1e30f);
        if (!h.valid) {
            if (s->bg_strength > 0) radiance = vadd(radiance, vmulv(throughput, sky(s, r.d)));
            break;
        }

        const material *m = &s->mat[h.mat];
        vec3 n = h.n;
        int front = vdot(n, r.d) < 0;
        if (!front) n = vneg(n);

        if (m->type == M_EMIT) {
            /* Already handled by NEE unless the path is purely specular. */
            if (specular_path) {
                vec3 contrib = vmulv(throughput, m->emit);
                /* A specular chain that finds the light is a real caustic or
                 * highlight, but these events are so rare that they show up as
                 * isolated white fireflies. Clamping the indirect contribution
                 * trades a little unbiasedness for a lot of visual comfort. */
                if (depth > 0) {
                    float peak = fmaxf(contrib.x, fmaxf(contrib.y, contrib.z));
                    if (peak > 8.0f) contrib = vmul(contrib, 8.0f / peak);
                }
                radiance = vadd(radiance, contrib);
            }
            break;
        }

        vec3 albedo = albedo_at(s, m, h.p);
        if (m->type == M_DIFFUSE) {
            radiance = vadd(radiance, vmulv(throughput, direct_light(s, h.p, n, albedo, rng)));
            throughput = vmulv(throughput, albedo);
            specular_path = 0;
            r.o = vadd(h.p, vmul(n, 1e-3f));
            r.d = cosine_hemisphere(n, rng);
        } else if (m->type == M_METAL) {
            vec3 d = vsub(r.d, vmul(n, 2.0f * vdot(r.d, n)));
            d = vnorm(vadd(d, vmul(random_in_sphere(rng), m->rough)));
            if (vdot(d, n) <= 0) break;
            throughput = vmulv(throughput, albedo);
            specular_path = 1;
            r.o = vadd(h.p, vmul(n, 1e-3f));
            r.d = d;
        } else { /* glass */
            float eta = front ? (1.0f / m->ior) : m->ior;
            float cos_i = -vdot(r.d, n);
            float sin2_t = eta * eta * (1.0f - cos_i * cos_i);
            vec3 dir;
            if (sin2_t > 1.0f || rnd(rng) < schlick(cos_i, eta)) {
                dir = vsub(r.d, vmul(n, 2.0f * vdot(r.d, n)));
            } else {
                float cos_t = sqrtf(1.0f - sin2_t);
                /* eta * d + (eta*cos_i - cos_t) * n, with n facing the ray */
                dir = vnorm(vadd(vmul(r.d, eta), vmul(n, eta * cos_i - cos_t)));
            }
            specular_path = 1;
            r.o = vadd(h.p, vmul(dir, 1e-3f));
            r.d = dir;
        }

        /* Russian roulette only after a few bounces: cheap and unbiased. */
        if (depth > 2) {
            float q = fmaxf(0.05f, fminf(0.95f, fmaxf(throughput.x, fmaxf(throughput.y, throughput.z))));
            if (rnd(rng) > q) break;
            throughput = vmul(throughput, 1.0f / q);
        }
    }
    return radiance;
}

/* --------------------------------------------------------------- camera */

typedef struct {
    vec3 origin, forward, right, up;
    float half_w, half_h;
    float aperture;  /* 0 disables depth of field */
    float focus;
} camera;

static void camera_setup(camera *c, vec3 from, vec3 to, float fov_deg, float aspect,
                         float aperture, float focus) {
    c->origin = from;
    c->forward = vnorm(vsub(to, from));
    vec3 world_up = v3(0, 1, 0);
    if (fabsf(vdot(c->forward, world_up)) > 0.999f) world_up = v3(0, 0, 1);
    c->right = vnorm(vcross(world_up, c->forward));
    c->up = vcross(c->forward, c->right);
    float th = tanf(fov_deg * 0.5f * 3.14159265358979f / 180.0f);
    c->half_h = th;
    c->half_w = th * aspect;
    c->aperture = aperture;
    c->focus = focus;
}

static ray camera_ray(const camera *c, float sx, float sy, pcg32 *rng) {
    /* sx, sy in [-1, 1] over the frame */
    vec3 dir = vnorm(vadd(c->forward,
                          vadd(vmul(c->right, sx * c->half_w), vmul(c->up, sy * c->half_h))));
    vec3 o = c->origin;
    if (c->aperture > 0) {
        float r = c->aperture * sqrtf(rnd(rng));
        float a = 6.28318530718f * rnd(rng);
        vec3 off = vadd(vmul(c->right, r * cosf(a)), vmul(c->up, r * sinf(a)));
        o = vadd(o, off);
        /* Shift the ray so rays through this lens point still hit the focal plane. */
        vec3 fp = vadd(c->origin, vmul(dir, c->focus / vdot(dir, c->forward)));
        dir = vnorm(vsub(fp, o));
    }
    return (ray){o, dir};
}

/* ---------------------------------------------------------------- image */

typedef struct { double r, g, b; } dvec3;

typedef struct {
    const scene *s;
    const camera *cam;
    int width, height, spp, max_depth;
    float exposure;
    dvec3 *accum;
    _Atomic int next_row;
    int rows_done;
    double t0;
    pthread_mutex_t lock;
} render_ctx;

/* Deterministic per-pixel hash in [0,1), used only for output dithering. */
static inline float hash01(int x, int y, uint32_t salt) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u + salt * 362437u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= h >> 16;
    return (float)(h & 0xFFFFFF) * (1.0f / 16777216.0f);
}

static inline vec3 aces(vec3 x) {
    /* Narkowicz 2015 ACES approximation: nicer highlight rolloff than Reinhard. */
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    vec3 num = vadd(vmulv(x, vadd(vmul(x, a), v3(b, b, b))), v3(0, 0, 0));
    vec3 den = vadd(vmulv(x, vadd(vmul(x, c), v3(d, d, d))), v3(e, e, e));
    return vclamp01(v3(num.x / den.x, num.y / den.y, num.z / den.z));
}

static void *worker(void *arg) {
    render_ctx *ctx = (render_ctx *)arg;
    const scene *s = ctx->s;
    const camera *cam = ctx->cam;

    for (;;) {
        int y = atomic_fetch_add(&ctx->next_row, 1);
        if (y >= ctx->height) break;
        for (int x = 0; x < ctx->width; x++) {
            dvec3 sum = {0, 0, 0};
            for (int k = 0; k < ctx->spp; k++) {
                pcg32 rng;
                pcg_seed(&rng, splitmix64((uint64_t)y * 0x9E3779B1u ^ (uint64_t)x * 0x85EBCA77u)
                                       + (uint64_t)k);
                float jx = (x + rnd(&rng)) / ctx->width;
                float jy = (y + rnd(&rng)) / ctx->height;
                float sx = 2.0f * jx - 1.0f;
                float sy = 1.0f - 2.0f * jy;
                ray r = camera_ray(cam, sx, sy, &rng);
                vec3 c = trace(s, r, &rng, ctx->max_depth);
                sum.r += c.x; sum.g += c.y; sum.b += c.z;
            }
            ctx->accum[(size_t)y * ctx->width + x] = sum;
        }
        pthread_mutex_lock(&ctx->lock);
        ctx->rows_done++;
        if ((ctx->rows_done % 64) == 0 || ctx->rows_done == ctx->height) {
            double dt = (double)(clock() - ctx->t0) / CLOCKS_PER_SEC;
            fprintf(stderr, "\r  %d/%d rows  %.1fs", ctx->rows_done, ctx->height, dt);
            fflush(stderr);
        }
        pthread_mutex_unlock(&ctx->lock);
    }
    return NULL;
}

/* ------------------------------------------------------------ png output */

static uint32_t crc_table[256];
static void crc_init(void) {
    for (uint32_t n = 0; n < 256; n++) {
        uint32_t c = n;
        for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crc_table[n] = c;
    }
}
static uint32_t crc32_buf(const uint8_t *buf, size_t len) {
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) c = crc_table[(c ^ buf[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}
static uint32_t adler32_buf(const uint8_t *buf, size_t len) {
    uint32_t a = 1, b = 0;
    for (size_t i = 0; i < len; i++) { a = (a + buf[i]) % 65521; b = (b + a) % 65521; }
    return (b << 16) | a;
}
static void put_be32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);  p[3] = (uint8_t)v;
}
static void write_chunk(FILE *f, const char *type, const uint8_t *data, uint32_t len) {
    uint8_t hdr[8];
    put_be32(hdr, len);
    memcpy(hdr + 4, type, 4);
    fwrite(hdr, 1, 8, f);
    if (len) fwrite(data, 1, len, f);
    uint8_t *all = malloc(len + 4);
    memcpy(all, type, 4);
    if (len) memcpy(all + 4, data, len);
    uint32_t crc = crc32_buf(all, len + 4);
    free(all);
    uint8_t tail[4];
    put_be32(tail, crc);
    fwrite(tail, 1, 4, f);
}

/*
 * Minimal PNG writer: 8-bit RGB, no filtering, one zlib stream made of
 * uncompressed ("stored") deflate blocks. Larger on disk than a real encoder
 * but a few dozen lines instead of a dependency.
 */
static int write_png(const char *path, int w, int h, const uint8_t *rgb) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    static const uint8_t sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    fwrite(sig, 1, 8, f);

    uint8_t ihdr[13];
    put_be32(ihdr, (uint32_t)w);
    put_be32(ihdr + 4, (uint32_t)h);
    ihdr[8] = 8;  /* bit depth */
    ihdr[9] = 2;  /* colour type: truecolour */
    ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
    write_chunk(f, "IHDR", ihdr, 13);

    size_t rowbytes = (size_t)w * 3;
    size_t rawlen = (rowbytes + 1) * h;
    uint8_t *raw = malloc(rawlen);
    for (int y = 0; y < h; y++) {
        raw[y * (rowbytes + 1)] = 0; /* filter: none */
        memcpy(raw + y * (rowbytes + 1) + 1, rgb + (size_t)y * rowbytes, rowbytes);
    }

    /* zlib header + stored blocks + adler32 */
    size_t nblocks = (rawlen + 65534) / 65535;
    size_t zlen = 2 + nblocks * 5 + rawlen + 4;
    uint8_t *z = malloc(zlen);
    size_t zpos = 0;
    z[zpos++] = 0x78; z[zpos++] = 0x01;
    size_t pos = 0;
    while (pos < rawlen) {
        size_t n = rawlen - pos > 65535 ? 65535 : rawlen - pos;
        int final = (pos + n == rawlen);
        z[zpos++] = (uint8_t)(final ? 1 : 0);
        z[zpos++] = (uint8_t)(n & 0xFF);
        z[zpos++] = (uint8_t)((n >> 8) & 0xFF);
        z[zpos++] = (uint8_t)(~n & 0xFF);
        z[zpos++] = (uint8_t)((~n >> 8) & 0xFF);
        memcpy(z + zpos, raw + pos, n);
        zpos += n;
        pos += n;
    }
    uint32_t ad = adler32_buf(raw, rawlen);
    put_be32(z + zpos, ad);
    zpos += 4;

    write_chunk(f, "IDAT", z, (uint32_t)zpos);
    write_chunk(f, "IEND", NULL, 0);
    free(z);
    free(raw);
    fclose(f);
    return 1;
}

/* ------------------------------------------------------------------ main */

static int add_default_box(scene *s) {
    int white = s->nmat; add_mat(s, (material){M_DIFFUSE, v3(0.73f, 0.73f, 0.73f), v3(0, 0, 0), 0, 1.5f, 0});
    int red   = s->nmat; add_mat(s, (material){M_DIFFUSE, v3(0.62f, 0.12f, 0.10f), v3(0, 0, 0), 0, 1.5f, 0});
    int green = s->nmat; add_mat(s, (material){M_DIFFUSE, v3(0.12f, 0.48f, 0.16f), v3(0, 0, 0), 0, 1.5f, 0});
    int glass = s->nmat; add_mat(s, (material){M_GLASS, v3(0.96f, 0.98f, 1.0f), v3(0, 0, 0), 0, 1.5f, 0});
    int metal = s->nmat; add_mat(s, (material){M_METAL, v3(0.95f, 0.86f, 0.62f), v3(0, 0, 0), 0.12f, 0, 0});
    int floor = s->nmat; add_mat(s, (material){M_DIFFUSE, v3(0.80f, 0.82f, 0.84f), v3(0, 0, 0), 0, 1.5f, 0});
    int cyan  = s->nmat; add_mat(s, (material){M_DIFFUSE, v3(0.10f, 0.42f, 0.55f), v3(0, 0, 0), 0, 1.5f, 0});

    /* Cornell box, 555 units on a side, open toward -z (the camera). */
    float S = 555.0f;
    add_quad(s, v3(0, 0, 0), v3(S, 0, 0), v3(S, 0, S), v3(0, 0, S), floor);
    add_quad(s, v3(0, S, 0), v3(S, S, 0), v3(S, S, S), v3(0, S, S), white);
    add_quad(s, v3(0, 0, S), v3(S, 0, S), v3(S, S, S), v3(0, S, S), white);
    add_quad(s, v3(0, 0, 0), v3(0, 0, S), v3(0, S, S), v3(0, S, 0), red);
    add_quad(s, v3(S, 0, 0), v3(S, 0, S), v3(S, S, S), v3(S, S, 0), green);

    /* Ceiling light: analytic quad for sampling, plus geometry so it is visible. */
    s->light.center = v3(278, S - 0.5f, 279);
    s->light.u = v3(65, 0, 0);
    s->light.v = v3(0, 0, 52);
    s->light.normal = v3(0, -1, 0);
    s->light.emit = v3(18, 17.4f, 15.5f);
    s->light.area = 4.0f * vlen(vcross(s->light.u, s->light.v));
    s->has_light = 1;
    int lamp = s->nmat;
    add_mat(s, (material){M_EMIT, v3(0, 0, 0), s->light.emit, 0, 0, 0});
    vec3 lc = s->light.center;
    add_quad(s, vsub(vsub(lc, s->light.u), s->light.v),
                vsub(vadd(lc, s->light.u), s->light.v),
                vadd(vadd(lc, s->light.u), s->light.v),
                vadd(vsub(lc, s->light.u), s->light.v), lamp);

    /* Objects. */
    add_sphere(s, v3(165, 120, 240), 120, glass);
    add_sphere(s, v3(400, 100, 150), 100, metal);
    add_sphere(s, v3(300, 70, 410), 70, cyan);

    return 1;
}

static int add_open_scene(scene *s) {
    int checker = s->nmat;
    add_mat(s, (material){M_DIFFUSE, v3(0.86f, 0.84f, 0.80f), v3(0, 0, 0), 0, 1.5f, 260.0f});
    int glass = s->nmat;
    add_mat(s, (material){M_GLASS, v3(0.96f, 0.98f, 1.0f), v3(0, 0, 0), 0, 1.6f, 0});
    int gold = s->nmat;
    add_mat(s, (material){M_METAL, v3(0.98f, 0.80f, 0.35f), v3(0, 0, 0), 0.02f, 0, 0});
    int chrome = s->nmat;
    add_mat(s, (material){M_METAL, v3(0.92f, 0.94f, 0.96f), v3(0, 0, 0), 0.05f, 0, 0});
    int red = s->nmat;
    add_mat(s, (material){M_DIFFUSE, v3(0.72f, 0.15f, 0.12f), v3(0, 0, 0), 0, 1.5f, 0});

    add_quad(s, v3(-1e6f, 0, -1e6f), v3(1e6f, 0, -1e6f),
                v3(1e6f, 0, 1e6f), v3(-1e6f, 0, 1e6f), checker);
    add_sphere(s, v3(-150, 115, 60), 115, glass);
    add_sphere(s, v3(120, 90, 110), 90, gold);
    add_sphere(s, v3(5, 60, 260), 60, chrome);
    add_sphere(s, v3(235, 55, 10), 55, red);

    /* A soft "sun" quad high up: small enough to give readable shadows. */
    s->light.center = v3(0, 900, 100);
    s->light.u = v3(150, 0, 0);
    s->light.v = v3(0, 0, 150);
    s->light.normal = v3(0, -1, 0);
    s->light.emit = v3(32, 30, 27);
    s->light.area = 4.0f * vlen(vcross(s->light.u, s->light.v));
    s->has_light = 1;
    int lamp = s->nmat;
    add_mat(s, (material){M_EMIT, v3(0, 0, 0), s->light.emit, 0, 0, 0});
    vec3 lc = s->light.center;
    add_quad(s, vsub(vsub(lc, s->light.u), s->light.v),
                vsub(vadd(lc, s->light.u), s->light.v),
                vadd(vadd(lc, s->light.u), s->light.v),
                vadd(vsub(lc, s->light.u), s->light.v), lamp);

    s->bg_top = v3(0.28f, 0.48f, 0.85f);
    s->bg_bot = v3(1.70f, 1.35f, 1.00f);
    s->bg_strength = 0.45f;
    s->checker_fade_start = 2500.0f;
    s->checker_fade_end = 9000.0f;
    return 1;
}

/* A material test bench: two rows of spheres covering the BSDF types. */
static int add_materials_scene(scene *s) {
    int floor = s->nmat;
    add_mat(s, (material){M_DIFFUSE, v3(0.86f, 0.84f, 0.80f), v3(0, 0, 0), 0, 1.5f, 260.0f});
    int gold = s->nmat;
    add_mat(s, (material){M_METAL, v3(0.98f, 0.78f, 0.32f), v3(0, 0, 0), 0.02f, 0, 0});
    int copper = s->nmat;
    add_mat(s, (material){M_METAL, v3(0.85f, 0.42f, 0.25f), v3(0, 0, 0), 0.10f, 0, 0});
    int brushed = s->nmat;
    add_mat(s, (material){M_METAL, v3(0.80f, 0.82f, 0.85f), v3(0, 0, 0), 0.28f, 0, 0});
    int chrome = s->nmat;
    add_mat(s, (material){M_METAL, v3(0.93f, 0.95f, 0.98f), v3(0, 0, 0), 0.04f, 0, 0});
    int glass = s->nmat;
    add_mat(s, (material){M_GLASS, v3(0.96f, 0.98f, 1.00f), v3(0, 0, 0), 0, 1.5f, 0});
    int dense = s->nmat;
    add_mat(s, (material){M_GLASS, v3(0.90f, 0.96f, 1.00f), v3(0, 0, 0), 0, 1.9f, 0});
    int clay = s->nmat;
    add_mat(s, (material){M_DIFFUSE, v3(0.72f, 0.16f, 0.13f), v3(0, 0, 0), 0, 1.5f, 0});

    add_quad(s, v3(-1e6f, 0, -1e6f), v3(1e6f, 0, -1e6f),
                v3(1e6f, 0, 1e6f), v3(-1e6f, 0, 1e6f), floor);
    /* Seamless backdrop so the glass has something readable to refract. */
    int wall = s->nmat;
    add_mat(s, (material){M_DIFFUSE, v3(0.78f, 0.77f, 0.74f), v3(0, 0, 0), 0, 1.5f, 0});
    add_quad(s, v3(-1e6f, 0, 1100), v3(1e6f, 0, 1100), v3(1e6f, 8000, 1100), v3(-1e6f, 8000, 1100), wall);

    float xs[4] = {-285, -95, 95, 285};
    int front[4] = {gold, copper, brushed, chrome};
    int blue = s->nmat;
    add_mat(s, (material){M_DIFFUSE, v3(0.16f, 0.34f, 0.62f), v3(0, 0, 0), 0, 1.5f, 0});
    int back[4] = {glass, dense, clay, blue};
    for (int i = 0; i < 4; i++) {
        add_sphere(s, v3(xs[i], 78, 20), 78, front[i]);
        add_sphere(s, v3(xs[i], 78, 330), 78, back[i]);
    }

    s->light.center = v3(0, 900, 150);
    s->light.u = v3(170, 0, 0);
    s->light.v = v3(0, 0, 170);
    s->light.normal = v3(0, -1, 0);
    s->light.emit = v3(28, 26.5f, 24);
    s->light.area = 4.0f * vlen(vcross(s->light.u, s->light.v));
    s->has_light = 1;
    int lamp = s->nmat;
    add_mat(s, (material){M_EMIT, v3(0, 0, 0), s->light.emit, 0, 0, 0});
    vec3 lc = s->light.center;
    add_quad(s, vsub(vsub(lc, s->light.u), s->light.v),
                vsub(vadd(lc, s->light.u), s->light.v),
                vadd(vadd(lc, s->light.u), s->light.v),
                vadd(vsub(lc, s->light.u), s->light.v), lamp);

    s->bg_top = v3(0.30f, 0.38f, 0.55f);
    s->bg_bot = v3(0.70f, 0.72f, 0.75f);
    s->bg_strength = 0.30f;
    s->checker_fade_start = 2500.0f;
    s->checker_fade_end = 9000.0f;
    return 1;
}

/*
 * Furnace test: white diffuse, perfect mirror and clear glass spheres floating
 * in a uniform environment of radiance 1. Energy conservation means the image
 * must be flat 1.0 everywhere, independent of path length. Used to verify that
 * the BSDFs, Russian roulette and environment sampling are all unbiased.
 */
static int add_furnace_scene(scene *s) {
    int white = s->nmat;
    add_mat(s, (material){M_DIFFUSE, v3(1, 1, 1), v3(0, 0, 0), 0, 1.5f, 0});
    int mirror = s->nmat;
    add_mat(s, (material){M_METAL, v3(1, 1, 1), v3(0, 0, 0), 0.0f, 0, 0});
    int glass = s->nmat;
    add_mat(s, (material){M_GLASS, v3(1, 1, 1), v3(0, 0, 0), 0, 1.5f, 0});
    add_sphere(s, v3(0, 0, 0), 1.0f, white);
    add_sphere(s, v3(2.7f, 0, 0), 1.0f, mirror);
    add_sphere(s, v3(-2.7f, 0, 0), 1.0f, glass);
    s->bg_top = v3(1, 1, 1);
    s->bg_bot = v3(1, 1, 1);
    s->bg_strength = 1.0f;
    return 1;
}

int main(int argc, char **argv) {
    const char *scene_name = "cornell";
    const char *out = "render.png";
    int W = 700, H = 700, spp = 200, depth = 10;
    int threads = 12;
    float exposure = 1.0f;
    int exposure_given = 0;
    float orbit_deg = 0.0f;
    int orbit_enabled = 0;
    int stats = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--scene") && i + 1 < argc) scene_name = argv[++i];
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--width") && i + 1 < argc) W = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--height") && i + 1 < argc) H = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--spp") && i + 1 < argc) spp = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--depth") && i + 1 < argc) depth = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--exposure") && i + 1 < argc) { exposure = (float)atof(argv[++i]); exposure_given = 1; }
        else if (!strcmp(argv[i], "--angle") && i + 1 < argc) { orbit_deg = (float)atof(argv[++i]); orbit_enabled = 1; }
        else if (!strcmp(argv[i], "--stats")) stats = 1;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }

    crc_init();

    scene *s = calloc(1, sizeof(scene));
    camera cam;

    if (!strcmp(scene_name, "cornell")) {
        add_default_box(s);
        camera_setup(&cam, v3(278, 278, -820), v3(278, 265, 300), 38.0f, (float)W / H, 0.0f, 0.0f);
        /* The box is open toward the camera. A very dim ambient gives the
         * metal/glass something to reflect instead of a hard black void. */
        s->bg_top = v3(0.10f, 0.10f, 0.11f);
        s->bg_bot = v3(0.10f, 0.10f, 0.11f);
        s->bg_strength = 1.0f;
    } else if (!strcmp(scene_name, "open")) {
        add_open_scene(s);
        if (orbit_enabled) {
            /* Turntable around the scene centre; --angle picks the frame. */
            float a = orbit_deg * 3.14159265358979f / 180.0f;
            vec3 eye = vadd(v3(0, 80, 120), v3(980.0f * sinf(a), 220.0f, -980.0f * cosf(a)));
            camera_setup(&cam, eye, v3(0, 80, 120), 40.0f, (float)W / H, 2.2f, 980.0f);
        } else {
            camera_setup(&cam, v3(-40, 300, -860), v3(0, 80, 120), 40.0f, (float)W / H, 2.2f, 980.0f);
        }
        if (!exposure_given) exposure = 0.50f;
    } else if (!strcmp(scene_name, "materials")) {
        add_materials_scene(s);
        camera_setup(&cam, v3(0, 470, -900), v3(0, 40, 160), 38.0f, (float)W / H, 2.0f, 950.0f);
        if (!exposure_given) exposure = 0.55f;
    } else if (!strcmp(scene_name, "furnace")) {
        add_furnace_scene(s);
        camera_setup(&cam, v3(0, 0, -6), v3(0, 0, 0), 40.0f, (float)W / H, 0.0f, 0.0f);
        if (!exposure_given) exposure = 1.0f;
    } else {
        fprintf(stderr, "unknown scene '%s'\n", scene_name);
        return 2;
    }

    dvec3 *accum = calloc((size_t)W * H, sizeof(dvec3));
    render_ctx ctx = {0};
    ctx.s = s; ctx.cam = &cam; ctx.width = W; ctx.height = H;
    ctx.spp = spp; ctx.max_depth = depth; ctx.exposure = exposure;
    ctx.accum = accum;
    atomic_store(&ctx.next_row, 0);
    pthread_mutex_init(&ctx.lock, NULL);
    ctx.t0 = clock();

    if (threads < 1) threads = 1;
    if (threads > 64) threads = 64;
    pthread_t tid[64];
    for (int i = 0; i < threads; i++) pthread_create(&tid[i], NULL, worker, &ctx);
    for (int i = 0; i < threads; i++) pthread_join(tid[i], NULL);

    double secs = (double)(clock() - ctx.t0) / CLOCKS_PER_SEC;
    fprintf(stderr, "\n  %d x %d, %d spp, %d bounces, %d threads: %.2fs CPU (%.1f Msamples/s)\n",
            W, H, spp, depth, threads, secs,
            (double)W * H * spp / (secs * 1e6));

    uint8_t *rgb = malloc((size_t)W * H * 3);
    double inv = 1.0 / (double)spp;
    if (stats) {
        double sr = 0, sg = 0, sb = 0, qr = 0;
        for (int i = 0; i < W * H; i++) {
            double r = accum[i].r * inv, g = accum[i].g * inv, b = accum[i].b * inv;
            sr += r; sg += g; sb += b; qr += r * r;
        }
        double n = (double)W * H;
        double mr = sr / n;
        double sd = sqrt(fmax(0.0, qr / n - mr * mr));
        fprintf(stderr, "  mean radiance: %.5f %.5f %.5f  (std %.5f)\n",
                mr, sg / n, sb / n, sd);
    }
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            dvec3 a = accum[(size_t)y * W + x];
            vec3 c = v3((float)(a.r * inv * exposure),
                        (float)(a.g * inv * exposure),
                        (float)(a.b * inv * exposure));
            c = aces(c);
            /* sRGB-ish gamma; 2.2 is close enough for a toy. */
            c = v3(powf(c.x, 1.0f / 2.2f), powf(c.y, 1.0f / 2.2f), powf(c.z, 1.0f / 2.2f));
            /* Triangular dither of +/-0.5 LSB: smooth gradients on curved
             * surfaces otherwise band badly at 8 bits. */
            float d = (hash01(x, y, 1) + hash01(x, y, 2) - 1.0f) / 255.0f;
            size_t i = ((size_t)y * W + x) * 3;
            rgb[i + 0] = (uint8_t)lrintf(clampf(c.x + d, 0, 1) * 255.0f);
            rgb[i + 1] = (uint8_t)lrintf(clampf(c.y + d, 0, 1) * 255.0f);
            rgb[i + 2] = (uint8_t)lrintf(clampf(c.z + d, 0, 1) * 255.0f);
        }
    }

    if (!write_png(out, W, H, rgb)) {
        fprintf(stderr, "failed to write %s\n", out);
        return 1;
    }
    fprintf(stderr, "  wrote %s\n", out);

    free(rgb);
    free(accum);
    free(s);
    return 0;
}
