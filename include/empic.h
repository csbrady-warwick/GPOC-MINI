#ifndef PARTICLESIM_H
#define PARTICLESIM_H

#include <iostream>
#include <cstring>
//CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"
#include "field.h"
#include "particle.h"
#include "axes.h"

#define C 1
#define EPSILON0 1

template<typename T, int rank>
__device__ inline void get2DIndex(deviceField<T,rank> * field, int64_t id, int64_t &index_x, int64_t &index_y){
    index_y = id/(field->dims[0]);
    index_x = id - index_y*field->dims[0];

    index_y+=1-field->nghosts;
    index_x+=1-field->nghosts;
}

template<typename T, int rank>
__global__ void depositDensity(deviceParticleSet<T,rank> * particles, deviceField<T,rank> * density, deviceAxis<T,rank> * axes){
    int64_t idx = blockIdx.x*blockDim.x + threadIdx.x+1;
    if (idx <= particles->size){
        particle<T,rank> &p = (*particles)(idx);
        int64_t index_x = (p.x[0] - axes->xb(1))/axes->dx()+1;
        int64_t index_y = (p.x[1] - axes->yb(1))/axes->dy()+1;

        TYPE frac_x = (p.x[0] - axes->xb(index_x))/axes->dx();
        TYPE frac_y = (p.x[1] - axes->yb(index_y))/axes->dy();


        TYPE data = p.weight;
        TYPE w00 = (1-frac_x)*(1-frac_y) * data;
        TYPE w01 = frac_x*(1-frac_y) * data;
        TYPE w10 = (1-frac_x)*frac_y * data;
        TYPE w11 = frac_x*frac_y *data;

        atomicAdd(&density->data[density->subscript(index_x,index_y)], w00);
        atomicAdd(&density->data[(*density).subscript(index_x+1,index_y)], w01);
        atomicAdd(&density->data[(*density).subscript(index_x,index_y+1)], w10);
        atomicAdd(&density->data[(*density).subscript(index_x+1,index_y+1)], w11);
     
    }
}

template<typename T, int rank>
__global__ void BCzeroGradient(deviceField<T,rank> * field, deviceAxis<T,rank> * axes){
    int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < field->size){
        int64_t index_x, index_y;
        get2DIndex(field, idx, index_x, index_y);
        if (index_x<1) field->data[idx] = field->data[field->subscript(1,index_y)];
        if (index_x>field->nx()) field->data[idx] = field->data[field->subscript(field->nx(),index_y)];
        if (index_y<1) field->data[idx] = field->data[field->subscript(index_x,1)];
        if (index_y>field->ny()) field->data[idx] = field->data[field->subscript(index_x,field->ny())];
    }
}

template<typename T, int rank>
__global__ void BCPeriodic(deviceField<T,rank> * field, deviceAxis<T,rank> * axes){
    int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < field->size){
        int64_t index_x, index_y;
        get2DIndex(field, idx, index_x, index_y);
        if (index_x<1) field->data[idx] = field->data[field->subscript(field->nx()+index_x,index_y)];
        if (index_x>field->nx()) field->data[idx] = field->data[field->subscript(0+(index_x-field->nx()),index_y)];
        if (index_y<1) field->data[idx] = field->data[field->subscript(index_x,field->ny()+index_y)];
        if (index_y>field->ny()) field->data[idx] = field->data[field->subscript(index_x,0+(index_y-field->ny()))];
    }
}

template<typename T, int rank>
__global__ void BCClampValue(deviceField<T,rank> * field, T value=0.0){
    int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < field->size){
        int64_t index_x, index_y;
        get2DIndex(field, idx, index_x, index_y);
        if (index_x<1) field->data[idx] = value;
        if (index_x>field->nx()) field->data[idx] = value;
        if (index_y<1) field->data[idx] = value;
        if (index_y>field->ny()) field->data[idx] = value;
    }
}

template<typename T, int rank>
__global__ void BCDriveLeft(deviceField<T,rank> * field, deviceAxis<T,rank> * axes, T time, T dt, T amplitude, T omega){
    int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < field->size){
        int64_t index_x, index_y;
        get2DIndex(field, idx, index_x, index_y);
        if (index_x==1) {
            field->data[field->subscript(index_x,index_y)] = amplitude*sin(omega*time) * exp(-axes->yb(index_y)*axes->yb(index_y)/0.0125);
        }
    }
}

template<typename T, int rank>
__global__ void setField(deviceField<T,rank> * field, T value=0){
    int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < field->size){
        field->data[idx] = 0;
    }
}

template<typename T, int rank>
__global__ void pushParticles(deviceParticleSet<T,rank> * particles, 
deviceField<T,rank> * Exp, deviceField<T,rank> * Eyp, deviceField<T,rank> * Ezp,
deviceField<T,rank> *Bxp, deviceField<T,rank> *Byp, deviceField<T,rank> *Bzp, 
deviceField<T,rank> *Jxp, deviceField<T,rank> *Jyp, deviceField<T,rank> *Jzp, 
deviceAxis<T,rank> * axes, T dt, T time){
    int64_t idx = blockIdx.x*blockDim.x + threadIdx.x+1;
    if (idx<= particles->size){

        deviceField<T,rank>  &Ex = *Exp; deviceField<T,rank>  &Ey = *Eyp; deviceField<T,rank>  &Ez = *Ezp;
        deviceField<T,rank>  &Bx = *Bxp; deviceField<T,rank>  &By = *Byp; deviceField<T,rank>  &Bz = *Bzp;
        deviceField<T,rank>  &Jx = *Jxp; deviceField<T,rank>  &Jy = *Jyp; deviceField<T,rank>  &Jz = *Jzp;

        T dtco2 = dt/(T(2.0)*C);
        T part_mc = C * particles->mass;
        T ipart_mc = T(1.0)/part_mc;
        T part_q = particles->charge;
        T cm_ratio = particles->charge/particles->mass;
        T ccmratio = C * cm_ratio;

        particle<T,rank> &p = (*particles)(idx);

        T part_weight = p.weight;

        T part_ux = p.p[0]/part_mc;
        T part_uy = p.p[1]/part_mc;
        T part_uz = p.p[2]/part_mc;

        T gamma_rel = std::sqrt(T(1.0) + part_ux*part_ux + part_uy*part_uy + part_uz*part_uz);
        T root = dtco2/gamma_rel;

        T part_x = p.x[0] - axes->xb(1);
        T part_y = p.x[1] - axes->yb(1);

        part_x += part_ux*root;
        part_y += part_uy*root;

        //Get the cell that the particle is in
        T cell_x_r = part_x/axes->dx();
        T cell_y_r = part_y/axes->dy();

        int64_t cell_x1 = (int64_t)cell_x_r;
        int64_t cell_y1 = (int64_t)cell_y_r;
        T cell_frac_x = T(cell_x1) - cell_x_r;
        T cell_frac_y = T(cell_y1) - cell_y_r;
        cell_x1 += 1;
        cell_y1 += 1;

        T gx[4]={}, gy[4]={};

        gx[1] = T(0.5)+cell_frac_x;
        gx[2] = T(0.5)-cell_frac_x;
        gy[1] = T(0.5)+cell_frac_y;
        gy[2] = T(0.5)-cell_frac_y;

        int64_t cell_x2 = int64_t(cell_x_r-T(0.5));
        int64_t cell_y2 = int64_t(cell_y_r-T(0.5));
        cell_frac_x = T(cell_x2) - cell_x_r;
        cell_frac_y = T(cell_y2) - cell_y_r;
        cell_x2 += 1;
        cell_y2 += 1;

        //Extra elements of hx, hy are for current deposition
        T hx[4], hy[4];
        hx[1] = T(0.5)+cell_frac_x;
        hx[2] = T(0.5)-cell_frac_x;
        hy[1] = T(0.5)+cell_frac_y;
        hy[2] = T(0.5)-cell_frac_y;

#ifndef NOFIELD

        T ex_part = gy[1]*(hx[1]*Ex(cell_x2, cell_y1)
            + hx[2]*Ex(cell_x2+1, cell_y1))
            + gy[2]*(hx[1]*Ex(cell_x2, cell_y1+1)
            + hx[2]*Ex(cell_x2+1, cell_y1+1));

        T ey_part = hy[1]*(gx[1] * Ey(cell_x1,cell_y2)
            + gx[2] * Ey(cell_x1+1, cell_y2))
            + hy[2]*(gx[1]*Ey(cell_x1, cell_y2+1)
            + gx[2]*Ey(cell_x1+1, cell_y2+1));

        T ez_part = gy[1]*(gx[1]*Ez(cell_x1, cell_y1)
            + gx[2]*Ez(cell_x1+1, cell_y1))
            + gy[2]*(gx[1]*Ez(cell_x1, cell_y1+1)
            + gx[2]*Ez(cell_x1+1, cell_y1+1));

        T bx_part = hy[1]*(gx[1]*Bx(cell_x1, cell_y2)
            + gx[2]*Bx(cell_x1+1, cell_y2))
            + hy[2]*(gx[1]*Bx(cell_x1, cell_y2+1)
            + gx[2]*Bx(cell_x1+1, cell_y2+1));

        T by_part = gy[1]*(hx[1]*By(cell_x2, cell_y1)
            + hx[2]*By(cell_x2+1, cell_y1))
            + gy[2]*(hx[1]*By(cell_x2, cell_y1+1)
            + hx[2]*By(cell_x2+1, cell_y1+1));

        T bz_part = hy[1]*(hx[1]*Bz(cell_x2, cell_y2)
            + hx[2]*Bz(cell_x2+1, cell_y2))
            + hy[2]*(hx[1]*Bz(cell_x2, cell_y2+1)
            + hx[2]*Bz(cell_x2+1, cell_y2+1));
#else

        T ex_part = 0.0; T ey_part = 0.0; T ez_part = 0.0;
        T bx_part = 0.0; T by_part = 0.0; T bz_part = 0.0;
#endif

        T uxm = part_ux + cm_ratio * ex_part;
        T uym = part_uy + cm_ratio * ey_part;
        T uzm = part_uz + cm_ratio * ez_part;

        gamma_rel = std::sqrt(T(1.0) + uxm*uxm + uym*uym + uzm*uzm);
        root = ccmratio/gamma_rel;

        T taux = bx_part*root; T tauy = by_part*root; T tauz = bz_part*root;
        T taux2 = taux * taux; T tauy2 = tauy * tauy; T tauz2 = tauz * tauz;

        T tau = T(1.0)/(1.0 + taux2 + tauy2 + tauz2);

        T uxp = ((T(1.0)+taux2 -tauy2 -tauz2)*uxm
                + T(2.0) * ((taux*tauy+tauz)*uym
                + (taux * tauz + tauy)*uzm))*tau;

        T uyp = ((T(1.0)-taux2+tauy2-tauz2)*uym
                +T(2.0)*((tauy*tauz+taux)*uzm
                +(tauy*taux-tauz)*uxm))*tau;

        T uzp = ((T(1.0)-taux2-tauy2+tauz2)*uzm
                + T(2.0)*((taux*tauz+tauy)*uxm
                +(tauz*tauy - taux)*uym))*tau;

        part_ux = uxp + cm_ratio * ex_part;
        part_uy = uyp + cm_ratio * ey_part;
        part_uz = uzp + cm_ratio * ez_part;

        T part_u2 = part_ux*part_ux + part_uy*part_uy + part_uz*part_uz;
        gamma_rel = std::sqrt(T(1.0) + part_u2);
        T igamma = T(1.0)/gamma_rel;
        root = dtco2*igamma;

        T deltax = part_ux*root;
        T deltay = part_uy*root;
        T part_vz = part_uz * C * igamma;

        part_x+=deltax;
        part_y+=deltay;

        p.x[0] = part_x + axes->xb(1);
        p.x[1] = part_y + axes->yb(1);

        p.p[0] = part_ux*part_mc;
        p.p[1] = part_uy*part_mc;
        p.p[2] = part_uz*part_mc;
        //Particle push finished. Now current deposition

        //Current deposition. Esirkepov method
        part_x += deltax;
        part_y += deltay;

        cell_x_r = (part_x)/axes->dx();
        cell_y_r = (part_y)/axes->dy();

        T cell_x3 = (int64_t)cell_x_r;
        T cell_y3 = (int64_t)cell_y_r;

        cell_frac_x = T(cell_x3) - cell_x_r;
        cell_frac_y = T(cell_y3) - cell_y_r;

        cell_x3 += 1;
        cell_y3 += 1;

        int64_t dcellx = cell_x3 - cell_x1 + 1;
        int64_t dcelly = cell_y3 - cell_y1 + 1;

        for (int i=0;i<3;++i){
            hx[i] = 0;
            hy[i] = 0;
        }

        hx[dcellx] = T(0.5)+cell_frac_x;
        hx[dcellx+1] = T(0.5)-cell_frac_x;
        hy[dcelly] = T(0.5)+cell_frac_y;
        hy[dcelly+1] = T(0.5)-cell_frac_y;

        for (int i=0;i<3;++i){
            hx[i] = hx[i]-gx[i];
            hy[i] = hy[i]-gy[i];
        }

        int64_t xmin = (dcellx-1)/2;
        int64_t xmax = 1 + (dcellx+1)/2;

        int64_t ymin = (dcelly-1)/2;
        int64_t ymax = 1 + (dcelly+1)/2;

        T fcx = part_weight/(dt * axes->dx());
        T fcy = part_weight/(dt * axes->dy());
        T fcz = part_weight/(axes->dx() * axes->dy());

        T fjx = fcx * part_q;
        T fjy = fcy * part_q;
        T fjz = fcz * part_q * part_vz;

        T jyh = -0.0;
        for (int iy=ymin;iy<=ymax;++iy){
            int64_t cy = cell_y1 + iy-1;
            T yfac1 = gy[iy] + T(0.5) * hy[iy];
            T yfac2 = hy[iy]/T(3.0) + T(0.5) * gy[iy];

            T jxh = 0.0;
            for (int ix=xmin;ix<=xmax;++ix){
                int64_t cx = cell_x1 + ix-1;
                T xfac1 = gx[ix] + T(0.5) * hx[ix];

                T wx = hx[ix] * yfac1;
                T wy = hy[iy] * xfac1;
                T wz = gx[ix] * yfac1 +hx[ix] * yfac2;

                jxh -= fjx * wx;
                jyh -= fjy * wy;
                T jzh = fjz * wz;

                atomicAdd(&Jx(cx,cy), jxh);
                atomicAdd(&Jy(cx,cy), -jyh);
                atomicAdd(&Jz(cx,cy), jzh);
            }
        }


        //Reflecting boundaries
        if (p.x[0] < axes->xb(1)){
            p.x[0] = axes->xb(1);
            p.p[0] = -p.p[0];
        }
        if (p.x[0] > axes->xb(axes->nx()+1)){
            p.x[0] = axes->xb(axes->nx()+1);
            p.p[0] = -p.p[0];
        }
        if (p.x[1] < axes->yb(1)){
            T dax = axes->yb(1)-p.x[1];
            p.x[1] = axes->yb(1)+dax;
            p.p[1] = -p.p[1];
        }
        if (p.x[1] > axes->yb(axes->ny()+1)){
            T dax = p.x[1] - axes->yb(axes->ny()+1);
            p.x[1] = axes->yb(axes->ny()+1)-dax;
            p.p[1] = -p.p[1];
        }
    }
}

template<typename T, int rank>
__global__ void updateE(deviceField<T,rank> * Exp, deviceField<T, rank> *Eyp, deviceField<T,rank> *Ezp,
    deviceField<T,rank> *Bxp, deviceField<T,rank> *Byp, deviceField<T,rank> *Bzp,
    deviceField<T,rank> *Jxp, deviceField<T,rank> *Jyp, deviceField<T,rank> *Jzp, deviceAxis<T,rank> *axes, 
    T dt, T time)
{
    int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx< Exp->size){
        int64_t index_x, index_y;
        get2DIndex(Exp, idx, index_x, index_y);
        //Just the domain
        if (index_x < 1 || index_x > Exp->nx()+1 || index_y < 1 || index_y > Exp->ny()+1) return;
        deviceField<T,rank> &Ex = *Exp; deviceField<T,rank> &Ey = *Eyp; deviceField<T,rank> &Ez = *Ezp;
        deviceField<T,rank> &Bx = *Bxp; deviceField<T,rank> &By = *Byp; deviceField<T,rank> &Bz = *Bzp;
        deviceField<T,rank> &Jx = *Jxp; deviceField<T,rank> &Jy = *Jyp; deviceField<T,rank> &Jz = *Jzp;
        T hdt = dt/T(2.0);
        T hdtx = hdt/axes->dx();
        T hdty = hdt/axes->dy();
        T cnx = hdtx * C * C;
        T cny = hdty * C * C;
        T fac = hdt/EPSILON0;
        Ex(index_x, index_y) += cny * (Bz(index_x, index_y) - Bz(index_x, index_y-1))
            - fac * Jx(index_x, index_y);
        Ey(index_x, index_y) -= cnx * (Bz(index_x, index_y) - Bz(index_x-1, index_y))
            - fac * Jy(index_x, index_y);
        Ez(index_x, index_y) += cnx * (By(index_x, index_y) - By(index_x-1, index_y))
            - cny * (Bx(index_x, index_y) - Bx(index_x, index_y-1))
            - fac * Jz(index_x, index_y);
    }
}

template<typename T, int rank>
__global__ void updateB(deviceField<T,rank> * Exp, deviceField<T, rank> *Eyp, deviceField<T,rank> *Ezp,
    deviceField<T,rank> *Bxp, deviceField<T,rank> *Byp, deviceField<T,rank> *Bzp,
    deviceAxis<T,rank> *axes, T dt, T time)
{
    int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx< Bxp->size){
        int64_t index_x, index_y;
        get2DIndex(Bxp, idx, index_x, index_y);
        //Just the domain
        if (index_x < 1 || index_x > Bxp->nx()+1 || index_y < 1 || index_y > Bxp->ny()+1) return;
        deviceField<T,rank> &Ex = *Exp; deviceField<T,rank> &Ey = *Eyp; deviceField<T,rank> &Ez = *Ezp;
        deviceField<T,rank> &Bx = *Bxp; deviceField<T,rank> &By = *Byp; deviceField<T,rank> &Bz = *Bzp;

        T hdt = dt/T(2.0);
        T hdtx = hdt/axes->dx();
        T hdty = hdt/axes->dy();
        T cnx = hdtx * C * C;
        T cny = hdty * C * C;
        T fac = hdt/EPSILON0;

        Bx(index_x, index_y) -= cny * (Ez(index_x, index_y+1) - Ez(index_x, index_y));
        By(index_x, index_y) += cnx * (Ez(index_x+1, index_y) - Ez(index_x, index_y));
        Bz(index_x, index_y) -= cnx * (Ey(index_x+1, index_y) - Ey(index_x, index_y))
            - cny * (Ex(index_x, index_y+1) - Ex(index_x, index_y));
    }
}

#endif
