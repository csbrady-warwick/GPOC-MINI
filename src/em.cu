#include <thrust/sort.h>
#include <random>
#include <fstream>
#include <string>

#define TYPE double
#define TPB 512

#define NPART 10000000ll
#define NX 2000ll
#define NY 2000ll
#define NITS 2000ll
#define NOUT 20
#define NGHOSTS 4
#define NSORT 100
#define NTOT (NX+2ll*NGHOSTS)*(NY+2ll*NGHOSTS)

//#define NOTIMESORT
//#define NOSORT
//#define NOINITIALSORT
//#define NODRIVE

#define USE_MOMENTUM
#include "empic.h"
#undef USE_MOMENTUM
#include "timer.h"

template<typename T, int rank>
struct part_comparator{
	__host__ __device__ bool operator()(const particle<T,rank> &a, const particle<T,rank> &b){
		if (a.x[0] < b.x[0]) return true;
		if (a.x[0] > b.x[0]) return false;
		return a.x[1] < b.x[1];
	}
};


void writePPM(std::string prefix, int index, field<> &f){
	TYPE *dP = f.get();
	TYPE mx = 0.0;
	TYPE mn = 1.0e10;
	TYPE total = 0.0;
	int imx = 0;
	int imy = 0;
	for (int j=1-NGHOSTS; j<=NY+NGHOSTS; j++){
		for (int i=1-NGHOSTS; i<=NX+NGHOSTS; i++){
			total += dP[f.subscript(i,j)];
			if (dP[f.subscript(i,j)] > mx) {
				mx = dP[f.subscript(i,j)];
				imx = i;
				imy = j;
			}
			if (dP[f.subscript(i,j)] < mn) mn = dP[f.subscript(i,j)];
		}
	}
	//mn=-0.0005;
	//mx=0.0005;
	std::cout << "Max: " << mx << " at " << imx << " " << imy << std::endl;
	std::cout << "Min: " << mn << std::endl;

	//std::cout << "Total: " << total<< std::endl;
#define MAXLEN 8
	char sindex[MAXLEN+1]={};
	snprintf(sindex, MAXLEN+1, "%08d", index);
	std::ofstream file(prefix + sindex + ".ppm", std::ios::binary);
	//Binary PPM
	file << "P6\n" << NX << " " << NY << "\n(TPB-1)\n";
	//Reverse the y axis since images start from top left
	for (int j=NY;j>=1;j--){
		for (int i=1;i<=NX;i++){
			TYPE data = dP[f.subscript(i,j)];
			char c = (char)(255.0*(data-mn)/(mx-mn));
			//std::cout << f(i,j) << " ";
			file.write(&c, 1);
			file.write(&c, 1);
			file.write(&c, 1);
		}
		//std::cout << std::endl;
	}
	//std::cout << std::endl;
	file.close();
}

void writePPMLog(std::string prefix, int index, field<> &f){
	TYPE *dP = f.get();
	TYPE mx = 0.0;
	TYPE mn = 1.0e10;
	TYPE total = 0.0;
	for (int j=1-NGHOSTS; j<=NY+NGHOSTS; j++){
		for (int i=1-NGHOSTS; i<=NX+NGHOSTS; i++){
			total += dP[f.subscript(i,j)];
			if (dP[f.subscript(i,j)] > mx) {
				mx = dP[f.subscript(i,j)];
			}
			if (dP[f.subscript(i,j)] < mn) mn = dP[f.subscript(i,j)];
		}
	}
	mn=std::log(std::max(mn,(TYPE)1.0e-3*mx));
	mx=std::log(mx);
	//std::cout << "Max: " << mx << " at " << imx << " " << imy << std::endl;
	//std::cout << "Min: " << mn << std::endl;
	//std::cout << "Total: " << total<< std::endl;
#define MAXLEN 8
	char sindex[MAXLEN+1]={};
	snprintf(sindex, MAXLEN+1, "%08d", index);
	std::ofstream file(prefix + sindex + ".ppm", std::ios::binary);
	//Binary PPM
	file << "P6\n" << NX << " " << NY << "\n(TPB-1)\n";
	//Reverse the y axis since images start from top left
	for (int j=NY;j>=1;j--){
		for (int i=1;i<=NX;i++){
			TYPE data = std::log(dP[f.subscript(i,j)]+1e-10);
			char c = (char)(255.0*(data-mn)/(mx-mn));
			//std::cout << f(i,j) << " ";
			file.write(&c, 1);
			file.write(&c, 1);
			file.write(&c, 1);
		}
		//std::cout << std::endl;
	}
	//std::cout << std::endl;
	file.close();
}

template<typename T, int rank>
inline void get2DIndex(field<T,rank> * f, int64_t id, int64_t &index_x, int64_t &index_y){
	index_y = id/(f->dims[0]);
	index_x = id - index_y*f->dims[0];

	index_y+=1-f->nghosts;
	index_x+=1-f->nghosts;
}

int main(){

	std::cout <<"\n\n";
	std::cout <<"        d########P  d########b        .######b          d#######  d##P      d##P" << "\n";
	std::cout <<"       d########P  d###########    d###########     .##########  d##P      d##P" << "\n";
	std::cout <<"      ----        ----     ----  -----     ----   -----         ----      -- P" << "\n";
	std::cout <<"     d########P  d####,,,####P ####.      .#### d###P          d############P   "<< "\n";
	std::cout <<"    d########P  d#########P   ####       .###P ####.          d############P    "<< "\n";
	std::cout <<"   d##P        d##P           ####     d####   ####.         d##P      d##P     "<< "\n";
	std::cout <<"  d########P  d##P            ###########P     ##########P  d##P      d##P      "<< "\n";
	std::cout <<" d########P  d##P              d######P          #######P  d##P      d##P       "<<"\n\n\n";
	field<> Ex(NGHOSTS,NX,NY), Ey(NGHOSTS,NX,NY), Ez(NGHOSTS,NX,NY);
	field<> Bx(NGHOSTS,NX,NY), By(NGHOSTS,NX,NY), Bz(NGHOSTS,NX,NY);
	field<> Jx(NGHOSTS,NX,NY), Jy(NGHOSTS,NX,NY), Jz(NGHOSTS,NX,NY);
	field<> density(NGHOSTS,NX,NY);
	axis<> axes(NGHOSTS,NX,NY);
	axes.buildX(-1,1);
	axes.buildY(-1,1);

	particleSet<> particles(NPART);

	particle<> *dparticles = particles.get();
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> disxc(0,axes.dx());
	std::uniform_real_distribution<> disyc(0,axes.dy());
	std::uniform_real_distribution<> disx(-0.8, 0.8);
	std::uniform_real_distribution<> disy(-0.8,0.8);
	std::normal_distribution<> temp(0.0,1.0e-5);

	for (int i=0; i<NPART; i++){
		dparticles[i].x[0] = disx(gen);
		dparticles[i].x[1] = disy(gen);
		dparticles[i].p[0] = temp(gen);
		dparticles[i].p[1] = temp(gen);
		dparticles[i].weight = TYPE(NPART)/(axes.dx()*axes.dy());
	}
	int64_t filled_parts = 0;
	for (int j=1;j<NY+1;j++){
		if (axes.yb(j) < -0.8) continue;
		if (axes.yb(j) > 0.8) continue;
		for (int i=1;i<NX+1;i++){
			if (axes.xb(i) < -0.8) continue;
			if (axes.xb(i) > 0.8) continue;
			filled_parts++;
		}
	}
	particles.put(dparticles, NPART);
#ifndef NOINITIALSORT
	thrust::sort(thrust::device, particles.d_data, particles.d_data+NPART-1ll, part_comparator<TYPE,2>());
#endif


	timer t;
	std::cout << "Starting simulation" << std::endl;
	TYPE dt=0.8*axes.dx()/(C*sqrt(2.0));
	TYPE time=0.0;
	TYPE * ExP = Ex.get();
	for (int j=1;j<NY+1;j++){
		for (int i=1;i<NX+1;i++){
			ExP[Ex.subscript(i,j)] = 0.0;
		}
	}
	Ex.put(ExP);

	BCClampValue<<<(NTOT+(TPB-1))/TPB,TPB>>>(Bx.onDevice());
	BCClampValue<<<(NTOT+(TPB-1))/TPB,TPB>>>(By.onDevice());
	BCClampValue<<<(NTOT+(TPB-1))/TPB,TPB>>>(Bz.onDevice());
	BCClampValue<<<(NTOT+(TPB-1))/TPB,TPB>>>(Ex.onDevice());
	BCClampValue<<<(NTOT+(TPB-1))/TPB,TPB>>>(Ey.onDevice());
	BCClampValue<<<(NTOT+(TPB-1))/TPB,TPB>>>(Ez.onDevice());
	cudaDeviceSynchronize();


	//Jy=0.0001;
	t.begin("Move");
	for (int i=0; i<NITS; i++){
		updateE<<<(NTOT+(TPB-1))/TPB,TPB>>>(Ex.onDevice(), Ey.onDevice(), Ez.onDevice(), Bx.onDevice(), By.onDevice(), Bz.onDevice(), Jx.onDevice(), Jy.onDevice(), Jz.onDevice(), axes.onDevice(), dt,time);
		cudaDeviceSynchronize();
		BCPeriodic<<<(NTOT+(TPB-1))/TPB,TPB>>>(Ex.onDevice(), axes.onDevice());
		BCPeriodic<<<(NTOT+(TPB-1))/TPB,TPB>>>(Ey.onDevice(), axes.onDevice());
		BCPeriodic<<<(NTOT+(TPB-1))/TPB,TPB>>>(Ez.onDevice(), axes.onDevice());
		cudaDeviceSynchronize();
		updateB<<<(NTOT+(TPB-1))/TPB,TPB>>>(Ex.onDevice(), Ey.onDevice(), Ez.onDevice(), Bx.onDevice(), By.onDevice(), Bz.onDevice(), axes.onDevice(), dt, time);
		cudaDeviceSynchronize();
		BCPeriodic<<<(NTOT+(TPB-1))/TPB,TPB>>>(Bx.onDevice(),axes.onDevice());
		BCPeriodic<<<(NTOT+(TPB-1))/TPB,TPB>>>(By.onDevice(),axes.onDevice());
		BCPeriodic<<<(NTOT+(TPB-1))/TPB,TPB>>>(Bz.onDevice(),axes.onDevice());
		cudaDeviceSynchronize();

		setField<TYPE,2><<<(NTOT+(TPB-1))/TPB,TPB>>>(Jx.onDevice(), 0.0);
		setField<TYPE,2><<<(NTOT+(TPB-1))/TPB,TPB>>>(Jy.onDevice(), 0.0);
		setField<TYPE,2><<<(NTOT+(TPB-1))/TPB,TPB>>>(Jz.onDevice(), 0.0);
		cudaDeviceSynchronize();

		pushParticles<<<(NPART+(TPB-1))/TPB,TPB>>>(particles.onDevice(), 
				Ex.onDevice(), Ey.onDevice(), Ez.onDevice(), 
				Bx.onDevice(), By.onDevice(), Bz.onDevice(), 
				Jx.onDevice(), Jy.onDevice(), Jz.onDevice(),
				axes.onDevice(), dt, time);
		cudaDeviceSynchronize();

		updateB<<<(NTOT+(TPB-1))/TPB,TPB>>>(Ex.onDevice(), Ey.onDevice(), Ez.onDevice(), Bx.onDevice(), By.onDevice(), Bz.onDevice(), axes.onDevice(), dt, time);
		cudaDeviceSynchronize();
		BCPeriodic<<<(NTOT+(TPB-1))/TPB,TPB>>>(Bx.onDevice(), axes.onDevice());
		BCPeriodic<<<(NTOT+(TPB-1))/TPB,TPB>>>(By.onDevice(), axes.onDevice());
		BCPeriodic<<<(NTOT+(TPB-1))/TPB,TPB>>>(Bz.onDevice(), axes.onDevice());
		cudaDeviceSynchronize();
#ifndef NODRIVE
		TYPE amp = 1e10;
		TYPE lambda = 1.0e-1;
		TYPE omega = 2.0*M_PI*C/lambda;
		BCDriveLeft<<<(NTOT+(TPB-1))/TPB,TPB>>>(Bz.onDevice(), axes.onDevice(), time, dt, amp, omega); 
		cudaDeviceSynchronize();
#endif
		updateE<<<(NTOT+(TPB-1))/TPB,TPB>>>(Ex.onDevice(), Ey.onDevice(), Ez.onDevice(), Bx.onDevice(), By.onDevice(), Bz.onDevice(), Jx.onDevice(), Jy.onDevice(), Jz.onDevice(), axes.onDevice(), dt,time);
		BCPeriodic<<<(NTOT+(TPB-1))/TPB,TPB>>>(Ex.onDevice(), axes.onDevice());
		BCPeriodic<<<(NTOT+(TPB-1))/TPB,TPB>>>(Ey.onDevice(), axes.onDevice());
		BCPeriodic<<<(NTOT+(TPB-1))/TPB,TPB>>>(Ez.onDevice(), axes.onDevice());
		cudaDeviceSynchronize();
#ifndef NOSORT
		if (i%NSORT==0){
#ifndef NOTIMESORT
			t.split();
#endif
			thrust::sort(thrust::device, particles.d_data, particles.d_data+NPART-1ll, part_comparator<TYPE,2>());
#ifndef NOTIMESORT
			t.split();
#endif
		}
#endif
#ifdef WITHIO
		if (i%NOUT == 0){
			std::cout << "Output " << output_id << " of " << NITS/NOUT << "(" << 100.0*TYPE(output_id)/double(NITS/NOUT) << "%)" << std::endl;
			output_id++;
			depositDensity<<<(NPART+(TPB-1))/TPB,TPB>>>(particles.onDevice(), density.onDevice(), axes.onDevice());
			cudaDeviceSynchronize();
			writePPM("output", i/NOUT, density);
			setField<<<(NTOT+(TPB-1))/TPB,TPB>>>(density.onDevice(), 0.0);
			cudaDeviceSynchronize();
		}
#endif
		time+=dt;
		//std::cout << "Time: " << time << std::endl;
	}
	t.end();

}
