#ifndef TIMER_H
#define TIMER_H
#include <chrono>

struct timer{

  std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
  std::string name;
  float duration=0.0;
  bool split_flag = false;

  void begin(std::string name = "unnamed"){ 
    this->name = name; start = std::chrono::high_resolution_clock::now();
    duration = 0.0;
    split_flag = false;
  }
  void split(){
    if (!split_flag) {
      stop = std::chrono::high_resolution_clock::now();
      auto lduration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
      duration += (float)lduration.count();
      split_flag = true;
    } else {
      split_flag = false;
      start = std::chrono::high_resolution_clock::now();
    }
  }
  float end(){
		float dt = end_silent();
    std::cout << "Time taken by " << name << " is " << dt  << " seconds\n";
		return dt;
  }
	float end_silent(){
		stop = std::chrono::high_resolution_clock::now();
    auto lduration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    duration += (float)lduration.count();
		return duration/1.0e6;
	}
};

#endif
