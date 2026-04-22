#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <cmath>
#include <cstddef>

inline const double A = 1.0;
inline const double T = 1.0;
inline const double X = 1.0;
inline const size_t K = 1000;
inline const size_t M = 1000;

inline double phi(double x) { return sin(x); }
inline double psi(double t) { return sin(-A * t); }
inline double f(double t, double x) { return 0.0; }

#endif // CONFIG_HPP
