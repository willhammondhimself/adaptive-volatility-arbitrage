/**
 * Heston FFT Option Pricer
 *
 * Implementation of Heston (1993) stochastic volatility model
 * with Carr-Madan (1999) FFT pricing method.
 *
 * References:
 *   - Heston (1993): "A Closed-Form Solution for Options with Stochastic Volatility"
 *   - Carr & Madan (1999): "Option Valuation using the Fast Fourier Transform"
 *   - Lord & Kahl (2006): "Optimal Fourier Inversion in Semi-Analytical Option Pricing"
 */

#ifndef HESTON_FFT_HPP
#define HESTON_FFT_HPP

#include <complex>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace heston {

// Heston model parameters
struct HestonParams {
    double v0;       // Initial variance
    double theta;    // Long-run variance
    double kappa;    // Mean reversion speed
    double sigma_v;  // Volatility of volatility
    double rho;      // Correlation between spot and variance
    double r;        // Risk-free rate
    double q;        // Dividend yield

    // Validate Feller condition: 2*kappa*theta > sigma_v^2
    bool feller_satisfied() const {
        return 2.0 * kappa * theta > sigma_v * sigma_v;
    }
};

// FFT configuration
struct FFTConfig {
    int N = 4096;           // FFT grid size (power of 2)
    double alpha = 1.5;     // Damping factor for Carr-Madan
    double eta = 0.25;      // Grid spacing in log-strike

    double lambda() const { return 2.0 * M_PI / (N * eta); }
    double b() const { return M_PI / eta; }
};


/**
 * Heston characteristic function (log-price under risk-neutral measure).
 *
 * Uses the "little Heston trap" formulation from Albrecher et al. (2007)
 * for numerical stability.
 */
inline std::complex<double> heston_cf(
    std::complex<double> u,
    double S, double T,
    const HestonParams& p
) {
    const std::complex<double> i(0.0, 1.0);

    // Log of forward price
    double F = S * std::exp((p.r - p.q) * T);

    // Heston parameters for CF
    std::complex<double> xi = p.kappa - p.rho * p.sigma_v * i * u;
    std::complex<double> d = std::sqrt(
        xi * xi + p.sigma_v * p.sigma_v * (i * u + u * u)
    );

    // "Little Heston trap" formulation for stability
    std::complex<double> g = (xi - d) / (xi + d);
    std::complex<double> exp_dT = std::exp(-d * T);

    std::complex<double> D = (xi - d) / (p.sigma_v * p.sigma_v) *
        (1.0 - exp_dT) / (1.0 - g * exp_dT);

    std::complex<double> C = p.kappa * (
        (xi - d) * T - 2.0 * std::log((1.0 - g * exp_dT) / (1.0 - g))
    ) / (p.sigma_v * p.sigma_v);

    // Characteristic function of log(S_T)
    return std::exp(i * u * std::log(F) + C * p.theta + D * p.v0);
}


/**
 * Carr-Madan integrand for FFT.
 *
 * Transforms the characteristic function for efficient FFT evaluation.
 */
inline std::complex<double> carr_madan_integrand(
    double v, double S, double K, double T,
    const HestonParams& params,
    double alpha
) {
    const std::complex<double> i(0.0, 1.0);
    std::complex<double> u(v, -(alpha + 1.0));

    std::complex<double> cf = heston_cf(u, S, T, params);
    std::complex<double> denom = alpha * alpha + alpha - v * v + i * (2.0 * alpha + 1.0) * v;

    return std::exp(-params.r * T) * cf / denom;
}


/**
 * In-place Cooley-Tukey FFT.
 *
 * Simple radix-2 implementation - not optimized, but sufficient for demo.
 * For production, link against FFTW.
 */
inline void fft_inplace(std::vector<std::complex<double>>& x) {
    int N = x.size();
    if (N <= 1) return;

    // Bit-reversal permutation
    for (int i = 1, j = 0; i < N; ++i) {
        int bit = N >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }

    // Cooley-Tukey iterative FFT
    for (int len = 2; len <= N; len <<= 1) {
        double theta = -2.0 * M_PI / len;
        std::complex<double> wlen(std::cos(theta), std::sin(theta));
        for (int i = 0; i < N; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                std::complex<double> u = x[i + j];
                std::complex<double> t = w * x[i + j + len / 2];
                x[i + j] = u + t;
                x[i + j + len / 2] = u - t;
                w *= wlen;
            }
        }
    }
}


/**
 * Price European call option using Heston FFT.
 *
 * Implements the Carr-Madan (1999) algorithm with proper grid construction.
 */
inline double price_call(
    double S, double K, double T,
    const HestonParams& params,
    const FFTConfig& config = FFTConfig()
) {
    const int N = config.N;
    const double eta = config.eta;
    const double alpha = config.alpha;
    const double b = config.b();
    const double lambda = config.lambda();

    // Build FFT input with Simpson weights
    std::vector<std::complex<double>> x(N);
    const std::complex<double> i(0.0, 1.0);

    for (int j = 0; j < N; ++j) {
        double v = j * eta;

        // Simpson's rule weights
        double simpson = (j == 0) ? 1.0 / 3.0 :
                        ((j % 2 == 0) ? 2.0 / 3.0 : 4.0 / 3.0);

        // Carr-Madan integrand with correct phase
        std::complex<double> integrand = carr_madan_integrand(v, S, K, T, params, alpha);
        x[j] = std::exp(i * b * v) * integrand * eta * simpson;
    }

    // FFT
    fft_inplace(x);

    // Find price at desired log-strike
    double log_K = std::log(K);
    int idx = static_cast<int>((log_K + b) / lambda);

    if (idx < 0 || idx >= N) {
        throw std::out_of_range("Strike outside FFT grid");
    }

    // Extract call price with damping reversal
    double call = std::exp(-alpha * log_K) * x[idx].real() / M_PI;
    return std::max(0.0, call);
}


/**
 * Price European put using put-call parity.
 */
inline double price_put(
    double S, double K, double T,
    const HestonParams& params,
    const FFTConfig& config = FFTConfig()
) {
    double call = price_call(S, K, T, params, config);
    double forward = S * std::exp((params.r - params.q) * T);
    double discount = std::exp(-params.r * T);
    return call - discount * (forward - K);
}


/**
 * Price option (call or put).
 */
inline double price(
    double S, double K, double T,
    const HestonParams& params,
    bool is_call = true,
    const FFTConfig& config = FFTConfig()
) {
    return is_call ? price_call(S, K, T, params, config)
                   : price_put(S, K, T, params, config);
}


/**
 * Price multiple strikes at once (vectorized).
 */
inline std::vector<double> price_strikes(
    double S,
    const std::vector<double>& strikes,
    double T,
    const HestonParams& params,
    bool is_call = true,
    const FFTConfig& config = FFTConfig()
) {
    std::vector<double> prices;
    prices.reserve(strikes.size());

    for (double K : strikes) {
        prices.push_back(price(S, K, T, params, is_call, config));
    }
    return prices;
}

}  // namespace heston

#endif  // HESTON_FFT_HPP
