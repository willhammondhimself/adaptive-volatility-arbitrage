/**
 * pybind11 bindings for Heston FFT pricer.
 *
 * Build with: pip install .
 * Usage:
 *   import heston_cpp
 *   price = heston_cpp.price_call(S=100, K=100, T=1.0, v0=0.04, ...)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "heston_fft.hpp"

namespace py = pybind11;

PYBIND11_MODULE(heston_cpp, m) {
    m.doc() = R"pbdoc(
        Heston FFT Option Pricer (C++ implementation)
        ---------------------------------------------

        High-performance option pricing using:
        - Heston (1993) stochastic volatility model
        - Carr-Madan (1999) FFT method

        Approximately 30-50x faster than pure Python/NumPy.
    )pbdoc";

    // Expose HestonParams struct
    py::class_<heston::HestonParams>(m, "HestonParams",
        "Heston model parameters")
        .def(py::init<>())
        .def(py::init([](double v0, double theta, double kappa,
                        double sigma_v, double rho, double r, double q) {
            heston::HestonParams p;
            p.v0 = v0;
            p.theta = theta;
            p.kappa = kappa;
            p.sigma_v = sigma_v;
            p.rho = rho;
            p.r = r;
            p.q = q;
            return p;
        }), py::arg("v0"), py::arg("theta"), py::arg("kappa"),
           py::arg("sigma_v"), py::arg("rho"), py::arg("r"), py::arg("q") = 0.0)
        .def_readwrite("v0", &heston::HestonParams::v0)
        .def_readwrite("theta", &heston::HestonParams::theta)
        .def_readwrite("kappa", &heston::HestonParams::kappa)
        .def_readwrite("sigma_v", &heston::HestonParams::sigma_v)
        .def_readwrite("rho", &heston::HestonParams::rho)
        .def_readwrite("r", &heston::HestonParams::r)
        .def_readwrite("q", &heston::HestonParams::q)
        .def("feller_satisfied", &heston::HestonParams::feller_satisfied,
            "Check if Feller condition (2*kappa*theta > sigma_v^2) is satisfied");

    // Expose FFTConfig struct
    py::class_<heston::FFTConfig>(m, "FFTConfig",
        "FFT grid configuration")
        .def(py::init<>())
        .def(py::init([](int N, double alpha, double eta) {
            heston::FFTConfig c;
            c.N = N;
            c.alpha = alpha;
            c.eta = eta;
            return c;
        }), py::arg("N") = 4096, py::arg("alpha") = 1.5, py::arg("eta") = 0.25)
        .def_readwrite("N", &heston::FFTConfig::N)
        .def_readwrite("alpha", &heston::FFTConfig::alpha)
        .def_readwrite("eta", &heston::FFTConfig::eta);

    // Price call option
    m.def("price_call", &heston::price_call,
        R"pbdoc(
            Price European call option using Heston FFT.

            Args:
                S: Spot price
                K: Strike price
                T: Time to maturity (years)
                params: HestonParams object
                config: FFTConfig object (optional)

            Returns:
                Call option price
        )pbdoc",
        py::arg("S"), py::arg("K"), py::arg("T"),
        py::arg("params"), py::arg("config") = heston::FFTConfig());

    // Price put option
    m.def("price_put", &heston::price_put,
        R"pbdoc(
            Price European put option using Heston FFT + put-call parity.

            Args:
                S: Spot price
                K: Strike price
                T: Time to maturity (years)
                params: HestonParams object
                config: FFTConfig object (optional)

            Returns:
                Put option price
        )pbdoc",
        py::arg("S"), py::arg("K"), py::arg("T"),
        py::arg("params"), py::arg("config") = heston::FFTConfig());

    // Price option (call or put)
    m.def("price", &heston::price,
        R"pbdoc(
            Price European option using Heston FFT.

            Args:
                S: Spot price
                K: Strike price
                T: Time to maturity (years)
                params: HestonParams object
                is_call: True for call, False for put
                config: FFTConfig object (optional)

            Returns:
                Option price
        )pbdoc",
        py::arg("S"), py::arg("K"), py::arg("T"),
        py::arg("params"), py::arg("is_call") = true,
        py::arg("config") = heston::FFTConfig());

    // Price multiple strikes
    m.def("price_strikes", &heston::price_strikes,
        R"pbdoc(
            Price options for multiple strikes at once.

            Args:
                S: Spot price
                strikes: List of strike prices
                T: Time to maturity (years)
                params: HestonParams object
                is_call: True for calls, False for puts
                config: FFTConfig object (optional)

            Returns:
                List of option prices
        )pbdoc",
        py::arg("S"), py::arg("strikes"), py::arg("T"),
        py::arg("params"), py::arg("is_call") = true,
        py::arg("config") = heston::FFTConfig());

    // Convenience function with all parameters inline
    m.def("price_option",
        [](double S, double K, double T,
           double v0, double theta, double kappa,
           double sigma_v, double rho, double r, double q,
           bool is_call) {
            heston::HestonParams params;
            params.v0 = v0;
            params.theta = theta;
            params.kappa = kappa;
            params.sigma_v = sigma_v;
            params.rho = rho;
            params.r = r;
            params.q = q;
            return heston::price(S, K, T, params, is_call);
        },
        R"pbdoc(
            Price option with all parameters inline (convenience function).

            Args:
                S: Spot price
                K: Strike price
                T: Time to maturity (years)
                v0: Initial variance
                theta: Long-run variance
                kappa: Mean reversion speed
                sigma_v: Vol of vol
                rho: Spot-variance correlation
                r: Risk-free rate
                q: Dividend yield
                is_call: True for call, False for put

            Returns:
                Option price
        )pbdoc",
        py::arg("S"), py::arg("K"), py::arg("T"),
        py::arg("v0"), py::arg("theta"), py::arg("kappa"),
        py::arg("sigma_v"), py::arg("rho"),
        py::arg("r") = 0.05, py::arg("q") = 0.0,
        py::arg("is_call") = true);

    // Version info
    m.attr("__version__") = "0.1.0";
}
