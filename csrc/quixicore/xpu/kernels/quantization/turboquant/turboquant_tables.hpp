#pragma once

// Host-only Lloyd-Max centroid tables for the TurboQuant key path
// (specs/formats/turboquant.md). After the sign flip + FWHT + 1/sqrt(d)
// rotation each coordinate of a unit-ish vector is approximately N(0, 1/d),
// so the optimal scalar quantizer solves the Lloyd-Max conditions for that
// Gaussian. Centroid identity is required cache metadata: encoder and decoder
// must use the same table, so callers generate it here (deterministically)
// and pass it to both.
//
// Translated from the TurboQuant reference solver (Zandieh et al.,
// arXiv:2504.19874; vLLM turboquant/centroids.py) — trapezoidal integration,
// 200 iterations, tolerance 1e-10, support ±3.5σ.

#include <cmath>
#include <cstddef>
#include <vector>

namespace quixicore::xpu::turboquant {

inline std::vector<float> lloyd_max_centroids(std::size_t d, int bits) {
  const int n_levels = 1 << bits;
  const double sigma2 = 1.0 / static_cast<double>(d);
  const double sigma = std::sqrt(sigma2);

  const auto pdf = [sigma2](double x) {
    return (1.0 / std::sqrt(2.0 * M_PI * sigma2)) *
           std::exp(-x * x / (2.0 * sigma2));
  };
  const auto trapz = [](const auto& f, double a, double b) {
    constexpr int n = 200;
    const double h = (b - a) / n;
    double result = 0.5 * (f(a) + f(b));
    for (int i = 1; i < n; ++i) result += f(a + i * h);
    return result * h;
  };

  const double lo = -3.5 * sigma, hi = 3.5 * sigma;
  std::vector<double> centroids(n_levels);
  for (int i = 0; i < n_levels; ++i)
    centroids[i] = lo + (hi - lo) * (i + 0.5) / n_levels;

  for (int iter = 0; iter < 200; ++iter) {
    std::vector<double> edges(n_levels + 1);
    edges[0] = lo * 3.0;
    for (int i = 0; i + 1 < n_levels; ++i)
      edges[i + 1] = 0.5 * (centroids[i] + centroids[i + 1]);
    edges[n_levels] = hi * 3.0;
    double delta = 0.0;
    std::vector<double> next(n_levels);
    for (int i = 0; i < n_levels; ++i) {
      const double num =
          trapz([&](double x) { return x * pdf(x); }, edges[i], edges[i + 1]);
      const double den = trapz(pdf, edges[i], edges[i + 1]);
      next[i] = den > 1e-15 ? num / den : centroids[i];
      delta = std::max(delta, std::abs(next[i] - centroids[i]));
    }
    centroids = std::move(next);
    if (delta < 1e-10) break;
  }

  std::vector<float> out(centroids.begin(), centroids.end());
  return out;
}

}  // namespace quixicore::xpu::turboquant
