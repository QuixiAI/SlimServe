#pragma once

#include <memory>

#include <sycl/sycl.hpp>

namespace quixicore::xpu {

// Whether the device supports the oneAPI command-graph extension.
bool command_graph_supported(const sycl::device &device) noexcept;

// Current-queue-only SYCL command graph. Capture and reset never perform a
// device-wide queue walk, which avoids the transient-queue lifetime failure
// seen in framework graph wrappers on the B60 runtime stack.
class CommandGraph {
public:
  explicit CommandGraph(sycl::queue queue);
  ~CommandGraph();

  CommandGraph(CommandGraph &&) noexcept;
  CommandGraph &operator=(CommandGraph &&) noexcept;
  CommandGraph(const CommandGraph &) = delete;
  CommandGraph &operator=(const CommandGraph &) = delete;

  void capture_begin();
  void capture_end();
  sycl::event replay();
  void synchronize();
  void reset();

  bool recording() const noexcept;
  bool ready() const noexcept;
  sycl::queue &queue() noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace quixicore::xpu
