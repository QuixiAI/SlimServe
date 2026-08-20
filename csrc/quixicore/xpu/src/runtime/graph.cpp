#include "quixicore/xpu/graph.hpp"

#include <stdexcept>
#include <utility>

#include <sycl/ext/oneapi/experimental/graph.hpp>

namespace quixicore::xpu {
namespace graph = sycl::ext::oneapi::experimental;

using ModifiableGraph = graph::command_graph<graph::graph_state::modifiable>;
using ExecutableGraph = graph::command_graph<graph::graph_state::executable>;

struct CommandGraph::Impl {
  explicit Impl(sycl::queue capture_queue)
      : queue(std::move(capture_queue)), modifiable(std::make_unique<ModifiableGraph>(queue)) {}

  sycl::queue queue;
  std::unique_ptr<ModifiableGraph> modifiable;
  std::unique_ptr<ExecutableGraph> executable;
  bool is_recording = false;
};

bool command_graph_supported(const sycl::device &device) noexcept {
  return device.has(sycl::aspect::ext_oneapi_graph);
}

CommandGraph::CommandGraph(sycl::queue queue) : impl_(std::make_unique<Impl>(std::move(queue))) {}

CommandGraph::~CommandGraph() = default;
CommandGraph::CommandGraph(CommandGraph &&) noexcept = default;
CommandGraph &CommandGraph::operator=(CommandGraph &&) noexcept = default;

void CommandGraph::capture_begin() {
  if (!command_graph_supported(impl_->queue.get_device())) {
    throw std::runtime_error("QuixiCore XPU: device does not support oneAPI command graphs");
  }
  if (impl_->is_recording) {
    throw std::logic_error("QuixiCore XPU: graph capture already active");
  }
  if (impl_->executable) {
    throw std::logic_error("QuixiCore XPU: reset graph before starting another capture");
  }
  impl_->queue.wait();
  impl_->modifiable->begin_recording(impl_->queue);
  impl_->is_recording = true;
}

void CommandGraph::capture_end() {
  if (!impl_->is_recording) {
    throw std::logic_error("QuixiCore XPU: graph capture is not active");
  }
  impl_->modifiable->end_recording(impl_->queue);
  impl_->is_recording = false;
  impl_->executable = std::make_unique<ExecutableGraph>(impl_->modifiable->finalize());
}

sycl::event CommandGraph::replay() {
  if (!impl_->executable) {
    throw std::logic_error("QuixiCore XPU: graph has not been finalized");
  }
  return impl_->queue.ext_oneapi_graph(*impl_->executable);
}

void CommandGraph::synchronize() {
  impl_->queue.wait();
}

void CommandGraph::reset() {
  if (impl_->is_recording) {
    throw std::logic_error("QuixiCore XPU: cannot reset an active capture");
  }
  impl_->queue.wait();
  impl_->executable.reset();
  impl_->modifiable = std::make_unique<ModifiableGraph>(impl_->queue);
}

bool CommandGraph::recording() const noexcept {
  return impl_->is_recording;
}
bool CommandGraph::ready() const noexcept {
  return static_cast<bool>(impl_->executable);
}
sycl::queue &CommandGraph::queue() noexcept {
  return impl_->queue;
}

} // namespace quixicore::xpu
