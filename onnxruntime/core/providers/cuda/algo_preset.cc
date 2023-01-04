#include "core/providers/cuda/algo_preset.h"
#include <fstream>
#include <sstream>

namespace onnxruntime{

AlgoPreset AlgoPreset::instance_;

void AlgoPreset::Load(const std::string& filename) {
  std::ifstream in(filename);
  std::string kernel_name;
  int algo;
  while (in.good()) {
    in >> kernel_name >> algo;
    conv_algo_map[kernel_name] = algo;
  }
}

AlgoPreset& AlgoPreset::Instance() {
  return instance_;
}

int AlgoPreset::GetAlgo(const std::string& kernel_name) const {
  auto it = conv_algo_map.find(kernel_name);
  if (it != conv_algo_map.end()) {
    return it->second;
  } else {
    return 1; // IMPLICIT_PRECOMP_GEMM
  }
}
}
