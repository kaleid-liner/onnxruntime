#pragma once
#include <string>
#include <unordered_map>

namespace onnxruntime {

class AlgoPreset {
public:
  void Load(const std::string& filename);

  static AlgoPreset& Instance();

  int GetAlgo(const std::string& kernel_name) const;

private:
  static AlgoPreset instance_;

  std::unordered_map<std::string, int> conv_algo_map;
};
} // namespace onnxruntime