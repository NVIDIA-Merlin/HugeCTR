#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <vector>

#include "argparse.hpp"
#include "common.hpp"
#include "data_readers/async_reader/async_reader.hpp"
#include "resource_manager.hpp"

using namespace HugeCTR;

std::vector<int> str_to_vec(const std::string& str) {
  std::istringstream iss(str);
  std::vector<std::string> tokens{std::istream_iterator<std::string>{iss},
                                  std::istream_iterator<std::string>{}};
  std::vector<int> res;
  for (auto& s : tokens) {
    res.push_back(std::stoi(s));
  }
  return res;
}

int main(int argc, char** argv) {
  argparse::ArgumentParser args("read_upload_bench");

  args.add_argument("--num_dense").default_value(13).action([](const std::string& value) {
    return std::stoi(value);
  });

  args.add_argument("--num_categorical").default_value(26).action([](const std::string& value) {
    return std::stoi(value);
  });

  args.add_argument("--batch_size").default_value(65536).action([](const std::string& value) {
    return std::stoi(value);
  });

  args.add_argument("--gpus")
      .default_value(std::string("0"))
      .help("Space-delimited list of GPUs to upload the data onto");

  args.add_argument("--num_threads").default_value(1).action([](const std::string& value) {
    return std::stoi(value);
  });

  args.add_argument("--num_batches_per_thread")
      .default_value(1)
      .action([](const std::string& value) { return std::stoi(value); });

  args.add_argument("--io_block_size").default_value(524288).action([](const std::string& value) {
    return std::stoi(value);
  });

  args.add_argument("--io_depth").default_value(2).action([](const std::string& value) {
    return std::stoi(value);
  });

  args.add_argument("--io_alignment").default_value(512).action([](const std::string& value) {
    return std::stoi(value);
  });

  args.add_argument("file").remaining();

  try {
    args.parse_args(argc, argv);
  } catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << args;
    exit(1);
  }

  std::string fname;
  try {
    fname = args.get<std::string>("file");
  } catch (std::logic_error& e) {
    std::cout << "No input file provided" << std::endl;
    exit(1);
  }

  const int sample_dim = args.get<int>("--num_dense") + args.get<int>("--num_categorical") + 1;
  const int batch_size_bytes = args.get<int>("--batch_size") * sample_dim * sizeof(int);

#ifdef ENABLE_MPI
  CK_MPI_THROW_(MPI_Init(&argc, &argv));
#endif
  CK_NVML_THROW_(nvmlInit_v2());

  std::vector<std::vector<int>> vvgpu;
  vvgpu.push_back(str_to_vec(args.get<std::string>("--gpus")));
  const auto resource_manager = ResourceManager::create(vvgpu, 424242);

  AsyncReaderImpl reader_impl(
      fname, batch_size_bytes, resource_manager.get(), args.get<int>("--num_threads"),
      args.get<int>("--num_batches_per_thread"), args.get<int>("--io_block_size"),
      args.get<int>("--io_depth"), args.get<int>("--io_alignment"));

  printf("Initialization done, starting to read...\n");
  fflush(stdout);
  auto start = std::chrono::high_resolution_clock::now();

  reader_impl.load_async();

  size_t sz = 1;
  while (sz > 0) {
    BatchDesc desc = reader_impl.get_batch();
    sz = desc.size_bytes;
    // usleep(200);
    reader_impl.finalize_batch();
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  printf("Reading took %.3fs, B/W %.2f GB/s\n", elapsed.count() / 1000.0,
         std::filesystem::file_size(fname) / ((double)elapsed.count() * 1e6));

#ifdef ENABLE_MPI
  CK_MPI_THROW_(MPI_Finalize());
#endif

  return 0;
}
