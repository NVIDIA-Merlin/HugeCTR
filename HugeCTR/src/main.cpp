/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <fstream>
#include <iomanip>
#include <iostream>
#include "HugeCTR/include/optimizer.hpp"
#include "HugeCTR/include/parser.hpp"
#include "HugeCTR/include/session.hpp"
#include "HugeCTR/include/utils.hpp"

#ifdef ENABLE_MPI
#include <mpi.h>

#define CK_MPI_THROW__(cmd)                                                                        \
  do {                                                                                             \
    auto retval = (cmd);                                                                           \
    if (retval != MPI_SUCCESS) {                                                                   \
      throw std::runtime_error(std::string("MPI Runtime error: ") + std::to_string(retval) + " " + \
                               __FILE__ + ":" + std::to_string(__LINE__) + " \n");                 \
    }                                                                                              \
  } while (0)

#endif

static const std::string simple_help =
    "usage: huge_ctr.exe [--train] [--help] [--version] config_file.json\n";

enum class CmdOptions_t { Train, Version, Help };

HugeCTR::Timer timer_log;

void train(std::string config_file) {
  int numprocs = 1, pid = 0;
#ifdef ENABLE_MPI
  CK_MPI_THROW__(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
  CK_MPI_THROW__(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
#endif

  const HugeCTR::SolverParser solver_config(config_file);
  std::shared_ptr<HugeCTR::Session> session_instance = HugeCTR::Session::Create(solver_config);
  std::unique_ptr<HugeCTR::LearningRateScheduler> lr_sch =
      HugeCTR::get_learning_rate_scheduler(config_file);

  HugeCTR::LOG(timer_log.elapsedMilliseconds(), "init_end");
  HugeCTR::Timer timer;
  timer.start();
  HugeCTR::LOG(timer_log.elapsedMilliseconds(), "run_start");

  HugeCTR::Timer timer_train;
  HugeCTR::Timer timer_eval;
  timer_train.start();
  session_instance->start_data_reading();
#ifdef DATA_READING_TEST
  HugeCTR::Timer timer_data_reading;
  timer_data_reading.start();
#endif

  // train
  if (pid == 0) {
    std::cout << "HugeCTR training start:" << std::endl;
  }
#ifndef VAL
  HugeCTR::LOG(timer_log.elapsedMilliseconds(), "train_epoch_start", 0);  // just 1 epoch

  for (int i = 0; i < solver_config.max_iter; i++) {
    float lr = lr_sch->get_next();
    session_instance->set_learning_rate(lr);

    session_instance->train();
    if (i % solver_config.display == 0 && i != 0) {
      timer_train.stop();
      // display
      float loss = 0;
      session_instance->get_current_loss(&loss);
      if (pid == 0) {
        MESSAGE_("Iter: " + std::to_string(i) + " Time(" + std::to_string(solver_config.display) +
                 " iters): " + std::to_string(timer_train.elapsedSeconds()) +
                 "s Loss: " + std::to_string(loss) + " lr:" + std::to_string(lr));
      }
      timer_train.start();
    }
    if (i % solver_config.snapshot == 0 && i != 0) {
      // snapshot
      session_instance->download_params_to_files(solver_config.snapshot_prefix, i);
    }

    if ((solver_config.eval_interval > 0 && i % solver_config.eval_interval == 0 && i != 0)) {
      HugeCTR::LOG(timer_log.elapsedMilliseconds(), "eval_start",
                   float(i) / solver_config.max_iter);
      timer_eval.start();
      for (int j = 0; j < solver_config.eval_batches; ++j) {
        session_instance->eval();
      }

      auto eval_metrics = session_instance->get_eval_metrics();
      for (auto& eval_metric : eval_metrics) {
        MESSAGE_("Evaluation, " + eval_metric.first + ": " + std::to_string(eval_metric.second));

        HugeCTR::LOG(timer_log.elapsedMilliseconds(), "eval_accuracy", eval_metric.second, float(i) / solver_config.max_iter, i);

        // early stop doesn't support multinodes
        if (!eval_metric.first.compare("AUC")) {
          const auto auc_threshold = const_cast<HugeCTR::SolverParser&>(solver_config).metrics_spec[HugeCTR::metrics::Type::AUC];
          if (eval_metric.second >= auc_threshold) {
            timer.stop();
            size_t train_samples =
                static_cast<size_t>(i + 1) * static_cast<size_t>(solver_config.batchsize);
            HugeCTR::LOG(timer_log.elapsedMilliseconds(), "train_samples", train_samples);

            std::string epoch_num_str = std::to_string(float(i) / solver_config.max_iter);

            std::cout << "Hit target accuracy AUC " + std::to_string(auc_threshold) +
                         " at epoch " + epoch_num_str + " with batchsize: "
                      << solver_config.batchsize << " in " << std::setiosflags(std::ios::fixed)
                      << std::setprecision(2) << timer.elapsedSeconds() << " s. Average speed "
                      << float(i) * solver_config.batchsize / timer.elapsedSeconds()
                      << " records/s." << std::endl;

            HugeCTR::LOG(timer_log.elapsedMilliseconds(), "eval_stop", epoch_num_str);

            HugeCTR::LOG(timer_log.elapsedMilliseconds(), "train_epoch_end", 1);

            HugeCTR::LOG(timer_log.elapsedMilliseconds(), "run_stop");
            timer_log.stop();
            return;
          }
        }
      }

      timer_eval.stop();

      MESSAGE_("Eval Time for " + std::to_string(solver_config.eval_batches) +
               " iters: " + std::to_string(timer_eval.elapsedSeconds()) + "s");

      HugeCTR::LOG(
          timer_log.elapsedMilliseconds(), "eval_stop",
          float(i) / solver_config.max_iter);  // use iteration to calculate it's in which epoch
    }
  }

#ifdef DATA_READING_TEST
  timer_data_reading.stop();
  std::cout << "Overall time: " << timer_data_reading.elapsedSeconds() << std::endl;
#endif

  HugeCTR::LOG(timer_log.elapsedMilliseconds(), "train_epoch_end", 1);

  HugeCTR::LOG(timer_log.elapsedMilliseconds(), "run_stop");
  timer_log.stop();

#else
  float loss = 0;
  bool start_test = false;
  int loop = 0;
  for (int i = 0; i < solver_config.max_iter; i++) {
    float lr = lr_sch->get_next();
    session_instance->set_learning_rate(lr(i));

    session_instance->train();
    if (start_test == true) {
      float loss_tmp = 0;
      session_instance->get_current_loss(&loss_tmp);
      loss += loss_tmp;
    }
    if (i % solver_config.eval_interval == solver_config.eval_batches &&
        i != solver_config.eval_batches) {
      loss = loss / solver_config.eval_batches;
      for (int j = 0; j < solver_config.eval_batches; ++j) {
        session_instance->eval();
      }
      if (pid == 0) {
        std::cout << loop << " ";
        auto eval_metrics = session_instance->get_eval_metrics();
        for (auto& eval_metric : eval_metrics) {
          std::cout << eval_metric.second << " ";
        }
        std::cout << std::endl;
      }
      start_test = false;
    }
    if (i != 0 && i % solver_config.eval_interval == 0) {
      start_test = true;
      loss = 0;
      loop = i;
    }
  }
#endif
  return;
}

int main(int argc, char* argv[]) {
  try {
    timer_log.start();
    HugeCTR::LOG(timer_log.elapsedMilliseconds(), "init_start");
    int numprocs = 1, pid = 0;
    cudaSetDevice(0);
#ifdef ENABLE_MPI
    int provided;
    CK_MPI_THROW__(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));
    CK_MPI_THROW__(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
    CK_MPI_THROW__(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
#endif
    const std::map<std::string, CmdOptions_t> CMD_OPTIONS_TYPE_MAP = {
        {"--train", CmdOptions_t::Train},
        {"--help", CmdOptions_t::Help},
        {"--version", CmdOptions_t::Version}};

    if (argc != 3 && argc != 2 && pid == 0) {
      std::cout << simple_help;
      return -1;
    }

    CmdOptions_t opt = CmdOptions_t::Help;
    if (!HugeCTR::find_item_in_map(opt, std::string(argv[1]), CMD_OPTIONS_TYPE_MAP) && pid == 0) {
      std::cerr << "wrong option: " << argv[1] << std::endl;
      std::cerr << simple_help;
      return -1;
    }

    switch (opt) {
      case CmdOptions_t::Help: {
        if (pid == 0) {
          std::cout << simple_help;
        }
        break;
      }
      case CmdOptions_t::Version: {
        if (pid == 0) {
          std::cout << "HugeCTR Version: " << HUGECTR_VERSION_MAJOR << "." << HUGECTR_VERSION_MINOR
                    << std::endl;
        }
        break;
      }
      case CmdOptions_t::Train: {
        if (pid == 0) {
          std::cout << "HugeCTR Version: " << HUGECTR_VERSION_MAJOR << "." << HUGECTR_VERSION_MINOR
                    << std::endl;
        }

        if (argc != 3 && pid == 0) {
          std::cerr << "expect config file." << std::endl;
          std::cerr << simple_help;
          return -1;
        }

        std::string config_file(argv[2]);
        if (pid == 0) {
          std::cout << "Config file: " << config_file << std::endl;
        }
        std::thread train_thread(train, config_file);
        HugeCTR::set_affinity(train_thread, {}, true);

        train_thread.join();
        break;
      }
      default: { assert(!"Error: no such option && should never get here!"); }
    }
#ifdef ENABLE_MPI
    CK_MPI_THROW__(MPI_Finalize());
#endif
  } catch (const std::runtime_error& rt_err) {
    std::cerr << rt_err.what() << std::endl;
    std::cerr << "Terminated with error\n";
  }

  return 0;
}
