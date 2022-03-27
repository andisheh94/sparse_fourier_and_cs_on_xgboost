//===================================================
// Integration benchmarking from the C/C++ interface
//===================================================


#include "python_utils.h"
#include "swht_kernel.h"
#include "benchmark_utils.h"
#include "build_info.h"
#include "random_signal.h"
#include "timing.h"
#ifdef PROFILING_BUILD
    #include "profiling.h"
    PROFILING_INIT
#endif

#include <chrono>
#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <unordered_map>


int main(int argc, char *argv[]) {

    // Run options
    int c;
    bool resume_mode = false;
    bool robust_mode = false;
    while ((c = getopt(argc, argv, "ar")) != -1) {
        switch (c) {
        case 'a':
            resume_mode = true;
            break;
        case 'r':
            robust_mode = true;
            break;
        default:
            throw std::runtime_error(std::string("Unrecognized option: ") + std::to_string(c));
            break;
        }
    }
    int opt_diff = argc - optind;
    if (opt_diff != 4)
        throw std::runtime_error("Requires 4 arguments (" + std::to_string(opt_diff) + " given).");
    std::string cs_algo(argv[optind]);
    std::stringstream str_n(argv[optind + 1]);
    std::stringstream str_K(argv[optind + 2]);
    std::stringstream str_d(argv[optind + 3]);
    int cs_algo_num = -1;
    try {
        cs_algo_num = cs_algos_index.at(cs_algo);
    } catch (const std::out_of_range &) {
        std::cerr << "Unrecognized CS algorithm: " << cs_algo << '.' << std::endl;
        return 1;
    }

    // Ready Python interpreter
    std::string project_root = PROJECT_ROOT;
    Py_Initialize();

    // Set runs parameters
    typedef std::chrono::high_resolution_clock timer;
    unsigned n_measurements = 10;
    unsigned long n, K, degree;
    str_n >> n;
    str_K >> K;
    str_d >> degree;
    double        C            = 1.3;
    double        ratio        = 1.4;
    unsigned      robust_iters = 3;
    unsigned long cs_bins      = degree * degree;
    unsigned long cs_iters     = 1;
    double        cs_ratio     = 2.0;

    // Ready file output
    std::stringstream filename_builder;
#ifndef PROFILING_BUILD
    filename_builder << project_root << "/benchmarking/results/raw_"
#else
    filename_builder << project_root << "/benchmarking/profiles/raw_"
#endif
        << (robust_mode ? "robust" : "basic") << '_' << cs_algo << '_' << n
        << '_' << K << '_' << degree << ".csv";
    std::string filename = filename_builder.str();
    csv_writer writer(filename, resume_mode);
    if (!resume_mode) {
#ifndef PROFILING_BUILD
        writer.write_line("time", "error", "algorithm", "n", "K", "C", "ratio",
            "degree", "robust_iters", "cs_bins", "cs_iters", "cs_ratio");
#else
        writer.write_line("time", "section", "samples", "algorithm", "n", "K", "C",
            "ratio", "degree", "robust_iters", "cs_bins", "cs_iters", "cs_ratio");
#endif
    }
    
    // Ready run set
    unsigned run = writer.line_number;
    unsigned number_of_errors = 0;

    // Ready signal
    RandomSignal signal(n, K, degree);

    // Warm-up round
    auto warmup_start = timer::now();
    {frequency_map out;
        if (robust_mode) {
            switch (cs_algo_num) {
            case 0:
                swht_robust<0>(&signal, out, n, K, C, ratio, robust_iters);
                break;
            case 1:
                swht_robust<1>(&signal, out, n, K, C, ratio, robust_iters, cs_bins, cs_iters, cs_ratio);
                break;
            case 2:
                swht_robust<2>(&signal, out, n, K, C, ratio, robust_iters, degree);
                break;
            }
        } else {
            switch (cs_algo_num) {
            case 0:
                swht_basic<0>(&signal, out, n, K, C, ratio);
                break;
            case 1:
                swht_basic<1>(&signal, out, n, K, C, ratio, cs_bins, cs_iters, cs_ratio);
                break;
            case 2:
                swht_basic<2>(&signal, out, n, K, C, ratio, degree);
                break;
            }
        }
#ifdef PROFILING_BUILD
        PROFILING_RESET
#endif
    }
    auto warmup_end = timer::now();
    
    // Time and write runs
    auto experiment_start = timer::now();
    for (; run < n_measurements; run++) {

        // Ready signal cache
        signal.ready_fast_calls();

        // Call and time function
        frequency_map out;
        timestamp start;
        uint64_t duration;
        if (robust_mode) {
            switch (cs_algo_num) {
            case 0:
                start_stamp(start)
                swht_robust<0>(&signal, out, n, K, C, ratio, robust_iters);
                end_stamp(duration, start)
                break;
            case 1:
                start_stamp(start)
                swht_robust<1>(&signal, out, n, K, C, ratio, robust_iters, cs_bins, cs_iters, cs_ratio);
                end_stamp(duration, start)
                break;
            case 2:
                start_stamp(start)
                swht_robust<2>(&signal, out, n, K, C, ratio, robust_iters, degree);
                end_stamp(duration, start)
                break;
            }
        } else {
            switch (cs_algo_num) {
            case 0:
                start_stamp(start)
                swht_basic<0>(&signal, out, n, K, C, ratio);
                end_stamp(duration, start)
                break;
            case 1:
                start_stamp(start)
                swht_basic<1>(&signal, out, n, K, C, ratio, cs_bins, cs_iters, cs_ratio);
                end_stamp(duration, start)
                break;
            case 2:
                start_stamp(start)
                swht_basic<2>(&signal, out, n, K, C, ratio, degree);
                end_stamp(duration, start)
                break;
            }
        }

        // Check for failure
        int failure = signal == out;
        number_of_errors += failure;

        // Output result
#ifndef PROFILING_BUILD
        writer.write_line(duration, failure, cs_algos_names[cs_algo_num], n, K,
            C, ratio, degree, robust_iters, cs_bins, cs_iters, cs_ratio);
#else
        (void) duration;
        iterate_sections(
            [&](std::string section, uint64_t cycles, uint64_t samples) {
                writer.write_partial_line(cycles, section,
                    samples, cs_algos_names[cs_algo_num], n, K, C, ratio,
                    degree, robust_iters, cs_bins, cs_iters, cs_ratio);
            }
        );
        writer.inc_line();
        PROFILING_RESET
#endif
    }

    // Cleanup python interpreter
    Py_FinalizeEx();
    
    // Experiments general insight
    auto experiment_end = timer::now();
    std::chrono::duration<double> warmup_duration = warmup_end - warmup_start;
    std::chrono::duration<double> experiment_duration = experiment_end - experiment_start;
    std::cout << "Warm-up took: " << warmup_duration.count() << " [s]\n"
        << "Experiments took: " << experiment_duration.count() << " [s]\n"
        << "Caused " << number_of_errors << " failures" << std::endl;

    return 0;
}
