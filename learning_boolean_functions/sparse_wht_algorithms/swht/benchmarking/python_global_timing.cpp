//====================================================
// Integration benchmarking from the Python interface
//====================================================


#include "python_utils.h"
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
    std::string cs_algo_name = cs_algos_names[cs_algo_num];

    // Ready Python interpreter
    std::string project_root = PROJECT_ROOT;
    std::vector<std::string> paths = {
        project_root + std::string("/build/src/python_module")
    };
    if (start_python(paths)) {
        std::cerr << "Could not properly start Python interpreter and path." << std::endl;
        return 1;
    }

    // Load python swht
    PyObject *swht_function = load_class("swht", "swht");
    if (swht_function == NULL) {
        std::cerr << "Failed to load swht function." << std::endl;
        return 1;
    }

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
    filename_builder << project_root << "/benchmarking/results/py_"
#else
    filename_builder << project_root << "/benchmarking/profiles/py_"
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

    // Ready runs arguments
    PyObject *vargs = Py_BuildValue("(Oskk)", &signal, cs_algo_name.c_str(), n, K);
    PyObject *kwargs = Py_BuildValue("{s:d,s:d,s:k,s:k,s:k,s:d,s:k}",
        "C", C,
        "ratio", ratio,
        "robust_iterations", robust_mode ? robust_iters : 1,
        "cs_bins",  cs_bins,
        "cs_iterations", cs_iters,
        "cs_ratio", cs_ratio,
        "degree", degree
    );
    
    // Warm-up round
    auto warmup_start = timer::now();
    {
        PyObject *out = PyObject_Call(swht_function, vargs, kwargs);
        Py_DECREF(out);
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
        timestamp start;
        uint64_t duration;
        start_stamp(start)
        PyObject *out = PyObject_Call(swht_function, vargs, kwargs);
        end_stamp(duration, start)

        // Check for failure
        int failure = signal == out;
        Py_DECREF(out);
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
    Py_INCREF(Py_None);
    ((PyTupleObject *) vargs)->ob_item[0] = Py_None; //substitute the signal to avoid freeing it.
    Py_DECREF(vargs);
    Py_DECREF(kwargs);
    Py_DECREF(swht_function);
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
