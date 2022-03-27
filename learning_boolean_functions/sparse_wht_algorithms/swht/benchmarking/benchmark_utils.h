//=================================
// Useful functions for benchmarks
//=================================

#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include <vector>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <cmath>
#include <limits>
#include <unordered_map>


static const std::unordered_map<std::string, int> cs_algos_index = {
    {"naive", 0},
    {"randbin", 1},
    {"reedsolo", 2}
};

static const std::vector<std::string> cs_algos_names = {
    "naive",
    "random binning",
    "reed-solomon"
};

/** Median
 * Computes the median of a given data vector.
 */
double median(std::vector<double> &data) {
    std::sort(data.begin(), data.end());
    size_t length = data.size();
    if (length % 2) {
        return data[length / 2u];
    }
    return (data[length / 2u] + data[length / 2u - 1]) / 2.0;
}

/** Standard deviation
 * Computes the standard deviation of a given data vector.
 */
double std_deviation(const std::vector<double> &data) {
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double variance = 0.0;
    for (const double &x: data) {
        variance += std::pow(x - mean, 2);
    }
    variance /= data.size();
    return std::sqrt(variance);
}

/** Linear range
 * Computes a list of linearly spaced values in the given range.
 */
unsigned long lin_range(unsigned long from, unsigned long to, unsigned long n_values, std::vector<unsigned long> &out) {
    double step = (to - from) / (n_values - 1.0);
    for (double val = from; val <= to; val += step)
        out.push_back(std::round(val));
    return out.size();
}

/** Logarithmic range
 * Computes a list of logarithmically spaced values in the given range.
 */
unsigned long log_range(unsigned long from, unsigned long to, unsigned long n_values, int base, std::vector<unsigned long> &out) {
    double from_power = std::log(from) / std::log(base);
    double to_power = std::log(to) / std::log(base);
    double step = (to_power - from_power) / (n_values - 1.0);
    for (double power = from_power; power <= to_power + std::numeric_limits<double>::epsilon(); power += step){
        out.push_back(std::round(std::pow(base, power)));}
    return out.size();
}

/** CSV writer
 * Handles output to a given csv file.
 */
struct csv_writer {
    std::string file_path;
    std::ios_base::openmode mode;
    unsigned line_number;

    static const std::ios_base::openmode overwrite_mode = std::ios_base::out | std::ios_base::trunc;
    static const std::ios_base::openmode append_mode = std::ios_base::out | std::ios_base::app;

    csv_writer(std::string file_path, bool append): file_path(file_path) {
        line_number = 0;
        if (append) {
            std::ifstream reference(file_path, std::ios_base::in);
            if (!reference.good()) throw std::runtime_error("Cannot open file: " + file_path);
            reference.seekg(-1, std::ios_base::end);
            if (reference.peek() != '\n') throw std::runtime_error("Weird file (cannot find last line).");
            reference.unget();
            while(reference.peek() != '\n') reference.unget();
            reference.get();
            reference >> line_number;
            line_number++;
            mode = append_mode;
        } else {
            mode = overwrite_mode;
        }
        std::ofstream destination(file_path, mode);
        if (!destination.good()) throw std::runtime_error("Cannot create file: " + file_path);
    }

    void inc_line() {
        line_number++;
    }

    template<typename ... Args>
    void write_line(Args ... data) {
        std::ofstream out_file(file_path, mode);
        if (mode == overwrite_mode) {
            unpack_write_data(out_file, "run", data...);
            mode = append_mode;
        } else {
            unpack_write_data(out_file, line_number++, data...);
        }
    }

    template<typename ... Args>
    void write_partial_line(Args ... data) {
        std::ofstream out_file(file_path, mode);
        if (mode == overwrite_mode) {
            unpack_write_data(out_file, "run", data...);
            mode = append_mode;
        } else {
            unpack_write_data(out_file, line_number, data...);
        }
    }

private:
    template <typename top_t>
    void unpack_write_data(std::ofstream &out, top_t top) {
        out << top << std::endl;
    }
    template <typename top_t, typename ... Args>
    void unpack_write_data(std::ofstream &out, top_t top, Args ... tail) {
        out << top << ",";
        unpack_write_data(out, tail...);
    }
};


#endif
