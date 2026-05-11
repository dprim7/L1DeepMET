#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "firmware/l1deepmet.h"
#include "firmware/nnet_utils/nnet_helpers.h"

// hls-fpga-machine-learning insert bram

#define CHECKPOINT 5000

namespace nnet {
bool trace_enabled = true;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
} // namespace nnet

int main(int argc, char **argv) {

    // load input data from text file
    std::ifstream fin("tb_data/tb_input_features.dat");
    // load predictions from text file
    std::ifstream fpr("tb_data/tb_output_predictions.dat");

#ifdef RTL_SIM
    std::string RESULTS_LOG = "tb_data/rtl_cosim_results.log";
#else
    std::string RESULTS_LOG = "tb_data/csim_results.log";
#endif
    std::ofstream fout(RESULTS_LOG);

    std::string iline;
    std::string pline;
    int e = 0;

    if (fin.is_open() && fpr.is_open()) {
        while (std::getline(fin, iline) && std::getline(fpr, pline)) {
            if (e % CHECKPOINT == 0)
                std::cout << "Processing input " << e << std::endl;
            char *cstr = const_cast<char *>(iline.c_str());
            char *current;
            std::vector<float> in;
            current = strtok(cstr, " ");
            while (current != NULL) {
                in.push_back(atof(current));
                current = strtok(NULL, " ");
            }
            cstr = const_cast<char *>(pline.c_str());
            std::vector<float> pr;
            current = strtok(cstr, " ");
            while (current != NULL) {
                pr.push_back(atof(current));
                current = strtok(NULL, " ");
            }

            // hls-fpga-machine-learning insert data
      continuous_inputs_t continuous_inputs[128*5];
      nnet::copy_data<float, continuous_inputs_t, 0, 128*5>(in, continuous_inputs);
      momentum_inputs_t momentum_inputs[128*2];
      nnet::copy_data<float, momentum_inputs_t, 640, 128*2>(in, momentum_inputs);
      pdgid_inputs_t pdgid_inputs[128*6];
      nnet::copy_data<float, pdgid_inputs_t, 896, 128*6>(in, pdgid_inputs);
      charge_inputs_t charge_inputs[128*4];
      nnet::copy_data<float, charge_inputs_t, 1664, 128*4>(in, charge_inputs);
      result_t layer37_out[2];

            // hls-fpga-machine-learning insert top-level-function
            l1deepmet(continuous_inputs,momentum_inputs,pdgid_inputs,charge_inputs,layer37_out);

            if (e % CHECKPOINT == 0) {
                std::cout << "Predictions" << std::endl;
                // hls-fpga-machine-learning insert predictions
                for(int i = 0; i < 2; i++) {
                  std::cout << pr[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "Quantized predictions" << std::endl;
                // hls-fpga-machine-learning insert quantized
                nnet::print_result<result_t, 2>(layer37_out, std::cout, true);
            }
            e++;

            // hls-fpga-machine-learning insert tb-output
            nnet::print_result<result_t, 2>(layer37_out, fout);
        }
        fin.close();
        fpr.close();
    } else {
        std::cout << "INFO: Unable to open input/predictions file, using default input." << std::endl;
        const unsigned NUM_TEST_SAMPLES = 5;
        for (unsigned i = 0; i < NUM_TEST_SAMPLES; i++) {
            // hls-fpga-machine-learning insert zero
            continuous_inputs_t continuous_inputs[128*5];
            nnet::fill_zero<continuous_inputs_t, 128*5>(continuous_inputs);
            momentum_inputs_t momentum_inputs[128*2];
            nnet::fill_zero<momentum_inputs_t, 128*2>(momentum_inputs);
            pdgid_inputs_t pdgid_inputs[128*6];
            nnet::fill_zero<pdgid_inputs_t, 128*6>(pdgid_inputs);
            charge_inputs_t charge_inputs[128*4];
            nnet::fill_zero<charge_inputs_t, 128*4>(charge_inputs);
            result_t layer37_out[2];

            // hls-fpga-machine-learning insert top-level-function
            l1deepmet(continuous_inputs,momentum_inputs,pdgid_inputs,charge_inputs,layer37_out);

            // hls-fpga-machine-learning insert output
            nnet::print_result<result_t, 2>(layer37_out, std::cout, true);

            // hls-fpga-machine-learning insert tb-output
            nnet::print_result<result_t, 2>(layer37_out, fout);
        }
    }

    fout.close();
    std::cout << "INFO: Saved inference results to file: " << RESULTS_LOG << std::endl;

    return 0;
}
