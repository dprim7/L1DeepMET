#ifndef L1DEEPMET_BRIDGE_H_
#define L1DEEPMET_BRIDGE_H_

#include "firmware/l1deepmet.h"
#include "firmware/nnet_utils/nnet_helpers.h"
#include <algorithm>
#include <map>

// hls-fpga-machine-learning insert bram

namespace nnet {
bool trace_enabled = false;
std::map<std::string, void *> *trace_outputs = NULL;
size_t trace_type_size = sizeof(double);
} // namespace nnet

extern "C" {

struct trace_data {
    const char *name;
    void *data;
};

void allocate_trace_storage(size_t element_size) {
    nnet::trace_enabled = true;
    nnet::trace_outputs = new std::map<std::string, void *>;
    nnet::trace_type_size = element_size;
}

void free_trace_storage() {
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        void *ptr = i->second;
        free(ptr);
    }
    nnet::trace_outputs->clear();
    delete nnet::trace_outputs;
    nnet::trace_outputs = NULL;
    nnet::trace_enabled = false;
}

void collect_trace_output(struct trace_data *c_trace_outputs) {
    int ii = 0;
    for (std::map<std::string, void *>::iterator i = nnet::trace_outputs->begin(); i != nnet::trace_outputs->end(); i++) {
        c_trace_outputs[ii].name = i->first.c_str();
        c_trace_outputs[ii].data = i->second;
        ii++;
    }
}

// hls-fpga-machine-learning insert tb_input_writer

// Wrapper of top level function for Python bridge
void l1deepmet_float(
    float *continuous_inputs, float *momentum_inputs, float *pdgid_inputs, float *charge_inputs,
    float *layer23_out
) {

    input2_t continuous_inputs_ap[128*5];
    nnet::convert_data<float, input2_t, 128*5>(continuous_inputs, continuous_inputs_ap);
    input20_t momentum_inputs_ap[128*2];
    nnet::convert_data<float, input20_t, 128*2>(momentum_inputs, momentum_inputs_ap);
    input_t pdgid_inputs_ap[128];
    nnet::convert_data<float, input_t, 128>(pdgid_inputs, pdgid_inputs_ap);
    input4_t charge_inputs_ap[128];
    nnet::convert_data<float, input4_t, 128>(charge_inputs, charge_inputs_ap);

    result_t layer23_out_ap[2];

    l1deepmet(continuous_inputs_ap,momentum_inputs_ap,pdgid_inputs_ap,charge_inputs_ap,layer23_out_ap);

    nnet::convert_data<result_t, float, 2>(layer23_out_ap, layer23_out);
}

void l1deepmet_double(
    double *continuous_inputs, double *momentum_inputs, double *pdgid_inputs, double *charge_inputs,
    double *layer23_out
) {

    input2_t continuous_inputs_ap[128*5];
    nnet::convert_data<double, input2_t, 128*5>(continuous_inputs, continuous_inputs_ap);
    input20_t momentum_inputs_ap[128*2];
    nnet::convert_data<double, input20_t, 128*2>(momentum_inputs, momentum_inputs_ap);
    input_t pdgid_inputs_ap[128];
    nnet::convert_data<double, input_t, 128>(pdgid_inputs, pdgid_inputs_ap);
    input4_t charge_inputs_ap[128];
    nnet::convert_data<double, input4_t, 128>(charge_inputs, charge_inputs_ap);

    result_t layer23_out_ap[2];

    l1deepmet(continuous_inputs_ap,momentum_inputs_ap,pdgid_inputs_ap,charge_inputs_ap,layer23_out_ap);

    nnet::convert_data<result_t, double, 2>(layer23_out_ap, layer23_out);
}
}

#endif
