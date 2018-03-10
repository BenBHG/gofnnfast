package gofnnfast

// #include "fnnfast/fnnfast.h"
// #include "fnnfast/fnnfast.c"
// neuron gofnnfast_msd_helper(struct fnnfast_data *data, neuron *input_set, neuron *output_set, size_t num) {
// 	neuron **is = malloc(num * sizeof(neuron*));
// 	neuron **os = malloc(num * sizeof(neuron*));
// 	for (size_t i = 0; i < num; ++i) {
// 		is[i] = &input_set[i * data->num_input];
// 		os[i] = &output_set[i * data->num_output];
// 	}
//  neuron msd = fnnfast_mean_squared_deviation(data, is, os, num);
// 	free(is);
// 	free(os);
// 	return msd;
// }
import "C"
import "unsafe"

type FnnfastValue float64

type FnnfastData struct {
	InputCount,
	HiddenCount,
	OutputCount uint
	HiddenNeurons,
	DeltaInputHidden,
	DeltaHiddenOutput,
	InputWeights,
	HiddenWeights *FnnfastValue
}

func (ffd *C.struct_fnnfast_data) ffn() *FnnfastData {
	p := unsafe.Pointer(ffd)
	return (*FnnfastData)(p)
}

func (ffn *FnnfastData) ffd() *C.struct_fnnfast_data {
	p := unsafe.Pointer(ffn)
	return (*C.struct_fnnfast_data)(p)
}

func FnnfastNew(inputs, hidden, outputs uint) *FnnfastData {
	ni := (C.ulonglong)(inputs)
	nh := (C.ulonglong)(hidden)
	no := (C.ulonglong)(outputs)
	ffd := C.fnnfast_new(ni, nh, no)
	return ffd.ffn()
}

func FnnfastDelete(ffn *FnnfastData) {
	C.fnnfast_delete(ffn.ffd())
}

func (ffn *FnnfastData) FixPointers() {
	C.fnnfast_fix_pointers(ffn.ffd())
}

func (ffn *FnnfastData) Size() uint {
	return (uint)(C.fnnfast_size(ffn.ffd()))
}

func (ffn *FnnfastData) Randomize(seed uint) {
	C.fnnfast_randomize(ffn.ffd(), (C.uint)(seed))
}

func (ffn *FnnfastData) FeedForwardBuf(input, buf []FnnfastValue) {
	C.fnnfast_feedforward(ffn.ffd(), (*C.double)(&input[0]), (*C.double)(&buf[0]))
}

func (ffn *FnnfastData) FeedForward(input []FnnfastValue) []FnnfastValue {
	o := make([]FnnfastValue, ffn.OutputCount)
	ffn.FeedForwardBuf(input, o)
	return o
}

func (ffn *FnnfastData) MeanSquaredDeviation(inputSet, outputSet [][]FnnfastValue) FnnfastValue {
	ns := (C.ulonglong)(len(inputSet))
	is := make([]FnnfastValue, 0)
	for _, input := range inputSet {
		is = append(is, input...)
	}
	os := make([]FnnfastValue, 0)
	for _, output := range outputSet {
		os = append(os, output...)
	}
	msd := C.gofnnfast_msd_helper(ffn.ffd(), (*C.double)(unsafe.Pointer(&is[0])), (*C.double)(unsafe.Pointer(&os[0])), ns)
	return (FnnfastValue)(msd)
}

func (ffn *FnnfastData) Train(input, output []FnnfastValue, rate, momentum FnnfastValue, buf []FnnfastValue) {
	i := (*C.double)(&input[0])
	o := (*C.double)(&output[0])
	r := (C.double)(rate)
	m := (C.double)(momentum)
	tb := (*C.double)(&buf[0])
	C.fnnfast_train(ffn.ffd(), i, o, r, m, tb)
}
