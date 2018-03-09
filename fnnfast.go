package gofnnfast

// #include "fnnfast/fnnfast.h"
// #include "fnnfast/fnnfast.c"
// typedef double* pDouble;
// neuron gofnnfast_msd_helper(struct fnnfast_data *data, void *input_set, void *output_set, size_t num) {
//   return fnnfast_mean_squared_deviation(data, (neuron**)input_set, (neuron**)output_set, num);
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
	_inputSet := make([]C.pDouble, len(inputSet))
	_outputSet := make([]C.pDouble, len(outputSet))
	for i := range inputSet {
		_inputSet[i] = (C.pDouble)(C.malloc(C.size_t(len(inputSet[i]))))
		C.memcpy(unsafe.Pointer(_inputSet[i]), unsafe.Pointer(&inputSet[i]), ns)
	}
	for i := range outputSet {
		_outputSet[i] = (C.pDouble)(C.malloc(C.size_t(len(outputSet[i]))))
		C.memcpy(unsafe.Pointer(_outputSet[i]), unsafe.Pointer(&outputSet[i]), ns)
	}
	is := unsafe.Pointer(_inputSet[0])
	os := unsafe.Pointer(_outputSet[0])
	msd := C.gofnnfast_msd_helper(ffn.ffd(), is, os, ns)
	for i := range _inputSet {
		C.free(unsafe.Pointer(_inputSet[i]))
	}
	for i := range _outputSet {
		C.free(unsafe.Pointer(_outputSet[i]))
	}
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
