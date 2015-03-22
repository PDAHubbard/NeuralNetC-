/*
 * Net.cpp
 *
 *  Created on: 20.03.2015
 *      Author: fargo
 */

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

using namespace std;

struct Connection {
	double weight;
	double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;


// ******************************* Class Neuron ***************************

class Neuron {
public:
		Neuron(unsigned numOutputs, unsigned myIndex);
		void setOutputValue(double val) { m_outputVal = val;}
		double getOutputValue(void) const {return m_outputVal;}
		void feedForward(const Layer &prevLayer);

private:
		double m_outputVal;
		vector<Connection> m_outputWeights;
		unsigned m_myIndex;

		static double randomWeight(void) { return rand() / double(RAND_MAX);}
		static double transferfunction(double x);
		static double transferfunctionDerivative(double x);
};

static double Neuron::transferfunction(double x){
	//tanh

	return tanh(x);
}

static double Neuron::transferfunctionDerivative(double x){
	//tanh derivative
	return 1.0 - x * x;
}

void Neuron::feedForward(const Layer &prevLayer){
	double sum = 0.0;

	//Sum the previous layers outputs (which are our inputs)
	//Include the bias node from the previous layer

	for (unsigned n=0; n < prevLayer.size(); ++n){
		sum += prevLayer[n].getOutputValue() *
				prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_outputVal = Neuron::transferfunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
	for (unsigned c=0; c < numOutputs; ++c){
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
}

// ******************************* Class Net ******************************

class Net {
public:
	Net(const vector<unsigned> &topology);
	void feedForward(const vector<double> &inputVals);
	void backProp(const vector<double> &targetVals){};
	void getResults(vector<double> &resultVals) const{};


private:
	vector<Layer> m_layers;  //m_layers[layerNum][neuroNum]
	double m_error;
	double m_recentAverageError;
	double m_recentAverageSmoothingFactor;


};

void Net::backProp(const vector<double> &targetVals){
	//Calculate overall net error (RMS of output neuron errors)
	Layer &outputLayer = m_layers.back();
	m_error=0.0;

	for (unsigned n=0; n < outputLayer.size()-1; ++n){
		double delta = targetVals[n] - outputLayer[n].getOutputValue;
		m_error += delta*delta;
	}
	m_error /= outputLayer.size()-1;
	m_error = sqrt(m_error);

	//Implement a recent avergae measurement
	m_recentAverageError =
			(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
			/ (m_recentAverageSmoothingFactor + 1.0);

	//Calculate output layer gradients
	for (unsigned n=0; n < outputLayer.size()-1; ++n){
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

	//Calculate hidden layer gradients
	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum){
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum+1];

		for (unsigned n = 0; n < hiddenLayer.size(); ++n){
			hiddenLayer[n]=calcHiddenGradients(nextLayer);
		}
	}

	//For all layers from outputs to the first hidden layer, update connection weights
	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum){
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum-1];

		for (unsigned n = 0; n<layer.size() - 1; ++n){
			layer[n].updateInputWeights(prevLayer);
		}

	}

}

void Net::feedForward(const vector<double> &inputVals){
	assert(inputVals.size() == m_layers[0].size() - 1);

	//Assign the input values to the input neurons.
	for (unsigned i = 0; i < inputVals.size(); ++i){
		m_layers[0][i].setOutputValue(inputVals[i]));
	}

	//Forward propogate. Start at first HIDDEN layer (1)
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum){
		Layer &prevLayer = m_layers[layerNum -1];
		for (unsigned n = 0; n < m_layers[layerNum].size(); ++n){
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

Net::Net(const vector<unsigned> &topology){

	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
		m_layers.push_back(Layer());	//create a new empty layer;
		unsigned numOutputs = layerNum == numLayers - 1 ? 0 : topology[layerNum+1];

		//Now fill the layer with neurons; <= to make space for the bias neuron.
		for (unsigned neuroNum = 0; neuroNum <= topology[layerNum]; ++neuroNum){
			m_layers.back().push_back(Neuron(numOutputs));
			cout << "Made a neuron" << endl;
		}
	}

}

int main() {

	// eg { 3, 2, 1}
	vector<unsigned> topology;
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(1);
	Net myNet(topology);

	vector<double> inputVals;
	myNet.feedForward(inputVals);

	vector<double> targetVals;
	myNet.backProp(targetVals);

	vector<double> resultVals;
	myNet.getResults(resultVals);
}
