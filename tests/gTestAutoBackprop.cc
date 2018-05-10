#include <map>
#include <vector>
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <math.h>
#include <sstream>
#include <memory>
// Include node and node children
#include "../src/node/node.h"
#include "../src/node/nodeDx.h"
#include "../src/node/nodeadd.h" 		// Tested
#include "../src/node/nodecos.h"		// Tested
#include "../src/node/nodecube.h"		// Test
#include "../src/node/nodedivide.h"		// Tested
#include "../src/node/nodeexponent.h"	// Tested
#include "../src/node/nodemultiply.h"	// Tested <- Check again
#include "../src/node/nodeexponential.h"// Tested
#include "../src/node/nodegaussian.h"   // Tested <- Check again
#include "../src/node/nodelog.h"		// Tested
#include "../src/node/nodelogit.h"		// Tested
// #include "../src/node/noderelu.h"		// Tested
#include "../src/node/nodesqrt.h"		// Tested
#include "../src/node/nodesin.h"		// Tested
#include "../src/node/nodesquare.h"		// Tested
#include "../src/node/nodesubtract.h"   // Tested
#include "../src/node/nodetanh.h"		// Tested
#include "../src/node/nodevariable.h"

// Non differentiable nodes
#include "../src/node/nodemax.h"
#include "../src/node/nodexor.h"
#include "../src/node/nodestep.h"

// Backprop progam
#include "../src/auto_backprop.h"

// Cost function
#include "../src/testMetrics.h"

// Stacks
#include "../src/stack.h"

// Nodevector
#include "../src/nodevector.h"

// Clean up code 
// CHeck with doxygen dconfig (run command from main folder)
// IPython notebook repair - get it working for testing

TEST(NodeDerivatives, NodeDerivativesTest)
{
	std::cout << "Starting tests\n";
	vector<ArrayXd> inputs;
	ArrayXd input1(5,1);
	input1(0,0) = 0;
	input1(1,0) = 1;
	input1(2,0) = 2;
	input1(3,0) = 3;
	input1(4,0) = 4;
	ArrayXd input2(5,1);
	input2(0,0) = 4;
	input2(1,0) = 3;
	input2(2,0) = 2;
	input2(3,0) = 1;
	input2(4,0) = 0;
	inputs.push_back(input1);
	inputs.push_back(input2);
	std::cout << "Initialized input vectors.\n";

	// ADD NODE CHECK -------------------------------------------------------------------------------
	NodeDx* toTest = new FT::NodeAdd();
	// Derivative wrt to first input
	expectedDerivative(0,0) = toTest->W[0];
	expectedDerivative(1,0) = toTest->W[0];
	expectedDerivative(2,0) = toTest->W[0];
	expectedDerivative(3,0) = toTest->W[0];
	expectedDerivative(4,0) = toTest->W[0];
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 0).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = toTest->W[1];
	expectedDerivative(1,0) = toTest->W[1];
	expectedDerivative(2,0) = toTest->W[1];
	expectedDerivative(3,0) = toTest->W[1];
	expectedDerivative(4,0) = toTest->W[1];
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 1).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = 4;
	expectedDerivative(1,0) = 3;
	expectedDerivative(2,0) = 2;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = 0;
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 2).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = 0;
	expectedDerivative(1,0) = 1;
	expectedDerivative(2,0) = 2;
	expectedDerivative(3,0) = 3;
	expectedDerivative(4,0) = 4;
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 3).matrix().norm(), 0.0001);

	// SUB NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeSubtract({1,1});
	expectedDerivative(0,0) = 1;
	expectedDerivative(1,0) = 1;
	expectedDerivative(2,0) = 1;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = 1;
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 0).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = -1;
	expectedDerivative(1,0) = -1;
	expectedDerivative(2,0) = -1;
	expectedDerivative(3,0) = -1;
	expectedDerivative(4,0) = -1;
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 1).matrix().norm(), 0.0001);
	
    expectedDerivative(0,0) = 4;
	expectedDerivative(1,0) = 3;
	expectedDerivative(2,0) = 2;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = 0;
    ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 2).matrix().norm(), 0.0001);

    expectedDerivative(0,0) = -0;
	expectedDerivative(1,0) = -1;
	expectedDerivative(2,0) = -2;
	expectedDerivative(3,0) = -3;
	expectedDerivative(4,0) = -4;
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 3).matrix().norm(), 0.0001);

	// MULT NODE CHECK-------------------------------------------------------------------------------
	toTest = new FT::NodeMultiply({1,1});
	expectedDerivative(0,0) = 4;
	expectedDerivative(1,0) = 3;
	expectedDerivative(2,0) = 2;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = 0;
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 0).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = 0;
	expectedDerivative(1,0) = 1;
	expectedDerivative(2,0) = 2;
	expectedDerivative(3,0) = 3;
	expectedDerivative(4,0) = 4;
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 1).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = 0;
	expectedDerivative(1,0) = 3;
	expectedDerivative(2,0) = 4;
	expectedDerivative(3,0) = 3;
	expectedDerivative(4,0) = 0;
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 2).matrix().norm(), 0.0001);

	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 3).matrix().norm(), 0.0001);

	// DIV NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeDivide({1,1});
	expectedDerivative(0,0) = MAX_DBL;	// Div by 0 (limited to 0)
	expectedDerivative(1,0) = 1.0/1;
	expectedDerivative(2,0) = 1.0/2;
	expectedDerivative(3,0) = 1.0/3;
	expectedDerivative(4,0) = 1.0/4; 
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 0).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = MIN_DBL;	// Div by 0
	expectedDerivative(1,0) = -3.0/1;
	expectedDerivative(2,0) = -2.0/4;
	expectedDerivative(3,0) = -1.0/9;
	expectedDerivative(4,0) = -0.0/16; 
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 1).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = MAX_DBL;	// Div by 0
	expectedDerivative(1,0) = 3.0/1;
	expectedDerivative(2,0) = 2.0/2;
	expectedDerivative(3,0) = 1.0/3;
	expectedDerivative(4,0) = 0.0/4;
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 2).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = -MAX_DBL;	//Div by 0
	expectedDerivative(1,0) = -3.0/1;
	expectedDerivative(2,0) = -2.0/2;
	expectedDerivative(3,0) = -1.0/3;
	expectedDerivative(4,0) = -0.0/4;
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 3).matrix().norm(), 0.0001);

	// x^y NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeExponent({1.0,1.0});
	expectedDerivative(0,0) = 0 * pow(4,0)/4; 
	expectedDerivative(1,0) = 1 * pow(3,1)/3;
	expectedDerivative(2,0) = 2 * pow(2,2)/2;
	expectedDerivative(3,0) = 3 * pow(1,3)/1;
	expectedDerivative(4,0) = 4 * pow(0,4)/0; // div by 0
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 0).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = 1 * pow(4,0) * log(4); 
    expectedDerivative(1,0) = 1 * pow(3,1) * log(3);
	expectedDerivative(2,0) = 1 * pow(2,2) * log(2);
	expectedDerivative(3,0) = 1 * pow(1,3) * log(1);
	expectedDerivative(4,0) = 1 * pow(0,4) * log(0); // log 0
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 1).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = 0 * pow(4,0)/1;
	expectedDerivative(1,0) = 1 * pow(3,1)/1;
	expectedDerivative(2,0) = 2 * pow(2,2)/1;
	expectedDerivative(3,0) = 3 * pow(1,3)/1;
	expectedDerivative(4,0) = 4 * pow(0,4)/1;
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 2).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = 4 * pow(0,4) * log(0); // Log by 0
	expectedDerivative(1,0) = 3 * pow(1,3) * log(1);
	expectedDerivative(2,0) = 2 * pow(2,2) * log(2);
	expectedDerivative(3,0) = 1 * pow(3,1) * log(3);
	expectedDerivative(4,0) = 0 * pow(4,0) * log(4);
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 3).matrix().norm(), 0.0001);

	// COS NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeCos({1.0});
	expectedDerivative(0,0) = -1 * sin(4);
	expectedDerivative(1,0) = -1 * sin(3);
	expectedDerivative(2,0) = -1 * sin(2);
	expectedDerivative(3,0) = -1 * sin(1);
	expectedDerivative(4,0) = -1 * sin(0);
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 0).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = -4 * sin(4);
	expectedDerivative(1,0) = -3 * sin(3);
	expectedDerivative(2,0) = -2 * sin(2);
	expectedDerivative(3,0) = -1 * sin(1);
	expectedDerivative(4,0) = -0 * sin(0);
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 1).matrix().norm(), 0.0001);

	// SIN NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeSin({1.0});
	expectedDerivative(0,0) = 1 * cos(4);
	expectedDerivative(1,0) = 1 * cos(3);
	expectedDerivative(2,0) = 1 * cos(2);
	expectedDerivative(3,0) = 1 * cos(1);
	expectedDerivative(4,0) = 1 * cos(0);
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 0).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = 4 * cos(4);
	expectedDerivative(1,0) = 3 * cos(3);
	expectedDerivative(2,0) = 2 * cos(2);
	expectedDerivative(3,0) = 1 * cos(1);
	expectedDerivative(4,0) = 0 * cos(0);
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 1).matrix().norm(), 0.0001);

	// ^3 NODE CHECK  -------------------------------------------------------------------------------
	toTest = new FT::NodeCube({1.0});
	expectedDerivative(0,0) = 3 * pow(4,2);
	expectedDerivative(1,0) = 3 * pow(3,2);
	expectedDerivative(2,0) = 3 * pow(2,2);
	expectedDerivative(3,0) = 3 * pow(1,2);
	expectedDerivative(4,0) = 3 * pow(0,2);
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 0).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = 3 * 64;
	expectedDerivative(1,0) = 3 * 27;
	expectedDerivative(2,0) = 3 *  8;
	expectedDerivative(3,0) = 3 *  1;
	expectedDerivative(4,0) = 3 *  0;
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 1).matrix().norm(), 0.0001);

	// e^x NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeExponential({1.0});
	expectedDerivative(0,0) = 1 * exp(4);
	expectedDerivative(1,0) = 1 * exp(3);
	expectedDerivative(2,0) = 1 * exp(2);
	expectedDerivative(3,0) = 1 * exp(1);
	expectedDerivative(4,0) = 1 * exp(0);
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 0).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = 4 * exp(4);
	expectedDerivative(1,0) = 3 * exp(3);
	expectedDerivative(2,0) = 2 * exp(2);
	expectedDerivative(3,0) = 1 * exp(1);
	expectedDerivative(4,0) = 0 * exp(0);
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 1).matrix().norm(), 0.0001);

	// GAUS NODE CHECK-------------------------------------------------------------------------------
	toTest = new FT::NodeGaussian({1.0});
	expectedDerivative(0,0) = -2 * 1 * 4 * exp(-16);
	expectedDerivative(1,0) = -2 * 1 * 3 * exp(-9);
	expectedDerivative(2,0) = -2 * 1 * 2 * exp(-4);
	expectedDerivative(3,0) = -2 * 1 * 1 * exp(-1);
	expectedDerivative(4,0) = -2 * 1 * 0 * exp(0);
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 0).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = -2 * 1 * 16 * exp(-16);
	expectedDerivative(1,0) = -2 * 1 * 9 * exp(-9);
	expectedDerivative(2,0) = -2 * 1 * 4 * exp(-4);
	expectedDerivative(3,0) = -2 * 1 * 1 * exp(-1);
	expectedDerivative(4,0) = -2 * 1 * 0 * exp(0);
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 1).matrix().norm(), 0.0001);

	// LOG NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeLog({1.0});
	expectedDerivative(0,0) = 1.0/4;
	expectedDerivative(1,0) = 1.0/3;
	expectedDerivative(2,0) = 1.0/2;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = MAX_DBL; // Check if this is intended
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 0).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = 1;
	expectedDerivative(1,0) = 1;
	expectedDerivative(2,0) = 1;
	expectedDerivative(3,0) = 1;
	expectedDerivative(4,0) = 1;
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 1).matrix().norm(), 0.0001);

	// LOGIT NODE CHECK------------------------------------------------------------------------------
	toTest = new FT::NodeLogit({1.0});
	expectedDerivative(0,0) = (1 * exp(1 * 4))/pow(exp(1 * 4) + 1, 2);
	expectedDerivative(1,0) = (1 * exp(1 * 3))/pow(exp(1 * 3) + 1, 2);
	expectedDerivative(2,0) = (1 * exp(1 * 2))/pow(exp(1 * 2) + 1, 2);
	expectedDerivative(3,0) = (1 * exp(1 * 1))/pow(exp(1 * 1) + 1, 2);
	expectedDerivative(4,0) = (1 * exp(1 * 0))/pow(exp(1 * 0) + 1, 2);
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 0).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = (4 * exp(1 * 4))/pow(exp(1 * 4) + 1, 2);
	expectedDerivative(1,0) = (3 * exp(1 * 3))/pow(exp(1 * 3) + 1, 2);
	expectedDerivative(2,0) = (2 * exp(1 * 2))/pow(exp(1 * 2) + 1, 2);
	expectedDerivative(3,0) = (1 * exp(1 * 1))/pow(exp(1 * 1) + 1, 2);
	expectedDerivative(4,0) = (0 * exp(1 * 0))/pow(exp(1 * 0) + 1, 2);
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 1).matrix().norm(), 0.0001);

	// RELU NODE CHECK------------------------------------------------------------------------------
	// TODO
	// toTest = new FT::NodeRelu({1.0});
	// expectedDerivative(0,0) = 1;
	// expectedDerivative(1,0) = 1;
	// expectedDerivative(2,0) = 1;
	// expectedDerivative(3,0) = 1;
	// expectedDerivative(4,0) = 1;
	// if ((expectedDerivative.matrix() - toTest->getDerivative(inputs, 0).matrix()).norm() > 0.0001) { // Currently pseudocode
	// 	std::cout << "Relu node FAILED!\n";
	// }

	// SQRT NODE CHECK-------------------------------------------------------------------------------
	toTest = new FT::NodeSqrt({1.0});
	expectedDerivative(0,0) = 1/(2 * sqrt(4));
	expectedDerivative(1,0) = 1/(2 * sqrt(3));
	expectedDerivative(2,0) = 1/(2 * sqrt(2));
	expectedDerivative(3,0) = 1/(2 * sqrt(1));
	expectedDerivative(4,0) = 1/(2 * sqrt(0));
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 0).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = 4/(2 * sqrt(4));
	expectedDerivative(1,0) = 3/(2 * sqrt(3));
	expectedDerivative(2,0) = 2/(2 * sqrt(2));
	expectedDerivative(3,0) = 1/(2 * sqrt(1));
	expectedDerivative(4,0) = 0/(2 * sqrt(0));
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 1).matrix().norm(), 0.0001);

	// ^2  NODE CHECK -------------------------------------------------------------------------------
	toTest = new FT::NodeSquare({1.0});
	expectedDerivative(0,0) = 2 * 1 * 4;
	expectedDerivative(1,0) = 2 * 1 * 3;
	expectedDerivative(2,0) = 2 * 1 * 2;
	expectedDerivative(3,0) = 2 * 1 * 1;
	expectedDerivative(4,0) = 2 * 1 * 0;
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 0).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = 2 * 16;
	expectedDerivative(1,0) = 2 *  9;
	expectedDerivative(2,0) = 2 *  4;
	expectedDerivative(3,0) = 2 *  1;
	expectedDerivative(4,0) = 2 *  0;
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 1).matrix().norm(), 0.0001);

	// TANH NODE CHECK-------------------------------------------------------------------------------
	toTest = new FT::NodeTanh({1.0});
	expectedDerivative(0,0) = 0.0013409506830258968799702;
	expectedDerivative(1,0) = 0.00986603716544019127315616968;
	expectedDerivative(2,0) = 0.07065082485316446568624765586105;
	expectedDerivative(3,0) = 0.41997434161402606939449673904170;
	expectedDerivative(4,0) = 1;
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 0).matrix().norm(), 0.0001);

	expectedDerivative(0,0) = 4 * 0.0013409506830258968799702;
	expectedDerivative(1,0) = 3 * 0.00986603716544019127315616968;
	expectedDerivative(2,0) = 2 * 0.07065082485316446568624765586105;
	expectedDerivative(3,0) = 1 * 0.41997434161402606939449673904170;
	expectedDerivative(4,0) = 0;
	ASSERT_NEAR(expectedDerivative.matrix().norm(), toTest->getDerivative(inputs, 1).matrix().norm(), 0.0001);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper functions for backprop testing
Node* parseToNode(std::string token) {
	if (token == "+") {
    	return new FT::NodeAdd();
    } else if (token == "-") {
    	return new FT::NodeSubtract();
    } else if (token == "/") {
    	return new FT::NodeDivide();
    } else if (token == "*") {
    	return new FT::NodeMultiply();
    } else if (token == "cos") {
    	return new FT::NodeCos();
    } else if (token == "sin") {
    	return new FT::NodeSin();
    } else if (token == "tanh") {
    	return new FT::NodeTanh();
    } else if (token == "x0") {
    	return new FT::NodeVariable(0);
    } else if (token == "x1") {
    	return new FT::NodeVariable(1);
    } else if (token == "exponent") {
    	return new FT::NodeExponent();
    } else if (token == "max") {
    	return new FT::NodeMax();
    } else if (token == "xor") {
    	return new FT::NodeXor();
    } else if (token == "step") {
    	return new FT::NodeStep();
    }
}

NodeVector programGen(std::string program) {
	FT::NodeVector program;
	std::string txt;

	char ch = ' ';
	size_t pos = txt.find( ch );
    size_t initialPos = 0;

    // Decompose statement
    std::string token;
    while( pos != std::string::npos ) {
    	token = txt.substr( initialPos, pos - initialPos );
        std::cout << token << "\n";

        program.push_back(unique_ptr<Node>(parseToNode(token)));

        initialPos = pos + 1;

        pos = txt.find( ch, initialPos );
    }

    // Add the last one
    token = txt.substr( initialPos, std::min( pos, txt.size() ) - initialPos + 1 );
    std::cout << token << "\n";
    program.push_back(unique_ptr<Node>(parseToNode(token)));
    std::cout << "ProgramGen done";

    return program;
}

TEST(SingleNodeBackprop, SingleNodeBackpropTest) {
	// Create input data and labels
	MatrixXd x(2, 2);
	VectorXd y(2);
	x << 7.3, 6.7, 
		 12.4, 13.2;

	y << 9.0, 
		 8.0;

	// Params for training
	std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z; 
	int iters = 100;
	double learning_rate = 0.1;

	FT::NodeVector program;
	FT::AutoBackProp* engine = new FT::AutoBackProp(FT::metrics::d_squared_difference, iters, learning_rate);

	program = programGen("x0 x1 +");
	engine->run(p0, x, y, Z); // Update pointer to NodeVector internally
	ASSERT_NEAR(program[2]->W[0], 5.2173983397349467e+161, 0.0001);
	ASSERT_NEAR(program[2]->W[1], 2.85026295066146e+161, 0.0001);

	program = programGen("x0 sin");
	engine->run(p0, x, y, Z); // Update pointer to NodeVector internally
	ASSERT_NEAR(program[1]->W[0], 110.9608076750401, 0.0001);

	program = programGen("x0 x1 cos");
	engine->run(p0, x, y, Z); // Update pointer to NodeVector internally
	ASSERT_NEAR(program[1]->W[0], 45.060811045865115, 0.0001);

	program = programGen("x0 x1 -");
	engine->run(p0, x, y, Z); // Update pointer to NodeVector internally
	ASSERT_NEAR(program[2]->W[0], -1.2378324212904017e+161, 0.0001);
	ASSERT_NEAR(program[2]->W[1], 6.7622743363521589e+160, 0.0001);
}

TEST(BranchingBackprop, BranchingBackpropTest) {
	// Create input data and labels
	MatrixXd x(2, 2);
	VectorXd y(2);
	x << 7.3, 6.7, 
		 12.4, 13.2;

	y << 9.0, 
		 8.0;

	// Params for training
	std::map<string, std::pair<vector<ArrayXd>, vector<ArrayXd> > > Z; 
	int iters = 100;
	double learning_rate = 0.1;

	FT::NodeVector p0;
	FT::AutoBackProp* engine = new FT::AutoBackProp(FT::metrics::d_squared_difference, iters, learning_rate);

	p0 = programGen("x0 x1 + cos x0 x1 - sin + cos");
	engine->run(p0, x, y, Z); // Update pointer to NodeVector internally
	ASSERT_NEAR(program[2]->W[0], -17.447603053106736, 0.0001);
	ASSERT_NEAR(program[2]->W[1], -8.8755031869983689, 0.0001);

	ASSERT_NEAR(program[3]->W[0], -17.294169609277102, 0.0001);

	ASSERT_NEAR(program[6]->W[0], -3.8150579823922475, 0.0001);
	ASSERT_NEAR(program[6]->W[1], 3.2999572235783612, 0.0001);

	ASSERT_NEAR(program[7]->W[0], 2.4126715215974466, 0.0001);

	ASSERT_NEAR(program[8]->W[0], 0.40660933720814846, 0.0001);
	ASSERT_NEAR(program[8]->W[1], 0.61777127007967136, 0.0001);

	ASSERT_NEAR(program[9]->W[0], -0.00043425247634781806, 0.0001);

	p0 = programGen("x0 sin cos tanh x1 x0 + tanh * cos");
	engine->run(p0, x, y, Z); // Update pointer to NodeVector internally
	ASSERT_NEAR(program[1]->W[0], 2.4435165950887514, 0.0001);
	
	ASSERT_NEAR(program[2]->W[0], 1.2480241579003173, 0.0001);
	
	ASSERT_NEAR(program[3]->W[0], 0.54841048305018669, 0.0001);
	
	ASSERT_NEAR(program[6]->W[0], 1, 0.0001);
	ASSERT_NEAR(program[6]->W[1], 1, 0.0001);

	ASSERT_NEAR(program[7]->W[0], 1, 0.0001);

	ASSERT_NEAR(program[8]->W[0], 0.30919643129113983, 0.0001);
	ASSERT_NEAR(program[8]->W[1], 0.30919643129113983, 0.0001);

	ASSERT_NEAR(program[9]->W[8], 0.30919643129113988, 0.0001);
}