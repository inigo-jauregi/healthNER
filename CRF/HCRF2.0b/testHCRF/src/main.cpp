#include "hCRF.h"
#include <iostream>
#include <string>
#ifdef WIN32
#include <conio.h>
#endif

#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

#define MODE_TRAIN 1
#define MODE_TEST  2
#define MODE_DEBUG 4
#define MODE_VALIDATE  8

#define TOOLBOX_CRF  1
#define TOOLBOX_HCRF  2
#define TOOLBOX_LDCRF  4
#define TOOLBOX_GHCRF  8
#define TOOLBOX_LVPERCEPTRON 32
#define TOOLBOX_SDCRF 16

void usage (char **argv)
{  
  cerr << "HCRF library"<< endl << endl;
  cerr << "usage> " << argv[0] << " [-t] [-T] [-d filename] [-l filename] [-D filename] [-L filename] [-m filename] [-f filename] [-r filename] [-o cg|bfgs]" << endl << endl;
  cerr << "options:" << endl;
  cerr << " -t\tTrain the model" << endl;
  cerr << " -tc\tContinue to train the model" << endl;
  cerr << " -a\tSelect the model type (crf, hcrf, ldcrf, fhcrd, ghcrf, sdcrf) (def. = crf)" << endl;
  cerr << " -h\tNumber of hidden state(def.=3)" << endl;
  cerr << " -d\tName of file containing the training data (def.= dataTrain.csv)" << endl;
  cerr << " -ds\tName of file containing the sparse training data" << endl;
  cerr << " -l\tName of file containing the training labels (def.= labelsTrain.csv)" << endl;
  cerr << " -T\tTest the model" << endl;
  cerr << " -TT\tTest the model on both the training set and the test set" << endl;
  cerr << " -D\tName of file containing the testing data (def.= dataTest.csv)" << endl;
  cerr << " -L\tName of file containing the testing labels (def.= labelsTest.csv)" << endl;
  cerr << " -m\tName of file where model is written (def.= model.txt)" << endl;
  cerr << " -f\tName of file where the features are written (def.= features.txt)" << endl;
  cerr << " -r\tName of file where the computed labels are written (def.= results.txt)" << endl;
  cerr << " -c\tName of file where the statistics are written (def.= stats.txt)" << endl;
  cerr << " -o\tSelect optimizer: cg, bfgs, lbfgs, asa or owlqn (def.= cg)" << endl;
  cerr << " -i\tMaximum number of iterations (def.= 200)" << endl;
  cerr << " -s2\tSigma2: L2-norm regularization factor(def.= 0.0)" << endl;
  cerr << " -s1\tSigma1: L1-norm regularization factor (def.= 0.0)" << endl;
  cerr << " -p\tDebug print level (def.= 1)" << endl;
  cerr << " -w\tSpecify the number of neighboring observations used in the input vector (def. window size= 0)" << endl;
  cerr << " -R\tRange for initail weigths (def.=[-1 1])" << endl;
  cerr << " -P\tnumber of parallel thread to use" << endl;
  cerr << " -I\tinitialization strategy (def.= random)" << endl;
  cerr << endl;
  exit(1);
}

int main(int argc, char **argv)
{
	int mode = MODE_TRAIN; // int mode = 0;
	bool bContinueTraining = false;
	int bTestOnTrainingSet = 0;
	int opt = OPTIMIZER_BFGS;
	int maxIt = 300; // int max = -1;
	double regFactorL2 = 0.0;
	double regFactorL1 = 0.0;
	int toolboxType = TOOLBOX_LDCRF;
	Toolbox* toolbox = NULL;
	int nbHiddenStates = 3;
	int windowSize = 0;
	int debugLevel = 1;
	double initWeightRangeMin = -1.0;
	double initWeightRangeMax = 1.0;
	int InitMode = INIT_RANDOM;
#ifdef UNIX
	string home = "./";
#else
	string home = ".\\";
#endif
	string filenameDataTrain = home + "dataTrain.csv";
	string filenameDataTrainSparse;
	string filenameLabelsTrain = home + "labelsTrain.csv" ;
	string filenameSeqLabelsTrain = home + "seqLabelsTrain.csv";
	string filenameDataTest = home + "dataTest.csv";
	string filenameDataTestSparse;
	string filenameLabelsTest = home + "labelsTest.csv";
	string filenameSeqLabelsTest = home + "seqLabelsTest.csv";
	string filenameDataValidate = home + "dataValidate.csv";
	string filenameDataValidateSparse;
	string filenameLabelsValidate = home + "labelsValidate.csv";
	string filenameSeqLabelsValidate = home + "seqLabelsValidate.csv";
	string filenameModel = home + "model.txt";
	string filenameFeatures = home + "features.txt";
	string filenameOutput = home + "results.txt";
	string filenameStats = home + "stats.txt";

	/* Read command-line arguments */
	for (int k=1; k<argc; k++)
	{
		if(argv[k][0] != '-') break;
		else if(argv[k][1] == 't') 
		{
			mode = mode | MODE_TRAIN;
			if(argv[k][2] == 'c') 
				bContinueTraining = true;
		}
		else if(argv[k][1] == 'T') 
		{
			mode = mode | MODE_TEST;
			if(argv[k][2] == 'T') 
				bTestOnTrainingSet = 1;
			if(argv[k][2] == 'V') 
				bTestOnTrainingSet = 2;
		}
		else if(argv[k][1] == 'v') 
		{
			mode += MODE_VALIDATE;
		}
		else if(argv[k][1] == 'd') 
		{
			if(argv[k][2] == 's') 
			{
				filenameDataTrainSparse = argv[++k];
				filenameDataTrain = "";
			}
			else
				filenameDataTrain = argv[++k];
		}
		else if(argv[k][1] == 'l') 
		{
			filenameLabelsTrain = argv[++k];
		}
		else if(argv[k][1] == 'D') 
		{
			if(argv[k][2] == 'S') 
			{
				filenameDataTestSparse = argv[++k];
				filenameDataTest = "";
			}
			else
				filenameDataTest = argv[++k];
		}
		else if(argv[k][1] == 'L') 
		{
			filenameLabelsTest = argv[++k];
		}
		else if(argv[k][1] == 'm') 
		{
			filenameModel = argv[++k];
		}
		else if(argv[k][1] == 'f') 
		{
			filenameFeatures = argv[++k];
		}
		else if(argv[k][1] == 'r') 
		{
			filenameOutput = argv[++k];
		}
		else if(argv[k][1] == 'c') 
		{
			filenameStats = argv[++k];
		}
		else if(argv[k][1] == 'I') 
		{
			if(!strcmp(argv[k+1],"random"))
				InitMode = INIT_RANDOM;
			else if(!strcmp(argv[k+1],"gaussian"))
				InitMode = INIT_RANDOM_GAUSSIAN;
			else if(!strcmp(argv[k+1],"zero"))
				InitMode = INIT_ZERO;
			k++;
		}
		else if(argv[k][1] == 'o') 
		{
			if(!strcmp(argv[k+1],"cg"))
				opt = OPTIMIZER_CG;
			else if(!strcmp(argv[k+1],"bfgs"))
				opt = OPTIMIZER_BFGS;
			else if(!strcmp(argv[k+1],"asa"))
				opt = OPTIMIZER_ASA;
			else if(!strcmp(argv[k+1],"owlqn"))
				opt = OPTIMIZER_OWLQN;
			else if(!strcmp(argv[k+1],"lbfgs"))
				opt = OPTIMIZER_LBFGS;
			k++;
		}
		else if(argv[k][1] == 'a') 
		{
			if(!strcmp(argv[k+1],"crf"))
				toolboxType = TOOLBOX_CRF;
			else if(!strcmp(argv[k+1],"hcrf"))
				toolboxType = TOOLBOX_HCRF;
			else if(!strcmp(argv[k+1],"ldcrf") || !strcmp(argv[k+1],"fhcrf"))
				toolboxType = TOOLBOX_LDCRF;
			else if(!strcmp(argv[k+1],"ghcrf"))
				toolboxType = TOOLBOX_GHCRF;
			else if(!strcmp(argv[k+1],"sdcrf"))
				toolboxType = TOOLBOX_SDCRF;
			k++;
		}
		else if(argv[k][1] == 'p')
		{
			debugLevel = atoi(argv[++k]);
		}
		else if(argv[k][1] == 'i')
		{
			maxIt = atoi(argv[++k]);
		}
		else if(argv[k][1] == 'h')
		{
			nbHiddenStates = atoi(argv[++k]);
		}
		else if(argv[k][1] == 'w')
		{
			windowSize = atoi(argv[++k]);
		}
		else if(argv[k][1] == 's')
		{
			if(argv[k][2] == '1')
			{
				regFactorL1 = atof(argv[++k]);
			}
			else
			{
				regFactorL2 = atof(argv[++k]);
			}
		}
		else if(argv[k][1] == 'R') {
			initWeightRangeMin = atof(argv[++k]);
			initWeightRangeMax = atof(argv[++k]);
		}
		else if(argv[k][1] == 'P'){
#ifdef _OPENMP
			omp_set_num_threads(atoi(argv[++k]));
#else
			cerr<<"No OpenMP support";
#endif
		}
		else usage(argv);
    }

	if(mode == 0)
		usage(argv);

	if(mode & MODE_TRAIN || mode & MODE_TEST || mode & MODE_VALIDATE)
	{
		if(toolboxType == TOOLBOX_HCRF)
			toolbox = new ToolboxHCRF(nbHiddenStates, opt, windowSize);
		else if(toolboxType == TOOLBOX_LDCRF)
			toolbox = new ToolboxLDCRF(nbHiddenStates, opt, windowSize);
		else if(toolboxType == TOOLBOX_GHCRF)
			toolbox = new ToolboxGHCRF(nbHiddenStates, opt, windowSize);
		#ifndef _PUBLIC
		else if (toolboxType == TOOLBOX_SDCRF)
			toolbox = new ToolboxSharedLDCRF(nbHiddenStates, opt, windowSize);
		#endif
		else
			toolbox = new ToolboxCRF(opt, windowSize);
		toolbox->setDebugLevel(debugLevel);
	}
	// TODO: Implement the validate function in Toolbox
	if(mode & MODE_VALIDATE)
	{
		cout << "Reading training set..." << endl;
		DataSet dataTrain;
		if(toolboxType == TOOLBOX_HCRF || toolboxType == TOOLBOX_GHCRF )
			dataTrain.load((char*)filenameDataTrain.c_str(),NULL, (char*)filenameSeqLabelsTrain.c_str());
		else
			dataTrain.load((char*)filenameDataTrain.c_str(),(char*)filenameLabelsTrain.c_str());

		cout << "Reading validation set..." << endl;
		DataSet dataValidate;
		if(toolboxType == TOOLBOX_HCRF || toolboxType == TOOLBOX_GHCRF )
			dataValidate.load((char*)filenameDataValidate.c_str(),NULL, (char*)filenameSeqLabelsValidate.c_str());
		else
			dataValidate.load((char*)filenameDataValidate.c_str(),(char*)filenameLabelsValidate.c_str());

		if(maxIt >= 0)
			toolbox->setMaxNbIteration(maxIt);

		cout << "Starting validation ..." << endl;
		toolbox->validate(dataTrain, dataValidate, regFactorL2,(char*)filenameStats.c_str());
	}
	if(mode & MODE_TRAIN)
	{
		cout << "Reading training set..." << endl;
		DataSet data;
		const char* fileData = filenameDataTrain.empty()?0:filenameDataTrain.c_str();
		const char* fileDataSparse = filenameDataTrainSparse.empty()?0:filenameDataTrainSparse.c_str();
		if(toolboxType == TOOLBOX_HCRF || toolboxType == TOOLBOX_GHCRF )
			data.load(fileData,NULL, (char*)filenameSeqLabelsTrain.c_str(),NULL,NULL,fileDataSparse);
		else
			data.load(fileData,(char*)filenameLabelsTrain.c_str(),NULL,NULL,NULL,fileDataSparse);

		if(maxIt >= 0)
			toolbox->setMaxNbIteration(maxIt);
		if(regFactorL2 >= 0)
			toolbox->setRegularizationL2(regFactorL2);
		if(regFactorL1 >= 0)
			toolbox->setRegularizationL1(regFactorL1);
		toolbox->setRangeWeights(initWeightRangeMin,initWeightRangeMax);
		toolbox->setWeightInitType(InitMode);
		// Modified by Hugues Salamin 07-16-09. To compare CRF and LDCRF with one hidden state. Looking at value of
		// gradient and function. Uncomment if you want same starting point.
		// toolbox->setWeightInitType(INIT_ZERO);
		cout << "Starting training ..." << endl;
		if(bContinueTraining)
		{
			toolbox->load((char*)filenameModel.c_str(),(char*)filenameFeatures.c_str());
			toolbox->train(data,false);
		}
		else
		{
			toolbox->train(data,true);
		}
		toolbox->save((char*)filenameModel.c_str(),(char*)filenameFeatures.c_str());
	}
	if(mode & MODE_TEST)
	{
		cout << "Reading testing set..." << endl;
		DataSet data;
		if(toolboxType == TOOLBOX_HCRF || toolboxType == TOOLBOX_GHCRF )
			data.load((char*)filenameDataTest.c_str(),NULL,(char*)filenameSeqLabelsTest.c_str());
		else
			data.load((char*)filenameDataTest.c_str(),(char*)filenameLabelsTest.c_str());

		ofstream fileStats1 ((char*)filenameStats.c_str());
		if (fileStats1.is_open())
		{
			fileStats1 << endl << endl << "TESTING DATA SET" << endl << endl;
			fileStats1.close();
		}
		cout << "Starting testing ..." << endl;
		toolbox->load((char*)filenameModel.c_str(),(char*)filenameFeatures.c_str());
		toolbox->test(data,(char*)filenameOutput.c_str(),(char*)filenameStats.c_str());
		if(bTestOnTrainingSet)
		{
			ofstream fileStats ((char*)filenameStats.c_str(), ios_base::out | ios_base::app);
			if (fileStats.is_open())
			{
				fileStats << endl << endl << "TRAINING DATA SET" << endl << endl;
				fileStats.close();
			}
/*			ofstream fileOutput ((char*)filenameOutput.c_str(), ios_base::out | ios_base::app);
			if (fileOutput.is_open())
			{
				fileOutput << endl << endl << "TRAINING DATA SET" << endl << endl;
				fileOutput.close();
			}
*/			cout << "Reading training set..." << endl;
			DataSet dataTrain((char*)filenameDataTrain.c_str(),(char*)filenameLabelsTrain.c_str(), (char*)filenameSeqLabelsTrain.c_str());

			cout << "Starting testing ..." << endl;
			toolbox->test(dataTrain,NULL,(char*)filenameStats.c_str());
		}
		if(bTestOnTrainingSet == 2)
		{
			ofstream fileStats ((char*)filenameStats.c_str(), ios_base::out | ios_base::app);
			if (fileStats.is_open())
			{
				fileStats << endl << endl << "VALIDATION DATA SET" << endl << endl;
				fileStats.close();
			}
/*			ofstream fileOutput ((char*)filenameOutput.c_str(), ios_base::out | ios_base::app);
			if (fileOutput.is_open())
			{
				fileOutput << endl << endl << "TRAINING DATA SET" << endl << endl;
				fileOutput.close();
			}
*/			cout << "Reading validation set..." << endl;
			DataSet dataValidate((char*)filenameDataValidate.c_str(),(char*)filenameLabelsValidate.c_str(), (char*)filenameSeqLabelsValidate.c_str());

			cout << "Starting testing ..." << endl;
			toolbox->test(dataValidate,NULL,(char*)filenameStats.c_str());
		}
	}

	if(toolbox)
		delete toolbox;
	
	cout << "Press a key to continue..." << endl;
#ifdef WIN32
	_getch();
#endif

	return 0;
}

