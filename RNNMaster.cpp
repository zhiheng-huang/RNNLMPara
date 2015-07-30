
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <set>
#include "Utils.h"
//#include <Windows.h>
#include <stdio.h>
#include <cfloat>
#include <unistd.h>

#include "RNNMaster.h"
#include "CommandRunner.h"

RNNMaster::RNNMaster(Parameters &paras)
{
    parameters = &paras;
    batchNum = parameters->getParaInt("batchNum", 10);
    rnnModelFile = parameters->getPara("rnnlm_file");
    trainFile = parameters->getPara("train_file");
    wordCounts = (int *)calloc(batchNum, sizeof(int));
    for (int i = 0; i < batchNum; i++) {
        wordCounts[i] = 0;
    }
}

//partition the training file to sub training files
void RNNMaster::partitionAndDispatch(Vocab *vocab, int iteration, bool use_hpc, FILE *logger)
{
    double percentage = parameters->getParaDouble("percentage", 0.2);
    ifstream reader(trainFile);
    string line;
    string token;
    vector<vector<int> > queries;
    for (int i = 0; i < batchNum; i++) {
        wordCounts[i] = 0;
    }
    while(getline(reader, line))
    {
        stringstream ss(line);
        vector<int> tokens;
        while (ss >> token) {
            tokens.push_back(vocab->getWordId(token));
        }
        queries.push_back(tokens);
    }
    reader.close();

    std::random_shuffle(queries.begin(), queries.end());
    size_t s = queries.size() / batchNum + 1;
    int ns = queries.size() * percentage;
    for (int i = 0; i < batchNum; i++)
    {
        std::stringstream sstm;
        sstm << trainFile << i;		
        string fn = sstm.str();
        ofstream writer(fn);
        int start = i * s;
        if (i == batchNum - 1)
        {
            start = i * s - (ns - s);
            if (start < 0) start = 0;
        }
        int end = start + ns;
        if (end > queries.size())
        {
            end = queries.size();
        }
        for (int j = start; j < end; j++)
        {
            for (int k = 0; k < queries[j].size(); k++)
            {
                //fprintf(batchTrains[i], "%d", sentences[j][k]);
                writer << queries[j][k];
                if (k != queries[j].size() - 1)
                {
                    writer << " ";
                }
                else
                {
                    writer << "\n";
                }
                wordCounts[i]++;
            }
            wordCounts[i]++; //add one for </s> tag
        }
        writer.close();
        //submit job
        dispatchSlaveTrain(iteration, i, use_hpc, logger);
    }
}

//Save master RNN model and generate copies for slave ones with different 
//training data
void RNNMaster::saveMasterRNNModel(RNN &model)
{
    std::stringstream sstm;
    sstm << rnnModelFile << "Iter" << model.iter;		
    string fn = sstm.str();
    model.saveNet(fn, true);
}


//call RNN slaves to train rnn models in parallel. 
void RNNMaster::dispatchSlaveTrain(int iteration, int batchId, bool use_hpc, FILE* logger)
{
    string jobQueue = " ";
    //generate random number between 1 and 136+29=165
    int n = rand() % 165 + 1;
    if(n > 136) jobQueue = " /jobtemplate:Express ";

    //VILFBLHPCHNC003.northamerica.corp.microsoft.com
    //RNNLM slave rnnModel batchTrainFile batchWordCount batchRnnFile
    std::stringstream sstm;
    sstm << "job submit /scheduler:VILFBLHPCHNC003.NORTHAMERICA.CORP.MICROSOFT.COM /memorypernode:10000 /jobtemplate:Express /exclusive:true /jobname:sRNN" << iteration << "-" << batchId;
    //sstm << "job submit /scheduler:VILFBLHPCHNC003.NORTHAMERICA.CORP.MICROSOFT.COM /memorypernode:5000 /jobtemplate:Express /jobname:sRNN" << iteration << "-" << batchId;
    //sstm << "job submit /scheduler:VILFBLHPCHNC003.NORTHAMERICA.CORP.MICROSOFT.COM /memorypernode:5000 /exclusive:true /jobname:sRNN" << iteration << "-" << batchId;
    //sstm << "job submit /scheduler:VILFBLHPCHNC003.NORTHAMERICA.CORP.MICROSOFT.COM /memorypernode:5000 /jobname:sRNN" << iteration << "-" << batchId;
    //sstm << "job submit /scheduler:VILFBLHPCHNC003.NORTHAMERICA.CORP.MICROSOFT.COM /memorypernode:5000" << jobQueue << "/exclusive:true /jobname:sRNN" << iteration << "-" << batchId;
    string hpcPrefix = sstm.str();
    string slave_bin = parameters->getPara("slave_bin");
    string cmd = slave_bin + " slave";
    sstm.str("");
    sstm << rnnModelFile << "Iter" << iteration;
    string rnnModel = sstm.str();
    sstm << "Batch" << batchId;
    string batchRnnFile = sstm.str();
    sstm.str("");
    sstm << trainFile << batchId;
    string batchTrainFile = sstm.str();
    sstm.str("");
    sstm << cmd << " " << rnnModel << " " << batchTrainFile << " " << wordCounts[batchId] << " " << batchRnnFile;		
    string command = sstm.str();
    std::replace(command.begin(), command.end(), '/', '\\');
    if(use_hpc) {
        command = hpcPrefix + " " + command;
    } 
    //command += " 2>&1"; //redirect stderr
    fprintf(logger, "submit command:%s\n", command.c_str());
    CommandRunner::exec(command.c_str());
}

//Test if all slave RNN are done with training
bool RNNMaster::finisedSlaveRnnsTrain(int iteration)
{
    set<int> finished;
    for (int i = 0; i < batchNum; i++)
    {
        std::stringstream sstm;
        sstm << rnnModelFile << "Iter" << iteration << "Batch" << i << "Log";		
        string rnnFileSlaveLog = sstm.str();
        ifstream file(rnnFileSlaveLog);
        sstm.str("");
        sstm << DONE_STR << iteration;
        if (file.good() && Utils::ContainsStr(rnnFileSlaveLog, sstm.str()))
        {
            finished.insert(i);
        }
    }
    if (finished.size() == batchNum)
    {
        return true;
    }
    else
    {
        return false;
    }
}

//average all slave RNN model files to generate master RNN model. 
//Note the head is copied from the previous master RNN model. Note
//only preAveModel contains the momentum info as RNN constructor
//doesnt copy those
RNN* RNNMaster::averageRnnModels(RNN &preAveModel, FILE* logger, double tolerance, bool master_para_adapt, double learningRate, double beta)
{
    RNN* modelSlaveAve = new RNN(preAveModel, false);
    modelSlaveAve->initNet(false);
    int modelUsed = 0;
    for (int i = 0; i < batchNum; i++)
    {
        std::stringstream sstm;
        sstm << rnnModelFile << "Iter" << preAveModel.iter << "Batch" << i;	
        string slaveModelFile = sstm.str();
        string slaveModelLogFile = slaveModelFile + "Log";
        vector<double> logPs;
        Utils::getLogPs(slaveModelFile, logPs);
        double llogpValid = logPs[0];
        double logpValid = logPs[1];
        fprintf(logger, "***** Reading %d-th slave RNN model logp:%f llogp:%f ratio:%f\n", i, logpValid, llogpValid, logpValid/llogpValid);		
        if (logpValid < llogpValid * tolerance) //allow some space to go worse
        {
            fprintf(logger, "reject slave RNN model %d\n", i);
        } else {
            RNN* model = new RNN(slaveModelFile, false); //do not read vocab info
            modelUsed++;
            modelSlaveAve->add(*model);
            delete model;            
        }
        remove(slaveModelFile.c_str());
        remove(slaveModelLogFile.c_str());
        fflush(logger);	
    }
    fprintf(logger, "***** %d slave RNN models are used with tolenrance %f\n", modelUsed, tolerance);    
    fflush(logger);
    RNN* bestAveModel = new RNN(preAveModel, true); //keep the master head info
    if(modelUsed == 0) {
        delete modelSlaveAve;
        return bestAveModel;
    }
    fprintf(logger, "***** Start dividing master RNN model\n");
    modelSlaveAve->divide(modelUsed);
    fprintf(logger, "***** Start updating master RNN model\n");
    fflush(logger);
    
    if(master_para_adapt) {
        double learningRateCands [] = {0.2, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8};
        //double betaCands [] = {0.0000001, 0.000001, 0.00001, 0.0001, 0.001};
        //double momentumCands [] = {0.1, 0.2, 0.3, 0.4, 0.5};
        //double learningRateCands [] = {0.2, 0.4, 0.6, 0.8, 1.0};
        double betaCands [] = {0.0000001, 0.000001, 0.00001, 0.0001, 0.001};
        double momentumCands [] = {0.02, 0.05, 0.1};

        double maxLogp = -DBL_MAX;
        double bestRate = -1;
        double bestRegu = 0;
        //double bestMomentum = 0;
        //learning rate
        int size = sizeof learningRateCands/sizeof(double);
        for( int j = 0; j < size; j++) {
            RNN* modelAve = new RNN(preAveModel, true); //keep the master head info
            modelAve->update(*modelSlaveAve, preAveModel, learningRateCands[j], 0);
            fprintf(logger, "\n\tlearning rate = %f\n", learningRateCands[j]);
            double logp = modelAve->evaluateNet(logger);
            if(maxLogp < logp) {
                maxLogp = logp;
                bestRate = learningRateCands[j];
            }
            delete modelAve;
        }
        fprintf(logger, "**** best learning rate: %f\n", bestRate);
        //regularization
        size = sizeof betaCands/sizeof(double);
        for( int j = 0; j < size; j++) {
            RNN* modelAve = new RNN(preAveModel, true); //keep the master head info
            modelAve->update(*modelSlaveAve, preAveModel, bestRate, betaCands[j]);
            fprintf(logger, "\n\tregularization = %f\n", betaCands[j]);
            double logp = modelAve->evaluateNet(logger);
            if(maxLogp < logp) {
                maxLogp = logp;
                bestRegu = betaCands[j];
            }
            delete modelAve;
        }
        fprintf(logger, "**** best regularization: %f\n", bestRegu);
        //momentum
        //size = sizeof momentumCands/sizeof(double);
        //for( int j = 0; j < size; j++) {
        //    RNN* modelAve = new RNN(preAveModel, true); //keep the master head info
        //    modelAve->update(*modelSlaveAve, preAveModel, bestRate, bestRegu, momentumCands[j]);
        //    fprintf(logger, "\n\tmomentum = %f\n", momentumCands[j]);
        //    double logp = modelAve->evaluateNet(logger);
        //    if(maxLogp < logp) {
        //        maxLogp = logp;
        //        bestMomentum = momentumCands[j];
        //    }
        //    delete modelAve;
        //}
        //fprintf(logger, "**** best momentum: %f\n", bestMomentum);
        fprintf(logger, "***** best learning rate:%f, best regularization:%f\n", bestRate, bestRegu);		
        bestAveModel->update(*modelSlaveAve, preAveModel, bestRate, bestRegu);
    } else {		
        bestAveModel->update(*modelSlaveAve, preAveModel, learningRate, beta);
    }
    delete modelSlaveAve;
    return bestAveModel;
}

void RNNMaster::master(string paraFile)
{	
    Parameters parameters(paraFile);
    int debug_mode = parameters.getParaInt("debug_mode", 0);
    //master training
    if (parameters.getParaBool("train_model")) {
        RNNMaster rnnPara(parameters);
        double masterMinImprovement = parameters.getParaDouble("masterMinImprovement", 1.0002);
        double tolerance = parameters.getParaDouble("tolerance", 1.1);
        bool master_para_adapt = parameters.getParaBool("master_para_adapt");
        double learningRate = parameters.getParaDouble("master_learning_rate", 1);
        double regularization = parameters.getParaDouble("master_regularization", 0);
        //double momentum = parameters.getParaDouble("master_momentum", 0);

        bool use_hpc = parameters.getParaBool("use_hpc");
        string logName = parameters.getPara("rnnlm_file") + "Log";
        FILE *masterLogger=fopen(logName.c_str(), "ab");
        setbuf(masterLogger, NULL);

        RNN *preAverageRNN = NULL;
        RNN *averageRNN = NULL;				

        while (true) //iteration loop
        {			
            if (preAverageRNN == NULL)
            {
                ifstream file(parameters.getPara("rnnlm_file"));
                if (file.good())
                {
                    fprintf(masterLogger, "*** Start loading previous model %s\n", parameters.getPara("rnnlm_file").c_str());
                    averageRNN = new RNN(parameters.getPara("rnnlm_file"), true);
                    fprintf(masterLogger, "*** Done loading previous model\n");
                }
                else
                {
                    //warm start
                    bool warm_start = parameters.getParaBool("warm_start");
                    if(warm_start) {
                        fprintf(masterLogger, "*** Start training one-iter model for warm start\n");
                        averageRNN = new RNN(parameters, true);	
                        averageRNN->one_iter = true;
                        averageRNN->train_file = parameters.getPara("warm_start_train_file");
                        averageRNN->trainNet(false, parameters.getPara("rnnlm_file") + "OneIterLog");
                        averageRNN->train_file = parameters.getPara("train_file"); //recover original train file
                        fprintf(masterLogger, "*** Done training one-iter model\n");
                    } else {
                        fprintf(masterLogger, "*** Start creating a random RNN model\n");
                        bool startRandom = parameters.getParaBool("random_start");
                        averageRNN = new RNN(parameters, startRandom);
                        fprintf(masterLogger, "*** Done creating a random RNN model\n");
                    }
                }
            }
            else
            {
                fprintf(masterLogger, "*** Start partitioning data and submit HPC jobs\n");
                //iteration number keeps the same after slave RNN run
                rnnPara.partitionAndDispatch(preAverageRNN->vocab, preAverageRNN->iter, use_hpc, masterLogger);
                fprintf(masterLogger, "*** Done partitioning data and submit HPC jobs\n");
                //wait for the slave RNN training done
                while (true)
                {
                    sleep(2000);
                    if (rnnPara.finisedSlaveRnnsTrain(preAverageRNN->iter))
                    {
                        fprintf(masterLogger, "*** Collected all slave RNN models\n");
                        break;
                    }
                }
                fprintf(masterLogger, "*** Start model average\n");
                //Get averaged RNN model from slave RNN model files
                averageRNN = rnnPara.averageRnnModels(*preAverageRNN, masterLogger, tolerance, master_para_adapt, learningRate, regularization);
                fprintf(masterLogger, "*** Done model average\n");
                //try different paras to average models
            }
            fprintf(masterLogger, "\n*** Evaluate averaged model on validation data in iter %d\n", averageRNN->iter);
            averageRNN->evaluateNet(masterLogger);			
            fprintf(masterLogger, "\n*** Evaluate averaged model on test data in iter %d\n", averageRNN->iter);
            averageRNN->testNet(parameters.getPara("test_file"), true, 0, masterLogger, debug_mode);
            fflush(masterLogger);

            if (preAverageRNN != NULL)
            {				
                averageRNN->iter++;
                if (averageRNN->logpValid * masterMinImprovement < preAverageRNN->logpValid)
                {
                    if (averageRNN->alpha_divide == false)
                    {
                        averageRNN->alpha_divide = true;
                    }
                    else
                    {
                        break;
                    }
                }
                if (averageRNN->alpha_divide == true)
                {
                    averageRNN->alpha /= 2;
                }
                if (averageRNN->logpValid < preAverageRNN->logpValid)
                {
                    averageRNN->copyNet(*preAverageRNN);
                    averageRNN->logpValid = preAverageRNN->logpValid;
                }
               /* fprintf(masterLogger, "*** Compute averaged model momentum\n");
                averageRNN->computeMomentum(*preAverageRNN);*/
            }
            fprintf(masterLogger, "*** Start saving master RNN model\n");
            rnnPara.saveMasterRNNModel(*averageRNN);
            fprintf(masterLogger, "*** Done saving master RNN model\n");
            fprintf(masterLogger, "----------------------------------------- \n\n");
            delete preAverageRNN;
            preAverageRNN = averageRNN;
        } //end iteration loop

        if (averageRNN->logpValid < preAverageRNN->logpValid)
        {
            preAverageRNN->saveNet(parameters.getPara("rnnlm_file"), true);
            fprintf(masterLogger, "*** save 2nd last average RNN model and exit");
        }
        else
        {
            averageRNN->saveNet(parameters.getPara("rnnlm_file"), true);
            fprintf(masterLogger, "*** save last average RNN model and exit");
        }
        fclose(masterLogger);
    }

    //test
    if (parameters.getParaBool("test_model"))
    {
        string logName = parameters.getPara("rnnlm_file") + "Log";
        FILE *masterLogger=fopen(logName.c_str(), "ab");
        RNN model1(parameters.getPara("rnnlm_file"), true);
        model1.testNet(parameters.getPara("test_file"), parameters.getParaBool("replace"), parameters.getParaDouble("unk_penalty", 0), masterLogger, debug_mode);
        fclose(masterLogger);
    }
}

//temp code
void RNNMaster::masterTemp(string paraFile)
{	
    Parameters parameters(paraFile);
    int debug_mode = parameters.getParaInt("debug_mode", 0);

    RNNMaster rnnPara(parameters);
    string logName = parameters.getPara("rnnlm_file") + "Log";
    FILE *masterLogger=fopen(logName.c_str(), "ab");
    setbuf(masterLogger, NULL);

    RNN *preAverageRNN = new RNN(parameters.getPara("rnnlm_file"), true);
    fprintf(masterLogger, "\n*** Evaluate warm-start model on validation data \n");
    preAverageRNN->evaluateNet(masterLogger);	
    RNN *averageRNN = rnnPara.averageRnnModels(*preAverageRNN, masterLogger, 1.0, false, 0.4, 0); //0.8, 0, 0.2		
    fprintf(masterLogger, "\n*** Evaluate averaged model on validation data in iter %d\n", averageRNN->iter);
    averageRNN->evaluateNet(masterLogger);			
    fprintf(masterLogger, "\n*** Evaluate averaged model on test data in iter %d\n", averageRNN->iter);
    averageRNN->testNet(parameters.getPara("test_file"), true, 0, masterLogger, debug_mode);
    fflush(masterLogger);
}

//temp code, to tune the reject parameter
void RNNMaster::masterTemp2(string paraFile)
{	
    Parameters parameters(paraFile);
    int debug_mode = parameters.getParaInt("debug_mode", 0);

    RNNMaster rnnPara(parameters);
    string logName = parameters.getPara("rnnlm_file") + "Log";
    FILE *masterLogger=fopen(logName.c_str(), "ab");
    setbuf(masterLogger, NULL);

    RNN *preAverageRNN = new RNN(parameters.getPara("rnnlm_file"), true);
    fprintf(masterLogger, "\n*** Evaluate warm-start model on validation data \n");
    preAverageRNN->evaluateNet(masterLogger);
    double rejectRates [] = {1.005, 1.02, 1.04};
    int size = sizeof rejectRates/sizeof(double);
    for(int i = 0; i < size; i++) {
        fprintf(masterLogger, "\n*** ======================================= reject rate %f\n", rejectRates[i]);
        RNN *averageRNN = rnnPara.averageRnnModels(*preAverageRNN, masterLogger, rejectRates[i], true, 0, 0);
        fprintf(masterLogger, "\n*** Evaluate averaged model on validation data in iter %d\n", averageRNN->iter);
        averageRNN->evaluateNet(masterLogger);			
        fprintf(masterLogger, "\n*** Evaluate averaged model on test data in iter %d\n", averageRNN->iter);
        averageRNN->testNet(parameters.getPara("test_file"), true, 0, masterLogger, debug_mode);
        fflush(masterLogger);
        std::stringstream sstm;
        sstm << parameters.getPara("rnnlm_file") << "Iter" << (averageRNN->iter+1) << "R" << rejectRates[i];		
        string fn = sstm.str();
        averageRNN->saveNet(fn, true);
        delete averageRNN;
    }
    delete preAverageRNN;
}

//train one more iteration for a given rnn model
void RNNMaster::slave(string rnnFile, string batchTrainFile, int batchWordCount, string batchRnnFile)
{		
    RNN *model = new RNN(rnnFile, true);
    model->train_file = batchTrainFile;
    model->train_words = batchWordCount;
    model->one_iter = true;	
    model->trainNet(true, batchRnnFile + "Log");
    model->saveNet(batchRnnFile, false); //save net after one iteration
    //caution, mark the model ready only after saving it! The saving takes times.
    ofstream writer(batchRnnFile + "Log", ios::app);
    writer << DONE_STR << model->iter <<"\n\n";
    writer.close();
    delete model;
    model = NULL;
}

